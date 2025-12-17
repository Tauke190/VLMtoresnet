#
# For acknowledgement see accompanying ACKNOWLEDGEMENTS file.
# Copyright (C) 2023 Apple Inc. All rights reserved.
#

""" ImageNet Validation Script

This is intended to be a lean and easily modifiable ImageNet validation script for evaluating pretrained
models or training checkpoints against ImageNet or similarly organized image datasets. It prioritizes
canonical PyTorch, standard Python style, and good performance. Repurpose as you see fit.

Hacked together by Ross Wightman (https://github.com/rwightman)
"""
import os
import csv
import glob
import time
import logging
import sys
import copy
from pathlib import Path
from collections import OrderedDict
from contextlib import suppress

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel

from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
import clip


from timm.layers import apply_test_time_pool
from timm.models import create_model, load_checkpoint, is_model, list_models
from timm.data import (
    create_dataset,
    create_loader,
    resolve_data_config,
    RealLabelsImagenet,
)
from timm.utils import (
    accuracy,
    AverageMeter,
    natural_key,
    setup_default_logging,
    set_jit_legacy,
)

from Functions.validation_arguments import _parse_args

import models
from models.modules.mobileone import reparameterize_model

has_apex = False
try:
    from apex import amp

    has_apex = True
except ImportError:
    pass

has_native_amp = False
try:
    if getattr(torch.cuda.amp, "autocast") is not None:
        has_native_amp = True
except AttributeError:
    pass


from CLIP.dataloaders.aircraft import aircraft as AircraftDataset
from CLIP.dataloaders.food101 import Food101 as Food101Dataset
from CLIP.dataloaders.cars import Cars as StanfordCarsDataset
from CLIP.dataloaders.caltech101 import Caltech101 as Caltech101Dataset
from CLIP.dataloaders.ucf101 import UCF101 as UCF101Dataset
from torchvision.datasets import FER2013, GTSRB, Country211, RenderedSST2, ImageFolder

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger("validate")


def _read_txt(file_path):
    with open(file_path, "r") as f:
        content = f.read()
    content = str(content).split("\n")
    try:
        content.remove("")
    except ValueError:
        pass
    return content


def _build_zeroshot_weights(args, device):
    """Build zero-shot classifier weights using CLIP text encoder and CLIP prompts.

    Uses CLIP/dataloaders/classes/<zeroshot-dataset>.txt and CLIP/dataloaders/templates/<zeroshot-dataset>.txt
    for class names and text templates.
    """
    dataset_key = args.dataset

    repo_root = Path(__file__).resolve().parent
    clip_root = repo_root / "CLIP"

    classes_dir = (
        Path(args.zeroshot_classes_dir)
        if args.zeroshot_classes_dir
        else clip_root / "dataloaders" / "classes"
    )
    templates_dir = (
        Path(args.zeroshot_templates_dir)
        if args.zeroshot_templates_dir
        else clip_root / "dataloaders" / "templates"
    )

    class_file = classes_dir / f"{dataset_key}.txt"
    template_file = templates_dir / f"{dataset_key}.txt"

    if not class_file.is_file():
        raise FileNotFoundError(f"Zero-shot class file not found: {class_file}")
    if not template_file.is_file():
        raise FileNotFoundError(f"Zero-shot template file not found: {template_file}")

    classes = _read_txt(class_file)
    templates = _read_txt(template_file)

    if str(clip_root) not in sys.path:
        sys.path.insert(0, str(clip_root))

    _logger.info(
        "Building zero-shot weights for dataset '%s' with CLIP backbone '%s'",
        dataset_key,
        args.zeroshot_backbone,
    )

    clip_model, preprocess = clip.load(args.zeroshot_backbone, device=device)
    clip_model.eval()

    with torch.no_grad():
        zeroshot_weights = []
        print("\n[Zero-shot] Example text prompts for first 3 classes:")
        for i, classname in enumerate(classes):
            texts = [template.format(classname) for template in templates]
            if i < 3:
                print(f"Class: {classname}")
                for t in texts[:min(3, len(texts))]:
                    print(f"  Prompt: {t}")
            text_tokens = clip.tokenize(texts).to(device)
            class_embeddings = clip_model.encode_text(text_tokens)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding = class_embedding / class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)

    return zeroshot_weights, len(classes)  # Return class count too


def _build_custom_dataset(dataset_name, root, is_train, transform, gtsrb_download=False):
    """Build custom downstream datasets for both linear-probe and zero-shot paths.
    """
    if dataset_name == "fgvc_aircraft":
        return AircraftDataset(root=root, train=is_train, transform=transform)
    if dataset_name == "food101":
        return Food101Dataset(root=root, train=is_train, transform=transform)
    if dataset_name == "cars":
        return StanfordCarsDataset(root=root, train=is_train, transform=transform)
    if dataset_name == "ucf101":
        return UCF101Dataset(root=root, train=is_train, transform=transform)
    if dataset_name == "fer2013":
        split = "train" if is_train else "test"
        return FER2013(root=root, split=split, transform=transform)
    if dataset_name == "gtsrb":
        split = "train" if is_train else "test"
        return GTSRB(root=root, split=split, transform=transform, download=gtsrb_download)
    if dataset_name == "country211":
        split = "train" if is_train else "test"
        return Country211(root=root, split=split, transform=transform, download=False)
    if dataset_name == "sst2":
        split = "train" if is_train else "test"
        return RenderedSST2(root=root, split=split, transform=transform, download=False)
    if dataset_name in ("imagenet", "imagenet1k"):
        split_dir = "train" if is_train else "validation"
        split_root = os.path.join(root, split_dir)
        return ImageFolder(root=split_root, transform=transform)



def _extract_linearprobe_features(model, loader, device):
    """Extract projected image embeddings and labels for linear-probe training/testing.
       in Model returns (projected_embed, logits, feature map) when output is a tuple,
    """
    feats = []
    labels = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            # Some loaders may return (input, target); others could add extra fields.
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                input, target = batch[0], batch[1]
            else:
                continue

            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(input)
            if isinstance(output, (list, tuple)):
                features = output[0] # Projected embeddings
           

            features = features.float()
            feats.append(features.cpu())
            labels.append(target.cpu())

    if not feats:
        raise RuntimeError("No features extracted for linear probe; check dataset/loader.")

    feats = torch.cat(feats, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    return feats, labels


def _linearprobe_eval(args, model, data_config):
    """Linear-probe evaluation on frozen projected embeddings.

    Follows the CLIP official template: extract image features on train/test splits,
    fit a logistic-regression classifier, and report top-1 / top-5 accuracy.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    clip_model, preprocess = clip.load('ViT-L/14', device)

    dataset_name = args.dataset if args.dataset else ""

    # Prefer custom datasets when available; otherwise fall back to timm.create_dataset.
    train_dataset = _build_custom_dataset(
        dataset_name=dataset_name,
        root=args.data,
        is_train=True,
        transform=preprocess,
        gtsrb_download=False,
    )
    eval_dataset = _build_custom_dataset(
        dataset_name=dataset_name,
        root=args.data,
        is_train=False,
        transform=preprocess,
        gtsrb_download=False,
    )

    crop_pct = data_config["crop_pct"]

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    _logger.info("[LinearProbe] Extracting train features...")
    train_features, train_labels = _extract_linearprobe_features(model, train_loader, device)
    _logger.info("[LinearProbe] Extracting eval features...")
    eval_features, eval_labels = _extract_linearprobe_features(model, eval_loader, device)

    _logger.info(
        "[LinearProbe] Training logistic regression (C=%.4f, max_iter=%d, n_train=%d, n_eval=%d)...",
        args.linearprobe_C,
        args.linearprobe_max_iter,
        train_features.shape[0],
        eval_features.shape[0],
    )

    classifier = LogisticRegression(
        random_state=0,
        C=args.linearprobe_C,
        max_iter=args.linearprobe_max_iter,
        verbose=1,
        n_jobs=-1,
    )
    classifier.fit(train_features, train_labels)

    # Top-1 accuracy
    preds = classifier.predict(eval_features)
    top1_acc = float((preds == eval_labels).astype(np.float32).mean() * 100.0)

    # Top-5 accuracy (if there are at least 5 classes)
    if len(classifier.classes_) >= 5:
        probs = classifier.predict_proba(eval_features)
        top5_idx = np.argsort(-probs, axis=1)[:, :5]
        correct_top5 = 0
        for i, label in enumerate(eval_labels):
            if label in top5_idx[i]:
                correct_top5 += 1
        top5_acc = float(correct_top5 / max(len(eval_labels), 1) * 100.0)
    else:
        top5_acc = top1_acc

    _logger.info("[LinearProbe] Top-1: %.3f%%  Top-5: %.3f%%", top1_acc, top5_acc)

    results = OrderedDict(
        top1=round(top1_acc, 4),
        top1_err=round(100.0 - top1_acc, 4),
        top5=round(top5_acc, 4),
        top5_err=round(100.0 - top5_acc, 4),
        img_size=data_config["input_size"][-1],
        cropt_pct=crop_pct,
        interpolation=data_config["interpolation"],
    )
    return results


def validate(args):
    # might as well try to validate something
    args.pretrained = args.pretrained or not args.checkpoint
    args.prefetcher = not args.no_prefetcher
    amp_autocast = suppress  # do nothing
    if args.amp:
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
        else:
            _logger.warning("Neither APEX or Native Torch AMP is available.")
    assert not args.apex_amp or not args.native_amp, "Only one AMP mode should be set."
    if args.native_amp:
        amp_autocast = torch.cuda.amp.autocast
        _logger.info("Validating in mixed precision with native PyTorch AMP.")
    elif args.apex_amp:
        _logger.info("Validating in mixed precision with NVIDIA APEX AMP.")
    else:
        _logger.info("Validating in float32. AMP not enabled.")

    if args.legacy_jit:
        set_jit_legacy()

    # create model (let model use its default num_classes)
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        in_chans=3,
        global_pool=args.gp,
        scriptable=args.torchscript,
        inference_mode=args.use_inference_mode,
    )

    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, args.use_ema)

    # Reparameterize model
    model.eval()
    if not args.use_inference_mode:
        _logger.info("Reparameterizing Model %s" % (args.model))
        model = reparameterize_model(model)
    setattr(model, "pretrained_cfg", model.__dict__["default_cfg"])

    data_config = resolve_data_config(
        vars(args), model=model, use_test_size=True, verbose=True
    )
    test_time_pool = False
    if args.test_pool:
        model, test_time_pool = apply_test_time_pool(
            model, data_config, use_test_size=True
        )

    if args.torchscript:
        torch.jit.optimized_execution(True)
        model = torch.jit.script(model)

    model = model.cuda()
    if args.apex_amp:
        model = amp.initialize(model, opt_level="O1")

    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu)))

    criterion = nn.CrossEntropyLoss().cuda()

    #---------------------------------------------------------------

    # Linear-probe evaluation (separate code path, returns early)
    if args.eval_mode == "linearprobe":
        return _linearprobe_eval(args, model, data_config)
     # Optional zero-shot setup (CLIP text encoder + classifier weights)
    if args.eval_mode == "zeroshot":
        device = torch.device("cuda")
        zeroshot_weights, class_count = _build_zeroshot_weights(args, device=device)
        dataset = _build_custom_dataset(
            dataset_name=args.dataset,
            root=args.data,
            is_train=False,
            transform=None,
            gtsrb_download=True,
        )
        print(f"[Zero-shot] Test dataset size: {len(dataset) if 'dataset' in locals() else 'N/A'}")

 #---------------------------------------------------------------
    # Dataset selection
    if args.eval_mode == "logits":
        # Standard ImageNet-style evaluation (ImageFolder / ImageTar via timm)
        dataset = create_dataset(
            root=args.data,
            name="",
            split=args.split,
            download=args.dataset_download,
            load_bytes=args.tf_preprocessing,
            class_map=args.class_map,
        )
   

    if args.valid_labels:
        with open(args.valid_labels, "r") as f:
            valid_labels = {int(line.rstrip()) for line in f}
            # store as sorted list of indices; no need for num_classes
            valid_labels = sorted(valid_labels)
    else:
        valid_labels = None

    if args.real_labels:
        real_labels = RealLabelsImagenet(
            dataset.filenames(basename=True), real_json=args.real_labels
        )
    else:
        real_labels = None

    crop_pct = 1.0 if test_time_pool else data_config["crop_pct"]
    loader = create_loader(
        dataset,
        input_size=data_config["input_size"],
        batch_size=args.batch_size,
        use_prefetcher=args.prefetcher,
        interpolation=data_config["interpolation"],
        mean=data_config["mean"],
        std=data_config["std"],
        num_workers=args.workers,
        crop_pct=crop_pct,
        pin_memory=args.pin_mem,
        tf_preprocessing=args.tf_preprocessing,
    )

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()
    with torch.no_grad():
        # warmup, reduce variability of first batch time, especially for comparing torchscript vs non
        input = torch.randn(
            (args.batch_size,) + tuple(data_config["input_size"])
        ).cuda()
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)
        model(input)
        end = time.time()
        if args.eval_mode == "zeroshot":
            eval_start = time.time()
        for batch_idx, (input, target) in enumerate(loader):
            if args.no_prefetcher:
                target = target.cuda()
                input = input.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            # compute output
            with amp_autocast():
                raw_output = model(input)

            if args.eval_mode == "zeroshot":
                # Use projected CLIP-space embeddings from the model and classify via similarity to CLIP text embeddings.
                if isinstance(raw_output, tuple):
                    image_features = raw_output[0]

                # normalize features
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )

                image_features = image_features.float()
                zeroshot_weights = zeroshot_weights.float()
                logits = 100.0 * image_features @ zeroshot_weights
                output = logits
                loss = criterion(logits, target)

            if args.eval_mode == "logits":    
                output = raw_output            
                if isinstance(output, tuple):
                    output = output[1]   # The model returns projected embeddings, classification logits, backbone feature map
                if valid_labels is not None:
                    output = output[:, valid_labels]
                loss = criterion(output, target)

            if real_labels is not None:
                real_labels.add_result(output)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output.detach(), target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1.item(), input.size(0))
            top5.update(acc5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % args.log_freq == 0:
                _logger.info(
                    "Test: [{0:>4d}/{1}]  "
                    "Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  "
                    "Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  "
                    "Acc@1: {top1.val:>7.3f} ({top1.avg:>7.3f})  "
                    "Acc@5: {top5.val:>7.3f} ({top5.avg:>7.3f})".format(
                        batch_idx,
                        len(loader),
                        batch_time=batch_time,
                        rate_avg=input.size(0) / batch_time.avg,
                        loss=losses,
                        top1=top1,
                        top5=top5,
                    )
                )
        if args.eval_mode == "zeroshot":
            eval_end = time.time()
            print(f"[Zero-shot] Text-based evaluation time: {eval_end - eval_start:.2f} seconds")

    if real_labels is not None:
        # real labels mode replaces topk values at the end
        top1a, top5a = real_labels.get_accuracy(k=1), real_labels.get_accuracy(k=5)
    else:
        top1a, top5a = top1.avg, top5.avg
    results = OrderedDict(
        top1=round(top1a, 4),
        top1_err=round(100 - top1a, 4),
        top5=round(top5a, 4),
        top5_err=round(100 - top5a, 4),
        img_size=data_config["input_size"][-1],
        cropt_pct=crop_pct,
        interpolation=data_config["interpolation"],
    )

    _logger.info(
        " * Acc@1 {:.3f} ({:.3f}) Acc@5 {:.3f} ({:.3f})".format(
            results["top1"], results["top1_err"], results["top5"], results["top5_err"]
        )
    )

    return results


def main():
    setup_default_logging()
    args, args_text = _parse_args()
    model_cfgs = []
    model_names = []
    if os.path.isdir(args.checkpoint):
        # validate all checkpoints in a path with same model
        checkpoints = glob.glob(args.checkpoint + "/*.pth.tar")
        checkpoints += glob.glob(args.checkpoint + "/*.pth")
        model_names = list_models(args.model)
        model_cfgs = [(args.model, c) for c in sorted(checkpoints, key=natural_key)]
    else:
        if args.model == "all":
            # validate all models in a list of names with pretrained checkpoints
            args.pretrained = True
            model_names = list_models(
                pretrained=True, exclude_filters=["*_in21k", "*_in22k"]
            )
            model_cfgs = [(n, "") for n in model_names]
        elif not is_model(args.model):
            # model name doesn't exist, try as wildcard filter
            model_names = list_models(args.model)
            model_cfgs = [(n, "") for n in model_names]

        if not model_cfgs and os.path.isfile(args.model):
            with open(args.model) as f:
                model_names = [line.rstrip() for line in f]
            model_cfgs = [(n, None) for n in model_names if n]

    if len(model_cfgs):
        results_file = args.results_file or "./results-all.csv"
        _logger.info(
            "Running bulk validation on these pretrained models: {}".format(
                ", ".join(model_names)
            )
        )
        results = []
        try:
            start_batch_size = args.batch_size
            for m, c in model_cfgs:
                batch_size = start_batch_size
                args.model = m
                args.checkpoint = c
                result = OrderedDict(model=args.model)
                r = {}
                while not r and batch_size >= args.num_gpu:
                    torch.cuda.empty_cache()
                    try:
                        args.batch_size = batch_size
                        print("Validating with batch size: %d" % args.batch_size)
                        r = validate(args)
                    except RuntimeError as e:
                        if batch_size <= args.num_gpu:
                            print(
                                "Validation failed with no ability to reduce batch size. Exiting."
                            )
                            raise e
                        batch_size = max(batch_size // 2, args.num_gpu)
                        print("Validation failed, reducing batch size by 50%")
                result.update(r)
                if args.checkpoint:
                    result["checkpoint"] = args.checkpoint
                results.append(result)
        except KeyboardInterrupt as e:
            pass
        results = sorted(results, key=lambda x: x["top1"], reverse=True)
        if len(results):
            write_results(results_file, results)
    else:
        # When --dataset is set to "all", loop over the predefined set of
        # downstream datasets and run evaluation in the *currently selected*
        # eval mode (args.eval_mode) for each, using fixed roots. Otherwise,
        # keep the original single-dataset behaviour.
        if args.dataset == "all":
            multi_datasets = [
                ("fgvc_aircraft", "/mnt/SSD2/fgvc-aircraft-2013b/data"),
                ("food101", "/mnt/SSD2/food-101"),
                ("cars", "/mnt/SSD2/stanford_cars"),
                ("ucf101", "/mnt/SSD2/UCF101_midframes"),
                ("gtsrb", "/mnt/SSD2/gtsrb"),
                ("sst2", "/mnt/SSD2/rendered-sst2"),
                ("imagenet1k", "/mnt/SSD2/ImageNet1k")
            ]

            summary = []
            for dataset_name, dataset_root in multi_datasets:
                run_args = copy.deepcopy(args)
                run_args.dataset = dataset_name
                run_args.data = dataset_root

                _logger.info(
                    "Running %s evaluation on dataset '%s' (data=%s)",
                    run_args.eval_mode,
                    dataset_name,
                    dataset_root,
                )
                result = validate(run_args)
                entry = OrderedDict(dataset=dataset_name, eval_mode=run_args.eval_mode)
                if isinstance(result, dict):
                    entry.update(result)
                summary.append(entry)

            print("\n==== Multi-dataset evaluation summary (dataset=all) ====")
            for r in summary:
                print(
                    f"{r['dataset']:>12} [{r['eval_mode']:^10}]  "
                    f"Top-1: {r.get('top1', float('nan')):.3f}  "
                    f"Top-5: {r.get('top5', float('nan')):.3f}"
                )
        else:
            validate(args)


def write_results(results_file, results):
    with open(results_file, mode="w") as cf:
        dw = csv.DictWriter(cf, fieldnames=results[0].keys())
        dw.writeheader()
        for r in results:
            dw.writerow(r)
        cf.flush()


if __name__ == "__main__":
    main()
