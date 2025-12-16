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
import argparse
import os
import csv
import glob
import time
import logging
import sys
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

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger("validate")


parser = argparse.ArgumentParser(description="PyTorch ImageNet Validation")
parser.add_argument("data", metavar="DIR", help="path to dataset")
## Remove --dataset argument. Use only --zeroshot-dataset for zero-shot mode.
parser.add_argument(
    "--split",
    metavar="NAME",
    default="validation",
    help="dataset split (default: validation)",
)
parser.add_argument(
    "--dataset-download",
    action="store_true",
    default=False,
    help="Allow download of dataset for torch/ and tfds/ datasets that support it.",
)
parser.add_argument(
    "--model",
    "-m",
    metavar="NAME",
    default="dpn92",
    help="model architecture (default: dpn92)",
)
parser.add_argument(
    "-j",
    "--workers",
    default=4,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 2)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=256,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256)",
)
parser.add_argument(
    "--img-size",
    default=None,
    type=int,
    metavar="N",
    help="Input image dimension, uses model default if empty",
)
parser.add_argument(
    "--input-size",
    default=[3, 256, 256],
    nargs=3,
    type=int,
    metavar="N N N",
    help="Input all image dimensions (d h w, e.g. --input-size 3 256 256), uses model default if empty",
)
parser.add_argument(
    "--crop-pct",
    default=None,
    type=float,
    metavar="N",
    help="Input image center crop pct",
)
parser.add_argument(
    "--mean",
    type=float,
    nargs="+",
    default=None,
    metavar="MEAN",
    help="Override mean pixel value of dataset",
)
parser.add_argument(
    "--std",
    type=float,
    nargs="+",
    default=None,
    metavar="STD",
    help="Override std deviation of of dataset",
)
parser.add_argument(
    "--interpolation",
    default="",
    type=str,
    metavar="NAME",
    help="Image resize interpolation type (overrides model)",
)
parser.add_argument(
    "--class-map",
    default="",
    type=str,
    metavar="FILENAME",
    help='path to class to idx mapping file (default: "")',
)
parser.add_argument(
    "--gp",
    default=None,
    type=str,
    metavar="POOL",
    help="Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.",
)
parser.add_argument(
    "--log-freq",
    default=10,
    type=int,
    metavar="N",
    help="batch logging frequency (default: 10)",
)
parser.add_argument(
    "--checkpoint",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "--pretrained", dest="pretrained", action="store_true", help="use pre-trained model"
)
parser.add_argument("--num-gpu", type=int, default=1, help="Number of GPUS to use")
parser.add_argument(
    "--test-pool", dest="test_pool", action="store_true", help="enable test time pool"
)
parser.add_argument(
    "--no-prefetcher",
    action="store_true",
    default=False,
    help="disable fast prefetcher",
)
parser.add_argument(
    "--pin-mem",
    action="store_true",
    default=False,
    help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
)
parser.add_argument(
    "--channels-last",
    action="store_true",
    default=False,
    help="Use channels_last memory layout",
)
parser.add_argument(
    "--amp",
    action="store_true",
    default=False,
    help="Use AMP mixed precision. Defaults to Apex, fallback to native Torch AMP.",
)
parser.add_argument(
    "--apex-amp",
    action="store_true",
    default=False,
    help="Use NVIDIA Apex AMP mixed precision",
)
parser.add_argument(
    "--native-amp",
    action="store_true",
    default=False,
    help="Use Native Torch AMP mixed precision",
)
parser.add_argument(
    "--tf-preprocessing",
    action="store_true",
    default=False,
    help="Use Tensorflow preprocessing pipeline (require CPU TF installed",
)
parser.add_argument(
    "--use-ema",
    dest="use_ema",
    action="store_true",
    help="use ema version of weights if present",
)
parser.add_argument(
    "--torchscript",
    dest="torchscript",
    action="store_true",
    help="convert model torchscript for inference",
)
parser.add_argument(
    "--legacy-jit",
    dest="legacy_jit",
    action="store_true",
    help="use legacy jit mode for pytorch 1.5/1.5.1/1.6 to get back fusion performance",
)
parser.add_argument(
    "--results-file",
    default="",
    type=str,
    metavar="FILENAME",
    help="Output csv file for validation results (summary)",
)
parser.add_argument(
    "--real-labels",
    default="",
    type=str,
    metavar="FILENAME",
    help="Real labels JSON file for imagenet evaluation",
)
parser.add_argument(
    "--valid-labels",
    default="",
    type=str,
    metavar="FILENAME",
    help="Valid label indices txt file for validation of partial label space",
)
parser.add_argument(
    "--use-inference-mode",
    dest="use_inference_mode",
    action="store_true",
    default=False,
    help="use inference mode version of model definition.",
)

parser.add_argument(
    "--eval-mode",
    default="logits",
    choices=["logits", "zeroshot", "linearprobe"],
    help=(
        "Evaluation mode: 'logits' uses classification head, 'zeroshot' uses CLIP text "
        "embeddings, 'linearprobe' trains a logistic-regression head on frozen projected embeddings."
    ),
)
parser.add_argument(
    "--zeroshot-dataset",
    default="",
    type=str,
    help="Dataset identifier for zero-shot prompts (e.g. 'food101'). If empty, falls back to --dataset.",
)
parser.add_argument(
    "--zeroshot-backbone",
    default="ViT-L/14",
    type=str,
    help="CLIP backbone to use for zero-shot text embeddings.",
)
parser.add_argument(
    "--zeroshot-classes-dir",
    default="",
    type=str,
    help="Optional override path to CLIP classes txt directory.",
)
parser.add_argument(
    "--zeroshot-templates-dir",
    default="",
    type=str,
    help="Optional override path to CLIP templates txt directory.",
)

parser.add_argument(
    "--linearprobe-dataset",
    default="",
    type=str,
    help=(
        "Dataset identifier for linear-probe evaluation (passed to timm.create_dataset). "
        "If empty, uses ImageFolder/ImageTar with --data as root."
    ),
)
parser.add_argument(
    "--linearprobe-C",
    default=0.316,
    type=float,
    help="Inverse regularization strength C for logistic regression.",
)
parser.add_argument(
    "--linearprobe-max-iter",
    default=1000,
    type=int,
    help="Maximum iterations for logistic regression optimizer.",
)


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
    dataset_key = args.zeroshot_dataset if args.zeroshot_dataset else "food101"

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

    # Ensure CLIP package (vendored in this repo) is importable as `clip`
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
            class_embeddings = class_embeddings / class_embeddings.norm(
                dim=-1, keepdim=True
            )
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding = class_embedding / class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)

    return zeroshot_weights, len(classes)  # Return class count too


def _extract_linearprobe_features(model, loader, device):
    """Extract projected image embeddings and labels for linear-probe training/testing.

     Model returns (projected_embed, logits, feature map) when output is a tuple,
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


    dataset_name = args.linearprobe_dataset if args.linearprobe_dataset else ""

    # NOTE: root must point to the actual dataset directory on disk.
    # For FGVC-Aircraft this should typically be the "data" folder that
    # contains variants.txt, images_variant_*.txt, and the images/ subdir.
    if dataset_name == "aircraft" or dataset_name == "fgvc_aircraft":
        train_dataset = AircraftDataset(root=args.data, train=True, transform=preprocess)
        eval_dataset = AircraftDataset(root=args.data, train=False, transform=preprocess)
    elif dataset_name == "food101":
        train_dataset = Food101Dataset(root=args.data, train=True, transform=preprocess)
        eval_dataset = Food101Dataset(root=args.data, train=False, transform=preprocess)
    else:
        # Build train and eval datasets via timm's factory following ImageFolder
        train_dataset = create_dataset(
            root=args.data,
            name=dataset_name,
            split="train",
            download=args.dataset_download,
            load_bytes=args.tf_preprocessing,
            class_map=args.class_map,
        )
        eval_dataset = create_dataset(
            root=args.data,
            name=dataset_name,
            split=args.split,
            download=args.dataset_download,
            load_bytes=args.tf_preprocessing,
            class_map=args.class_map,
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
    zeroshot_weights = None
    class_count = None
    if args.eval_mode == "zeroshot":
        device = torch.device("cuda")
        zeroshot_weights, class_count = _build_zeroshot_weights(args, device=device)

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
    if args.eval_mode == "zeroshot":
        # Zero-shot evaluation: allow special-case loaders for datasets that
        # do not follow ImageFolder layout (e.g., FGVC-Aircraft).
        if args.zeroshot_dataset in ["aircraft", "fgvc_aircraft"]:
            dataset = AircraftDataset(root=args.data, train=False, transform=None)      # Use custom Aircraft dataset; let timm.create_loader handle
        if args.zeroshot_dataset in ["food101"]:
            dataset = Food101Dataset(root=args.data, train=False, transform=None)
        if args.zeroshot_dataset in ["stanfordcars"]:
            dataset = StanfordCarsDataset(root=args.data, train=False, transform=None)
        else:
            # For other zero-shot datasets that follow ImageFolder/ImageTar
            # layout, we can still use timm's generic dataset factory.
            dataset = create_dataset(
                root=args.data,
                name=args.zeroshot_dataset,
                split=args.split,
                download=args.dataset_download,
                load_bytes=args.tf_preprocessing,
                class_map=args.class_map,
            )

    if  args.eval_mode == "linearprobe":
        # Zero-shot evaluation: allow special-case loaders for datasets that
        # do not follow ImageFolder layout (e.g., FGVC-Aircraft).
        if args.linearprobe_dataset in ["aircraft", "fgvc_aircraft"]:
            dataset = AircraftDataset(root=args.data, train=False, transform=None)      # Use custom Aircraft dataset; let timm.create_loader handle
        if args.linearprobe_dataset in ["food101"]:
            dataset = Food101Dataset(root=args.data, train=False, transform=None)
        # if args.zeroshot_dataset in ["stanfordcars"]:
            # dataset = StanfordCarsDataset(root=args.data, train=False, transform=None)
        else:
            # For other zero-shot datasets that follow ImageFolder/ImageTar
            # layout, we can still use timm's generic dataset factory.
            dataset = create_dataset(
                root=args.data,
                name=args.zeroshot_dataset,
                split=args.split,
                download=args.dataset_download,
                load_bytes=args.tf_preprocessing,
                class_map=args.class_map,
            )

 #---------------------------------------------------------------
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
                # Use projected CLIP-space embeddings from the model and
                # classify via similarity to CLIP text embeddings.
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
    args = parser.parse_args()
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
