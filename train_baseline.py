#
# For acknowledgement see accompanying ACKNOWLEDGEMENTS file.
# Copyright (C) 2023 Apple Inc. All rights reserved.
#

""" ImageNet Training Script

This is intended to be a lean and easily modifiable ImageNet training script that reproduces ImageNet
training results with some of the latest networks and training techniques. It favours canonical PyTorch
and standard Python style over trying to be able to 'do it all.' That said, it offers quite a few speed
and training result improvements over the usual PyTorch example scripts. Repurpose as you see fit.

This script was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)

NVIDIA CUDA specific speedups adopted from NVIDIA Apex examples
(https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""

import time
import os
import glob
import math
import logging
from contextlib import suppress
from datetime import datetime
from misc import ClipLoss
import clip 
from typing import Union


import torch
import torch.nn as nn
import torchvision.utils
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from timm.data import (
    create_dataset,
    create_loader,
    Mixup,
    FastCollateMixup,
    AugMixDataset,
)
from timm.models import (
    create_model,
    safe_model_name,
    resume_checkpoint,
    load_checkpoint,
)
from timm.layers import convert_splitbn_model
from timm.utils import *
from timm.loss import *
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.utils import ApexScaler, NativeScaler

import models
from misc.distillation_loss import DistillationLoss
from misc.cosine_annealing import CosineWDSchedule

from Functions.train import train_one_epoch
from Functions.eval import run_zeroshot_eval, validate
from Functions.argument import _parse_args
from Functions.setup import (
    build_imagenet_clip_text_features, build_clip_text_features, 
    setup_validation_zeroshot, setup_model, setup_distributed, basic_setup
    )
import pdb


try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model

    has_apex = True
except ImportError:
    has_apex = False


try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False


_logger = logging.getLogger("train")


def main():
    setup_default_logging()
    args, args_text = _parse_args()

    if args.log_wandb:
        if has_wandb:
            wandb.init(project=args.experiment, config=args)
        else:
            _logger.warning(
                "You've requested to log metrics to wandb but package not found. "
                "Metrics not being logged to wandb, try `pip install wandb`"
            )

    
    use_amp = basic_setup(args)

    
    model , num_aug_splits, data_config = setup_model(args)
        
    clip_text_features = None
    clip_loss_fn = None
    clip_logit_scale = None
    
    if args.clip_loss_weight > 0.0:
        clip_model, _ = clip.load("ViT-L/14", device=args.device, jit=False)
        for p in clip_model.parameters():
            p.requires_grad = False
        clip_text_features = build_imagenet_clip_text_features(clip_model, args.device)
        
        if args.local_rank == 0:
            print(f"[DEBUG] CLIP text embeddings created: {clip_text_features.shape}")
        clip_logit_scale = clip_model.logit_scale.exp().detach().to(args.device)

        clip_loss_fn = ClipLoss(
            local_loss=False,
            gather_with_grad=False,
            cache_labels=True,
            rank=args.rank,
            world_size=args.world_size,
        )

        clip_dim = clip_text_features.shape[-1]  # 768 for ViT-L/14

    #-----------------------------------------------------------
    optimizer = create_optimizer_v2(
        list(model.parameters()),
        **optimizer_kwargs(cfg=args),
    )
    ####### Print all optimizer registers params 
    param_to_name = {param: name for name, param in model.named_parameters()}
    for i, param_group in enumerate(optimizer.param_groups):
        _logger.info(f"Parameter group {i}")
        for p in param_group["params"]:
            if p.requires_grad:
                name = param_to_name.get(p, "unnamed")
                _logger.info(f"{name:<80} | {str(p.shape):<40}")


    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == "apex":
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        loss_scaler = ApexScaler()
        if args.local_rank == 0:
            _logger.info("Using NVIDIA APEX AMP. Training in mixed precision.")
    elif use_amp == "native":
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        if args.local_rank == 0:
            _logger.info("Using native Torch AMP. Training in mixed precision.")
    else:
        if args.local_rank == 0:
            _logger.info("AMP not enabled. Training in float32.")

    # optionally resume from a checkpoint
    resume_epoch = None
    if args.resume:
        if not args.finetune:
            # If folder give, pick the last checkpoint from a sorted list of checkpoints
            if os.path.isdir(args.resume):
                ckpt_paths = sorted(glob.glob(os.path.join(args.resume, "*.pth.tar")))
                resume_path = ckpt_paths[-1]
                setattr(args, "resume", resume_path)
                print("Resuming from {}".format(resume_path))

            resume_epoch = resume_checkpoint(
                model,
                args.resume,
                optimizer=None if args.no_resume_opt else optimizer,
                loss_scaler=None if args.no_resume_opt else loss_scaler,
                log_info=args.local_rank == 0,
            )
        else:
            print(
                "Finetune option selected, not loading optimizer state and loss_scaler"
            )
            _ = resume_checkpoint(
                model,
                args.resume,
                optimizer=None,
                loss_scaler=None,
                log_info=args.local_rank == 0,
            )

            data_config["crop_pct"] = 1.0
            print("data config: {}".format(data_config))

    # setup exponential moving average of model weights, SWA could be used here too
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEmaV2(
            model,
            decay=args.model_ema_decay,
            device="cpu" if args.model_ema_force_cpu else None,
        )
        # Do not load EMA model when running in finetuning mode
        if args.resume and not args.finetune:
            print("Loading EMA model")
            load_checkpoint(model_ema.module, args.resume, use_ema=True)

    # setup distributed training
    if args.distributed:
        if has_apex and use_amp == "apex":
            # Apex DDP preferred unless native amp is activated
            if args.local_rank == 0:
                _logger.info("Using NVIDIA APEX DistributedDataParallel.")
            model = ApexDDP(model, delay_allreduce=True)
        else:
            if args.local_rank == 0:
                _logger.info("Using native Torch DistributedDataParallel.")
            # Only wrap model if it has trainable parameters
            if not args.freeze_backbone:
                model = NativeDDP(
                    model,
                    device_ids=[args.local_rank],
                    broadcast_buffers=not args.no_ddp_bb,
                )
        
    # setup learning rate schedule and starting epoch
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    # setup weight decay scheduler
    wd_scheduler = None
    if args.wd_schedule == "cosine":
        wd_scheduler = CosineWDSchedule(
            optimizer=optimizer,
            eta_min=args.weight_decay * 0.1,
            t_max=int(
                args.epochs
                * args.imagenet_trainset_size
                // args.batch_size
                // args.world_size
            ),
        )

    if args.local_rank == 0:
        _logger.info("Scheduled epochs: {}".format(num_epochs))

    # Instantiate teacher model, if distillation is requested.
    teacher_model = None
    if args.distillation_type != "none":
        assert args.teacher_path, "need to specify teacher-path when using distillation"
        print(f"Creating teacher model: {args.teacher_model}")
        teacher_model = create_model(
            args.teacher_model,
            pretrained=False,
            num_classes=args.num_classes,
            global_pool="avg",
        )
        if args.teacher_path.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.teacher_path, map_location="cpu", check_hash=True
            )
        else:
            checkpoint = torch.load(args.teacher_path, map_location="cpu")
        teacher_model.load_state_dict(checkpoint["model"])
        teacher_model.cuda()
        teacher_model.eval()

    # create the train and eval datasets
    dataset_train = create_dataset(
        args.dataset,
        root=args.data_dir,
        split=args.train_split,
        is_training=True,
        class_map=args.class_map,
        download=args.dataset_download,
        batch_size=args.batch_size,
        repeats=args.epoch_repeats,
    )
    dataset_eval = None 
    if args.vanilla_eval:
        dataset_eval = create_dataset(
            args.dataset,
            root=args.data_dir,
            split=args.val_split,
            is_training=False,
            class_map=args.class_map,
            download=args.dataset_download,
            batch_size=args.batch_size,
        )


    # setup mixup / cutmix
    collate_fn = None
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.num_classes,
        )
        if args.prefetcher:
            assert (
                not num_aug_splits
            )  # collate conflict (need to support deinterleaving in collate mixup)
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)

    # wrap dataset in AugMix helper
    if num_aug_splits > 1:
        dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)

    # create data loaders w/ augmentation pipeiine
    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config["interpolation"]
    loader_train = create_loader(
        dataset_train,
        input_size=data_config["input_size"],
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=args.prefetcher,
        no_aug=args.no_aug,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        num_aug_repeats=args.aug_repeats,
        num_aug_splits=num_aug_splits,
        interpolation=train_interpolation,
        mean=data_config["mean"],
        std=data_config["std"],
        num_workers=args.workers,
        distributed=args.distributed,
        collate_fn=collate_fn,
        pin_memory=args.pin_mem,
        use_multi_epochs_loader=args.use_multi_epochs_loader,
        worker_seeding=args.worker_seeding,
    )

    loader_eval = None 
    if args.vanilla_eval:
        loader_eval = create_loader(
            dataset_eval,
            input_size=data_config["input_size"],
            batch_size=args.validation_batch_size or args.batch_size,
            is_training=False,
            use_prefetcher=args.prefetcher,
            interpolation=data_config["interpolation"],
            mean=data_config["mean"],
            std=data_config["std"],
            num_workers=args.workers,
            distributed=args.distributed,
            crop_pct=data_config["crop_pct"],
            pin_memory=args.pin_mem,
        )
    
    # setup loss function
    if args.jsd_loss:
        assert num_aug_splits > 1  # JSD only valid with aug splits set
        train_loss_fn = JsdCrossEntropy(
            num_splits=num_aug_splits, smoothing=args.smoothing
        )
    elif mixup_active:
        # smoothing is handled with mixup target transform which outputs sparse, soft targets
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(target_threshold=args.bce_target_thresh)
        else:
            train_loss_fn = SoftTargetCrossEntropy()
    elif args.smoothing:
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(
                smoothing=args.smoothing, target_threshold=args.bce_target_thresh
            )
        else:
            train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        train_loss_fn = nn.CrossEntropyLoss()
    train_loss_fn = train_loss_fn.cuda()
    validate_loss_fn = nn.CrossEntropyLoss().cuda()

    # -----------------------------------------------------------------
    # Generic zero-shot evaluation setup (CLIP-based)
    zeroshot_eval_ctx = None
    if args.val_set:
        template_file = os.path.join("CLIP", "dataloaders", "templates", f"{args.val_set}.txt")
        zeroshot_eval_ctx = setup_validation_zeroshot(
            validation_dataset=args.val_set,
            validation_root=args.validation_data_dir,
            device=args.device,
            template_file=template_file,
            num_workers=args.workers,
            batch_size=args.validation_batch_size or args.batch_size,
            args=args,
        )
        _logger.info(
            f"Initialized {args.val_set} zero-shot evaluation with "
            f"{len(zeroshot_eval_ctx['class_names'])} classes."
        )

        # --- Run zero-shot validation before training ---
        # run_zeroshot_eval(
        #     zeroshot_eval_ctx,
        #     args,
        #     model,
        #     when="before_train",
        #     epoch=-1,
        #     has_wandb=has_wandb,
        # )
    # -----------------------------------------------------------------
    if args.distillation_type != "none":
        # use distill loss wrapper, which returns base loss when distillation is disabled
        train_loss_fn = DistillationLoss(
            train_loss_fn,
            teacher_model,
            args.distillation_type,
            args.distillation_alpha,
            args.distillation_tau,
        )

    # setup checkpoint saver and eval metric tracking
    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    saver = None
    output_dir = None
    if args.rank == 0:
        if args.experiment:
            exp_name = args.experiment
        else:
            exp_name = "-".join(
                [
                    datetime.now().strftime("%Y%m%d-%H%M%S"),
                    safe_model_name(args.model),
                    str(data_config["input_size"][-1]),
                ]
            )
        output_dir = get_outdir(args.output if args.output else "./output/train", exp_name)
        decreasing = True if eval_metric == "loss" else False
        saver = CheckpointSaver(
            model=model,
            optimizer=optimizer,
            args=args,
            model_ema=model_ema,
            amp_scaler=loss_scaler,
            checkpoint_dir=output_dir,
            recovery_dir=output_dir,
            decreasing=decreasing,
            max_history=args.checkpoint_hist,
        )
        with open(os.path.join(output_dir, "args.yaml"), "w") as f:
            f.write(args_text)

    # ---- Track best zero-shot Acc@1 ----
    best_zeroshot_acc1 = None
    best_zeroshot_epoch = None

    if args.distributed:
        torch.distributed.barrier()

    # pdb.set_trace()
    for epoch in range(start_epoch, num_epochs):
        if args.distributed and hasattr(loader_train.sampler, "set_epoch"):
            loader_train.sampler.set_epoch(epoch)

        train_metrics = train_one_epoch(
            epoch,
            model,
            loader_train,
            optimizer,
            train_loss_fn,
            args,
            lr_scheduler=lr_scheduler,
            saver=saver,
            output_dir=output_dir,
            amp_autocast=amp_autocast,
            loss_scaler=loss_scaler,
            model_ema=model_ema,
            mixup_fn=mixup_fn,
            wd_scheduler=wd_scheduler,
            clip_text_features=clip_text_features,
            clip_logit_scale=clip_logit_scale,
            clip_loss_fn=clip_loss_fn, # Clip loss funciton
            zeroshot_eval_ctx=zeroshot_eval_ctx,
        )

        if args.distributed and args.dist_bn in ("broadcast", "reduce"):
            if args.local_rank == 0:
                _logger.info("Distributing BatchNorm running means and vars")
            distribute_bn(model, args.world_size, args.dist_bn == "reduce")

        if args.vanilla_eval:
            eval_metrics = validate(
                model, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast
            )

        if args.vanilla_eval:
            if model_ema is not None and not args.model_ema_force_cpu:
                if args.distributed and args.dist_bn in ("broadcast", "reduce"):
                    distribute_bn(model_ema, args.world_size, args.dist_bn == "reduce")
                ema_eval_metrics = validate(
                    model_ema.module,
                    loader_eval,
                    validate_loss_fn,
                    args,
                    amp_autocast=amp_autocast,
                    log_suffix=" (EMA)",
                )
                eval_metrics = ema_eval_metrics


        # ---------------- Generic zero-shot evaluation after epochs ----------------
        if args.val_set:
            acc1_zeroshot, acc5_zeroshot = run_zeroshot_eval(
                zeroshot_eval_ctx,
                args,
                model,
                when="epoch",
                epoch=epoch,
                has_wandb=has_wandb,
            )
            eval_metrics = dict(top1=acc1_zeroshot, top5=acc5_zeroshot)

            # Save best backbone + projector based on zero-shot Acc@1
            if acc1_zeroshot is not None:
                if best_zeroshot_acc1 is None or acc1_zeroshot > best_zeroshot_acc1:
                    best_zeroshot_acc1 = acc1_zeroshot
                    best_zeroshot_epoch = epoch

                    if output_dir is not None:
                        # unwrap DDP if needed
                        backbone = model
                        if hasattr(model, "module"):
                            backbone = model.module

                        # Use custom checkpoint name if provided
                        ckpt_name = args.checkpoint_name if args.checkpoint_name else "model_best_zeroshot.pth.tar"
                        model_path = os.path.join(output_dir, ckpt_name)
                        torch.save(
                            {
                                "epoch": epoch,
                                "zeroshot_acc1": acc1_zeroshot,
                                "state_dict": backbone.state_dict(),
                            },
                            model_path,
                        )

                        _logger.info(
                            f"Saved zero-shot-best model at epoch {epoch} "
                            f"(Acc@1={acc1_zeroshot:.2f}%)"
                        )
        # ----------------------------------------------------------------
        if lr_scheduler is not None:
            # step LR for next epoch
            lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

        if output_dir is not None:
            update_summary(
                epoch,
                train_metrics,
                eval_metrics,
                os.path.join(output_dir, "summary.csv"),
                write_header=best_metric is None,
                log_wandb=args.log_wandb and has_wandb,
            )
        
    if best_metric is not None:
        _logger.info("*** Best metric: {0} (epoch {1})".format(best_metric, best_epoch))


if __name__ == "__main__":
    main()
