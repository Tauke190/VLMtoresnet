import argparse
import time
import yaml
import os
import glob
import clip
import logging
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
import torch.nn.functional as F
import torch
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from timm.data import (
    create_dataset,
    create_loader,
    resolve_data_config,
    Mixup,
    FastCollateMixup,
    AugMixDataset,
)
from timm.models import (
    safe_model_name,
    resume_checkpoint,
    model_parameters,
)
from timm.utils import *
from timm.loss import *
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.utils import ApexScaler, NativeScaler

from FastViT_KD import create_fastvit_clip
from eval_zeroshot import prepare_zeroshot_head, evaluate_zero_shot
import sys

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model

    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, "autocast") is not None:
        has_native_amp = True
except AttributeError:
    pass

import sys

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger("train")

# --- Intermediate KD config ---
CLIP_BLOCK_IDX = 18
FINAL_FEATURE_WEIGHT = 1.0
INTERMEDIATE_FEATURE_WEIGHT = 1.0

teacher_feats = {}


def register_clip_hook(module, store_dict, key="clip_block"):
    def hook_fn(_m, _inp, out):
        # CLIP resblocks usually output [seq, B, C]; convert to [B, seq, C]
        if out.dim() == 3 and out.shape[0] > out.shape[1]:
            out = out.permute(1, 0, 2)
        store_dict[key] = out

    return module.register_forward_hook(hook_fn)


import sys


def crash_on_bad_loss(
    loss,
    args,
    batch_idx,
    images,
    student_features,
    teacher_features,
    epoch=None,
):
    """Debug helper: print stats and hard-exit if loss is NaN/Inf."""
    if not (torch.isnan(loss) or torch.isinf(loss)):
        return

    if epoch is not None:
        header = (
            f"\n[CRITICAL FAIL] Rank {args.rank} | Epoch {epoch} | Batch {batch_idx}"
        )
    else:
        header = f"\n[CRITICAL FAIL] Rank {args.rank} | Batch {batch_idx}"

    print(header)
    print(f"Loss value: {loss.item()}")

    # 1. Input stats
    print(
        f"Input: min={images.min().item():.4f}, "
        f"max={images.max().item():.4f}, "
        f"NaN={torch.isnan(images).any().item()}"
    )

    # 2. Student stats
    s_min, s_max = student_features.min().item(), student_features.max().item()
    print(f"Student Features: min={s_min:.4f}, max={s_max:.4f}")
    if torch.isnan(student_features).any():
        print("!!! FAULT: Student output contains NaNs")
    if torch.isinf(student_features).any():
        print("!!! FAULT: Student output contains Infinity")

    # 3. Teacher stats
    t_min, t_max = teacher_features.min().item(), teacher_features.max().item()
    print(f"Teacher Features: min={t_min:.4f}, max={t_max:.4f}")
    if torch.isnan(teacher_features).any():
        print("!!! FAULT: Teacher output contains NaNs")

    sys.stdout.flush()
    sys.exit(1)


# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(
    description="Training Config", add_help=False
)
parser.add_argument(
    "-c",
    "--config",
    default="",
    type=str,
    metavar="FILE",
    help="YAML config file specifying default arguments",
)


parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")

# Dataset parameters
parser.add_argument("data_dir", metavar="DIR", help="path to dataset")
parser.add_argument(
    "--dataset",
    "-d",
    metavar="NAME",
    default="",
    help="dataset type (default: ImageFolder/ImageTar if empty)",
)
parser.add_argument(
    "--train-split",
    metavar="NAME",
    default="train",
    help="dataset train split (default: train)",
)
parser.add_argument(
    "--val-split",
    metavar="NAME",
    default="validation",
    help="dataset validation split (default: validation)",
)
parser.add_argument(
    "--dataset-download",
    action="store_true",
    default=False,
    help="Allow download of dataset for torch/ and tfds/ datasets that support it.",
)
parser.add_argument(
    "--class-map",
    default="",
    type=str,
    metavar="FILENAME",
    help='path to class to idx mapping file (default: "")',
)

# Model parameters
parser.add_argument(
    "--model",
    default="resnet50",
    type=str,
    metavar="MODEL",
    help='Name of model to train (default: "resnet50"',
)
parser.add_argument(
    "--pretrained",
    action="store_true",
    default=False,
    help="Start with pretrained version of specified network (if avail)",
)
parser.add_argument(
    "--initial-checkpoint",
    default="",
    type=str,
    metavar="PATH",
    help="Initialize model from this checkpoint (default: none)",
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="Resume full model and optimizer state from checkpoint (default: none)",
)
parser.add_argument(
    "--no-resume-opt",
    action="store_true",
    default=False,
    help="prevent resume of optimizer state when resuming model",
)
parser.add_argument(
    "--num-classes",
    type=int,
    default=None,
    metavar="N",
    help="number of label classes (Model default if None)",
)
parser.add_argument(
    "--gp",
    default=None,
    type=str,
    metavar="POOL",
    help="Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.",
)
parser.add_argument(
    "--img-size",
    type=int,
    default=None,
    metavar="N",
    help="Image patch size (default: None => model default)",
)
parser.add_argument(
    "--input-size",
    default=None,
    nargs=3,
    type=int,
    metavar="N N N",
    help="Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty",
)
parser.add_argument(
    "--crop-pct",
    default=None,
    type=float,
    metavar="N",
    help="Input image center crop percent (for validation only)",
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
    "-b",
    "--batch-size",
    type=int,
    default=128,
    metavar="N",
    help="input batch size for training (default: 128)",
)
parser.add_argument(
    "-vb",
    "--validation-batch-size",
    type=int,
    default=None,
    metavar="N",
    help="validation batch size override (default: None)",
)

# Optimizer parameters
parser.add_argument(
    "--opt",
    default="adamw",
    type=str,
    metavar="OPTIMIZER",
    help='Optimizer (default: "adamw"',
)
parser.add_argument(
    "--momentum",
    type=float,
    default=0.9,
    metavar="M",
    help="Optimizer momentum (default: 0.9)",
)
parser.add_argument(
    "--weight-decay", type=float, default=0.05, help="weight decay (default: 0.05)"
)
parser.add_argument(
    "--clip-grad",
    type=float,
    default=None,
    metavar="NORM",
    help="Clip gradient norm (default: None, no clipping)",
)
parser.add_argument(
    "--clip-mode",
    type=str,
    default="norm",
    help='Gradient clipping mode. One of ("norm", "value", "agc")',
)


# Learning rate schedule parameters
parser.add_argument(
    "--sched",
    default="cosine",
    type=str,
    metavar="SCHEDULER",
    help='LR scheduler (default: "step"',
)
parser.add_argument(
    "--lr", type=float, default=1e-3, metavar="LR", help="learning rate (default: 1e-3)"
)
parser.add_argument(
    "--warmup-lr",
    type=float,
    default=1e-6,
    metavar="LR",
    help="warmup learning rate (default: 1e-6)",
)
parser.add_argument(
    "--min-lr",
    type=float,
    default=1e-5,
    metavar="LR",
    help="lower lr bound for cyclic schxedulers that hit 0 (1e-5)",
)

parser.add_argument(
    "--epochs",
    type=int,
    default=300,
    metavar="N",
    help="number of epochs to train (default: 300)",
)
parser.add_argument(
    "--epoch-repeats",
    type=float,
    default=0.0,
    metavar="N",
    help="epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).",
)
parser.add_argument(
    "--start-epoch",
    default=None,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "--decay-epochs",
    type=float,
    default=100,
    metavar="N",
    help="epoch interval to decay LR",
)
parser.add_argument(
    "--warmup-epochs",
    type=int,
    default=5,
    metavar="N",
    help="epochs to warmup LR, if scheduler supports",
)
parser.add_argument(
    "--cooldown-epochs",
    type=int,
    default=10,
    metavar="N",
    help="epochs to cooldown LR at min_lr, after cyclic schedule ends",
)
parser.add_argument(
    "--patience-epochs",
    type=int,
    default=10,
    metavar="N",
    help="patience epochs for Plateau LR scheduler (default: 10",
)
parser.add_argument(
    "--decay-rate",
    "--dr",
    type=float,
    default=0.1,
    metavar="RATE",
    help="LR decay rate (default: 0.1)",
)

parser.add_argument(
    "--color-jitter",
    type=float,
    default=0.4,
    metavar="PCT",
    help="Color jitter factor (default: 0.4)",
)
parser.add_argument(
    "--aa",
    type=str,
    default="rand-m9-mstd0.5-inc1",
    metavar="NAME",
    help='Use AutoAugment policy. "v0" or "original". (default: rand-m9-mstd0.5-inc1)',
)
parser.add_argument(
    "--aug-repeats",
    type=int,
    default=0,
    help="Number of augmentation repetitions (distributed training only) (default: 0)",
)


# Augmentation & regularization parameters
parser.add_argument(
    "--no-aug",
    action="store_true",
    default=False,
    help="Disable all training augmentation, override other train aug args",
)
parser.add_argument(
    "--scale",
    type=float,
    nargs="+",
    default=[0.08, 1.0],
    metavar="PCT",
    help="Random resize scale (default: 0.08 1.0)",
)
parser.add_argument(
    "--ratio",
    type=float,
    nargs="+",
    default=[3.0 / 4.0, 4.0 / 3.0],
    metavar="RATIO",
    help="Random resize aspect ratio (default: 0.75 1.33)",
)
parser.add_argument(
    "--hflip", type=float, default=0.5, help="Horizontal flip training aug probability"
)
parser.add_argument(
    "--vflip", type=float, default=0.0, help="Vertical flip training aug probability"
)
parser.add_argument(
    "--reprob",
    type=float,
    default=0.25,
    metavar="PCT",
    help="Random erase prob (default: 0.25)",
)
parser.add_argument(
    "--remode", type=str, default="pixel", help='Random erase mode (default: "pixel")'
)
parser.add_argument(
    "--recount", type=int, default=1, help="Random erase count (default: 1)"
)
parser.add_argument(
    "--resplit",
    action="store_true",
    default=False,
    help="Do not random erase first (clean) augmentation split",
)
parser.add_argument(
    "--mixup",
    type=float,
    default=0.8,
    help="mixup alpha, mixup enabled if > 0. (default: 0.8)",
)
parser.add_argument(
    "--cutmix",
    type=float,
    default=1.0,
    help="cutmix alpha, cutmix enabled if > 0. (default: 1.0)",
)
parser.add_argument(
    "--cutmix-minmax",
    type=float,
    nargs="+",
    default=None,
    help="cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)",
)
parser.add_argument(
    "--mixup-prob",
    type=float,
    default=1.0,
    help="Probability of performing mixup or cutmix when either/both is enabled",
)
parser.add_argument(
    "--mixup-switch-prob",
    type=float,
    default=0.5,
    help="Probability of switching to cutmix when both mixup and cutmix enabled",
)
parser.add_argument(
    "--mixup-mode",
    type=str,
    default="batch",
    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"',
)
parser.add_argument(
    "--smoothing", type=float, default=0.1, help="Label smoothing (default: 0.1)"
)
parser.add_argument(
    "--train-interpolation",
    type=str,
    default="bicubic",
    help='Training interpolation (random, bilinear, bicubic default: "random")',
)

# Misc
parser.add_argument(
    "--seed", type=int, default=42, metavar="S", help="random seed (default: 42)"
)
parser.add_argument(
    "--log-interval",
    type=int,
    default=50,
    metavar="N",
    help="how many batches to wait before logging training status",
)
parser.add_argument(
    "--recovery-interval",
    type=int,
    default=0,
    metavar="N",
    help="how many batches to wait before writing recovery checkpoint",
)
parser.add_argument(
    "--checkpoint-hist",
    type=int,
    default=10,
    metavar="N",
    help="number of checkpoints to keep (default: 10)",
)
parser.add_argument(
    "-j",
    "--workers",
    type=int,
    default=8,
    metavar="N",
    help="how many training processes to use (default: 8)",
)
parser.add_argument(
    "--amp",
    action="store_true",
    default=False,
    help="use NVIDIA Apex AMP or Native AMP for mixed precision training",
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
    "--no-ddp-bb",
    action="store_true",
    default=False,
    help="Force broadcast buffers for native DDP to off.",
)
parser.add_argument(
    "--channels-last",
    action="store_true",
    default=False,
    help="Use channels_last memory layout",
)
parser.add_argument(
    "--pin-mem",
    action="store_true",
    default=False,
    help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
)
parser.add_argument(
    "--no-prefetcher",
    action="store_true",
    default=False,
    help="disable fast prefetcher",
)
parser.add_argument(
    "--output",
    default="",
    type=str,
    metavar="PATH",
    help="path to output folder (default: none, current dir)",
)
parser.add_argument(
    "--experiment",
    default="",
    type=str,
    metavar="NAME",
    help="name of train experiment, name of sub-folder for output",
)
parser.add_argument(
    "--eval-metric",
    default="top1",
    type=str,
    metavar="EVAL_METRIC",
    help='Best metric (default: "top1"',
)
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument(
    "--use-multi-epochs-loader",
    action="store_true",
    default=False,
    help="use the multi-epochs-loader to save time at the beginning of every epoch",
)
parser.add_argument(
    "--torchscript",
    dest="torchscript",
    action="store_true",
    help="convert model torchscript for inference",
)


def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, "r") as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def main():
    setup_default_logging()
    args, args_text = _parse_args()

    args.prefetcher = not args.no_prefetcher

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        # CRITICAL: torchrun puts local_rank in os.environ, not always args
        args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        args.distributed = True
    else:
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0
        args.distributed = False

    args.device = "cuda:%d" % args.local_rank

    if args.distributed:
        # Now this will be 0 for rank 0, and 1 for rank 1
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        _logger.info(
            f"Training in distributed mode: Process {args.rank}/{args.world_size}, Local Rank {args.local_rank}"
        )
    else:
        _logger.info("Training with a single process on 1 GPU.")

    assert args.rank >= 0

    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    if args.amp:
        # `--amp` chooses native amp before apex (APEX ver not actively maintained)
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
    if args.apex_amp and has_apex:
        use_amp = "apex"
    elif args.native_amp and has_native_amp:
        use_amp = "native"
    elif args.apex_amp or args.native_amp:
        _logger.warning(
            "Neither APEX or native Torch AMP is available, using float32. "
            "Install NVIDA apex or upgrade to PyTorch 1.6"
        )

    random_seed(args.seed, args.rank)

    if args.local_rank == 0:
        _logger.info(f"Creating FastViT_CLIP model: {args.model}")

    # Create the student FastViT model
    model = create_fastvit_clip(
        model_name=args.model,
        pretrained=args.pretrained,
        embed_dim=768,  # CLIP L/14 dimension
        lock=True,  # Freeze backbone
    )
    # Load CLIP teacher model
    clip_model, _ = clip.load("ViT-L/14", device="cuda")
    clip_model.eval()  # Always in eval mode
    for param in clip_model.parameters():
        param.requires_grad = False

    # Attach hook to chosen CLIP transformer block (for intermediate KD)
    clip_handle = register_clip_hook(
        clip_model.visual.transformer.resblocks[CLIP_BLOCK_IDX],
        teacher_feats,
        key="clip_block",
    )

    # ---- Build CLIP zero-shot head for ImageNet-1K once ----
    if args.local_rank == 0:
        _logger.info("Preparing CLIP zero-shot head (ImageNet-1K)...")
    text_features, _ = prepare_zeroshot_head(clip_model=clip_model)

    # We will use zero-shot Top-1 as the eval metric
    args.eval_metric = "zs_top1"
    eval_metric = args.eval_metric

    if args.local_rank == 0:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        _logger.info(
            f"Model created - Total: {total:,}, Trainable: {trainable:,} ({100*trainable/total:.2f}%)"
        )

    data_config = resolve_data_config(
        vars(args),
        model=model.fastvit,
        verbose=args.local_rank == 0,
    )

    if args.local_rank == 0:
        _logger.info("Overriding data config with CLIP Mean/Std/Res")

    # OpenAI CLIP specific normalization
    data_config["mean"] = (0.48145466, 0.4578275, 0.40821073)
    data_config["std"] = (0.26862954, 0.26130258, 0.27577711)

    # Force input size to 3x224x224
    data_config["input_size"] = (3, 224, 224)

    # setup augmentation batch splits for contrastive loss or split bn
    num_aug_splits = 0

    # move model to GPU, enable channels last layout if set
    model.cuda()
    clip_model.cuda()

    if args.torchscript:
        assert not use_amp == "apex", "Cannot use APEX AMP with torchscripted model"
        model = torch.jit.script(model)

    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))

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
            model = NativeDDP(
                model,
                device_ids=[args.local_rank],
                broadcast_buffers=not args.no_ddp_bb,
            )
        # NOTE: EMA model does not need to be wrapped by DDP

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

    if args.local_rank == 0:
        _logger.info("Scheduled epochs: {}".format(num_epochs))

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
    dataset_eval = create_dataset(
        args.dataset,
        root=args.data_dir,
        split=args.val_split,
        is_training=False,
        class_map=args.class_map,
        download=args.dataset_download,
        batch_size=args.batch_size,
    )

    if args.num_classes is None:
        args.num_classes = dataset_train.num_classes

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
        worker_seeding="all",
    )

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

    # setup checkpoint saver and eval metric tracking
    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    saver = None
    output_dir = None

    train_loss_fn = None
    validate_loss_fn = None

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
        output_dir = get_outdir(
            args.output if args.output else "./output/train", exp_name
        )
        decreasing = True if eval_metric == "loss" else False
        saver = CheckpointSaver(
            model=model,
            optimizer=optimizer,
            args=args,
            amp_scaler=loss_scaler,
            checkpoint_dir=output_dir,
            recovery_dir=output_dir,
            decreasing=decreasing,
            max_history=args.checkpoint_hist,
        )
        with open(os.path.join(output_dir, "args.yaml"), "w") as f:
            f.write(args_text)

    try:
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
                mixup_fn=mixup_fn,
                clip_model=clip_model,  # Pass CLIP model to training function
            )

            eval_metrics = validate(
                model,
                loader_eval,
                validate_loss_fn,
                args,
                clip_model=clip_model,
                amp_autocast=amp_autocast,
            )

            # ---- Zero-shot evaluation on ImageNet val using CLIP prompts ----
            model_for_zs = model.module if isinstance(model, NativeDDP) else model
            zs_metrics = evaluate_zero_shot(
                model=model_for_zs,
                loader=loader_eval,
                text_features=text_features,
                args=args,
                amp_autocast=amp_autocast,
            )
            eval_metrics.update(zs_metrics)

            if lr_scheduler is not None:
                # step LR for next epoch, using zs_top1
                lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

            if output_dir is not None:
                update_summary(
                    epoch,
                    train_metrics,
                    eval_metrics,
                    os.path.join(output_dir, "summary.csv"),
                    write_header=best_metric is None,
                    log_wandb=False,
                )

            # ----- Checkpointing based on zero-shot Top-1 -----
            if saver is not None:
                # save proper checkpoint with eval metric (zs_top1)
                save_metric = eval_metrics[eval_metric]
                best_metric, best_epoch = saver.save_checkpoint(
                    epoch, metric=save_metric
                )

    except KeyboardInterrupt:
        pass
    if best_metric is not None:
        _logger.info("*** Best metric: {0} (epoch {1})".format(best_metric, best_epoch))
    try:
        clip_handle.remove()
    except Exception:
        pass


def train_one_epoch(
    epoch,
    model,
    loader,
    optimizer,
    loss_fn,
    args,
    lr_scheduler=None,
    saver=None,
    output_dir=None,
    amp_autocast=suppress,
    loss_scaler=None,
    mixup_fn=None,
    clip_model=None,
):

    second_order = hasattr(optimizer, "is_second_order") and optimizer.is_second_order
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    model.train()
    if clip_model is not None:
        clip_model.eval()  # Keep CLIP frozen in eval mode

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)

    for batch_idx, (input, target) in enumerate(loader):
        teacher_feats.clear()
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)

        if not args.prefetcher:
            input, target = input.cuda(), target.cuda()

        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        with amp_autocast():
            # Student Forward (final + stage2(14x14) intermediate)
            student_final, student_stage2 = model(input, return_intermediate=True)
            student_final = F.normalize(student_final.float(), dim=-1)
            student_stage2 = F.normalize(student_stage2.float(), dim=-1)

            teacher_final = None
            teacher_cls = None

            if clip_model is not None:
                with torch.no_grad():
                    clip_input = input
                    if input.shape[-1] != 224:
                        clip_input = F.interpolate(
                            input, size=(224, 224), mode="bicubic", align_corners=False
                        )

                    teacher_final = clip_model.encode_image(clip_input).float()
                    teacher_final = F.normalize(teacher_final, dim=-1)

                if teacher_feats is not None and "clip_block" in teacher_feats:
                    t_block = teacher_feats["clip_block"]  # [B, seq, C=1024]
                    cls_pre = t_block[:, 0].float()  # CLS token before projection

                    # Project CLS token into CLIP embedding space (1024 -> 768)
                    proj = clip_model.visual.proj.float()  # [1024, 768]
                    teacher_cls = cls_pre @ proj  # [B, 768]
                    teacher_cls = F.normalize(teacher_cls, dim=-1)

            # Total loss: MSE final + MSE intermediate
            final_loss = (
                F.mse_loss(student_final, teacher_final)
                if teacher_final is not None
                else torch.zeros((), device=input.device)
            )

            if teacher_cls is not None:
                intermediate_loss = F.mse_loss(student_stage2, teacher_cls)
            else:
                intermediate_loss = torch.zeros((), device=input.device)

            loss = (
                FINAL_FEATURE_WEIGHT * final_loss
                + INTERMEDIATE_FEATURE_WEIGHT * intermediate_loss
            )

        crash_on_bad_loss(
            loss=loss,
            args=args,
            batch_idx=batch_idx,
            images=input,
            student_features=student_final,
            teacher_features=teacher_final,
            epoch=epoch,
        )

        if not args.distributed:
            losses_m.update(loss.item(), input.size(0))

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                loss,
                optimizer,
                clip_grad=args.clip_grad,
                clip_mode=args.clip_mode,
                parameters=model_parameters(
                    model, exclude_head="agc" in args.clip_mode
                ),
                create_graph=second_order,
            )
        else:
            loss.backward(create_graph=second_order)
            if args.clip_grad is not None:
                dispatch_clip_grad(
                    model_parameters(model, exclude_head="agc" in args.clip_mode),
                    value=args.clip_grad,
                    mode=args.clip_mode,
                )
            optimizer.step()

        torch.cuda.synchronize()
        num_updates += 1
        batch_time_m.update(time.time() - end)

        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [pg["lr"] for pg in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            param_groups = list(optimizer.param_groups)
            wd0 = param_groups[0]["weight_decay"]
            wd1 = param_groups[1]["weight_decay"] if len(param_groups) > 1 else wd0

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                losses_m.update(reduced_loss.item(), input.size(0))

            if args.local_rank == 0:
                _logger.info(
                    "Train: {} [{:>4d}/{} ({:>3.0f}%)]  "
                    "Loss: {loss.val:#.4g} ({loss.avg:#.3g})  "
                    "Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  "
                    "({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  "
                    "LR: {lr:.3e}, WD0: {wd0:.6e}, WD1: {wd1:.6e}    "
                    "Data: {data_time.val:.3f} ({data_time.avg:.3f})".format(
                        epoch,
                        batch_idx,
                        len(loader),
                        100.0 * batch_idx / last_idx,
                        loss=losses_m,
                        batch_time=batch_time_m,
                        rate=input.size(0) * args.world_size / batch_time_m.val,
                        rate_avg=input.size(0) * args.world_size / batch_time_m.avg,
                        lr=lr,
                        wd0=wd0,
                        wd1=wd1,
                        data_time=data_time_m,
                    )
                )

        if (
            saver is not None
            and args.recovery_interval
            and (last_batch or (batch_idx + 1) % args.recovery_interval == 0)
        ):
            saver.save_recovery(epoch, batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()

    if hasattr(optimizer, "sync_lookahead"):
        optimizer.sync_lookahead()

    return OrderedDict([("loss", losses_m.avg)])


def validate(
    model, loader, loss_fn, args, clip_model=None, amp_autocast=suppress, log_suffix=""
):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()

    model.eval()
    if clip_model is not None:
        clip_model.eval()

    end = time.time()
    last_idx = len(loader) - 1

    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            teacher_feats.clear()
            last_batch = batch_idx == last_idx
            if not args.prefetcher:
                input = input.cuda()
                target = target.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                # Student features
                student_final, student_stage2 = model(input, return_intermediate=True)
                student_final = F.normalize(student_final.float(), dim=-1)
                student_stage2 = F.normalize(student_stage2.float(), dim=-1)

                teacher_final = None
                teacher_cls = None

                if clip_model is not None:
                    clip_input = input
                    if input.shape[-1] != 224:
                        clip_input = F.interpolate(
                            input, size=(224, 224), mode="bicubic", align_corners=False
                        )

                    teacher_final = clip_model.encode_image(clip_input).float()
                    teacher_final = F.normalize(teacher_final, dim=-1)

                    if teacher_feats is not None and "clip_block" in teacher_feats:
                        t_block = teacher_feats["clip_block"]  # [B, seq, C=1024]
                        cls_pre = t_block[:, 0].float()  # CLS token before projection

                        # Project CLS token into CLIP embedding space (1024 -> 768)
                        proj = clip_model.visual.proj.float()  # [1024, 768]
                        teacher_cls = cls_pre @ proj  # [B, 768]
                        teacher_cls = F.normalize(teacher_cls, dim=-1)

                # Losses
                final_loss = (
                    F.mse_loss(student_final, teacher_final)
                    if teacher_final is not None
                    else torch.zeros((), device=input.device)
                )

                if teacher_cls is not None:
                    intermediate_loss = F.mse_loss(student_stage2, teacher_cls)
                else:
                    intermediate_loss = torch.zeros((), device=input.device)

                loss = (
                    FINAL_FEATURE_WEIGHT * final_loss
                    + INTERMEDIATE_FEATURE_WEIGHT * intermediate_loss
                )

            crash_on_bad_loss(
                loss=loss,
                args=args,
                batch_idx=batch_idx,
                images=input,
                student_features=student_final,
                teacher_features=teacher_final,
                epoch=None,
            )

            torch.cuda.synchronize()
            losses_m.update(loss.item(), input.size(0))
            batch_time_m.update(time.time() - end)
            end = time.time()

            if args.local_rank == 0 and (
                last_batch or batch_idx % args.log_interval == 0
            ):
                log_name = "Test" + log_suffix
                _logger.info(
                    "{0}: [{1:>4d}/{2}]  "
                    "Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  "
                    "Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})".format(
                        log_name,
                        batch_idx,
                        last_idx,
                        batch_time=batch_time_m,
                        loss=losses_m,
                    )
                )

    return OrderedDict([("loss", losses_m.avg)])


if __name__ == "__main__":
    main()
