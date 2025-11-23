import os, time, logging
from datetime import datetime
from collections import OrderedDict
from contextlib import suppress

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from timm.data import create_dataset, create_loader
from timm.models import model_parameters
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
from timm.utils import (
    setup_default_logging,
    AverageMeter,
    CheckpointSaver,
    get_outdir,
    update_summary,
    random_seed,
    reduce_tensor,
)

import clip
from FastViT_KD import create_fastvit_clip

# ---- CONFIG ----
DATA_DIR = "/datasets/ImageNet2012nonpub"
BATCH_SIZE = 64
EPOCHS = 30
LR = 2e-4
WEIGHT_DECAY = 0.05
WORKERS = 8

USE_AMP = True
LOG_INTERVAL = 50
OUTPUT_DIR = "./output/train"
SEED = 42

WARMUP_EPOCHS = 2
MIN_LR = 1e-6
CLIP_GRAD = 0.01
CHECKPOINT_HIST = 5  # keep last 5 checkpoints (still saves every epoch)
# ---------------

_logger = logging.getLogger("train")
torch.backends.cudnn.benchmark = True


def setup_distributed():
    """Detect + init DDP automatically from torchrun env vars."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        distributed = True

        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group("nccl", init_method="env://")
        _logger.info(f"DDP: Rank {rank}/{world_size}, Local {local_rank}")
    else:
        rank = world_size = 0
        local_rank = 0
        distributed = False
        _logger.info("Single GPU training")

    return rank, world_size, local_rank, distributed


def train_one_epoch(
    epoch,
    model,
    loader,
    optimizer,
    clip_model,
    scaler,
    amp_autocast,
    world_size,
    local_rank,
    distributed,
):
    """One epoch of CLIP feature KD: student -> teacher embeddings."""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()
    clip_model.eval()  # teacher always frozen

    end = time.time()
    last_idx = len(loader) - 1

    for batch_idx, (images, _) in enumerate(loader):
        data_time.update(time.time() - end)
        images = images.cuda(non_blocking=True)

        with amp_autocast():
            # Student embeddings (already normalized in architecture)
            student_feats = model(images)  # (B, 768)

            # Teacher CLIP embeddings (normalize for stable target space)
            with torch.no_grad():
                teacher_feats = clip_model.encode_image(images)
                teacher_feats = F.normalize(teacher_feats.float(), dim=-1)

            # Embedding matching loss
            loss = F.mse_loss(student_feats.float(), teacher_feats)

        # Safety check
        if torch.isnan(loss) or torch.isinf(loss):
            raise RuntimeError(f"NaN/Inf loss at epoch {epoch}, batch {batch_idx}")

        # Backprop (AMP)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()

        # Optional grad clipping on trainable params
        if CLIP_GRAD:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model_parameters(model), CLIP_GRAD)

        scaler.step(optimizer)
        scaler.update()

        torch.cuda.synchronize()
        batch_time.update(time.time() - end)

        # Reduce loss across GPUs if needed
        if distributed:
            reduced_loss = reduce_tensor(loss.data, world_size)
            losses.update(reduced_loss.item(), images.size(0))
        else:
            losses.update(loss.item(), images.size(0))

        # Basic logging
        if local_rank == 0 and (batch_idx % LOG_INTERVAL == 0 or batch_idx == last_idx):
            lr = optimizer.param_groups[0]["lr"]
            thr = images.size(0) * (world_size if distributed else 1) / batch_time.val
            _logger.info(
                f"[{epoch}][{batch_idx}/{len(loader)}] "
                f"Loss: {losses.val:.4f} ({losses.avg:.4f}) | "
                f"Time: {batch_time.val:.3f}s ({thr:.0f} img/s) | LR: {lr:.2e}"
            )

        end = time.time()

    return OrderedDict([("loss", losses.avg)])


@torch.no_grad()
def validate(
    model, loader, clip_model, amp_autocast, world_size, local_rank, distributed
):
    """Validation = same KD loss but no gradients."""
    losses = AverageMeter()
    model.eval()
    clip_model.eval()

    for images, _ in loader:
        images = images.cuda(non_blocking=True)

        with amp_autocast():
            student_feats = model(images)  # assumed normalized already
            teacher_feats = clip_model.encode_image(images)
            teacher_feats = F.normalize(teacher_feats.float(), dim=-1)
            loss = F.mse_loss(student_feats.float(), teacher_feats)

        if distributed:
            loss = reduce_tensor(loss.data, world_size)

        losses.update(loss.item(), images.size(0))

    if local_rank == 0:
        _logger.info(f"Val Loss: {losses.avg:.4f}")

    return OrderedDict([("loss", losses.avg)])


def main():
    setup_default_logging()

    # DDP + seed
    rank, world_size, local_rank, distributed = setup_distributed()
    random_seed(SEED, rank)

    if local_rank == 0:
        _logger.info("Creating FastViT SA36 -> CLIP L/14")

    # Student (FastViT + projector) — frozen backbone inside create_fastvit_clip
    model = create_fastvit_clip(
        model_name="fastvit_sa36",
        pretrained=True,
        embed_dim=768,
        lock=True,
    ).cuda()

    # Teacher (CLIP ViT-L/14)
    clip_model, _ = clip.load("ViT-L/14", device="cuda")
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    # Wrap student in DDP if multi-GPU
    if distributed:
        model = DDP(model, device_ids=[local_rank])

    # Optimizer on trainable params only
    optimizer = create_optimizer_v2(
        model, opt="adamw", lr=LR, weight_decay=WEIGHT_DECAY
    )

    # AMP setup
    amp_autocast = torch.cuda.amp.autocast if USE_AMP else suppress
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    # LR scheduler (cosine)
    lr_scheduler, num_epochs = create_scheduler_v2(
        optimizer,
        sched="cosine",
        num_epochs=EPOCHS,
        warmup_epochs=WARMUP_EPOCHS,
        warmup_lr=1e-6,
        min_lr=MIN_LR,
    )

    # CLIP normalization for dataloaders
    CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

    # Datasets / loaders
    dataset_train = create_dataset("", root=DATA_DIR, split="train", is_training=True)
    dataset_val = create_dataset(
        "", root=DATA_DIR, split="validation", is_training=False
    )

    loader_train = create_loader(
        dataset_train,
        input_size=(3, 224, 224),
        batch_size=BATCH_SIZE,
        is_training=True,
        use_prefetcher=True,
        interpolation="bicubic",
        mean=CLIP_MEAN,
        std=CLIP_STD,
        num_workers=WORKERS,
        distributed=distributed,
        pin_memory=True,
    )
    loader_val = create_loader(
        dataset_val,
        input_size=(3, 224, 224),
        batch_size=BATCH_SIZE,
        is_training=False,
        use_prefetcher=True,
        interpolation="bicubic",
        mean=CLIP_MEAN,
        std=CLIP_STD,
        num_workers=WORKERS,
        distributed=distributed,
        crop_pct=0.95,
        pin_memory=True,
    )

    # Checkpointing / logs (rank 0 only)
    saver = None
    output_dir = None
    if rank == 0:
        exp_name = datetime.now().strftime("%Y%m%d-%H%M%S") + "-fastvit_sa36"
        output_dir = get_outdir(OUTPUT_DIR, exp_name)
        saver = CheckpointSaver(
            model=model,
            optimizer=optimizer,
            args=None,
            model_ema=None,
            amp_scaler=scaler,
            checkpoint_dir=output_dir,
            recovery_dir=output_dir,
            decreasing=True,
            max_history=CHECKPOINT_HIST,
        )
        _logger.info(f"Output: {output_dir}")

    # Main loop
    best_metric = best_epoch = None
    for epoch in range(num_epochs):
        if distributed and hasattr(loader_train.sampler, "set_epoch"):
            loader_train.sampler.set_epoch(epoch)

        train_metrics = train_one_epoch(
            epoch,
            model,
            loader_train,
            optimizer,
            clip_model,
            scaler,
            amp_autocast,
            world_size,
            local_rank,
            distributed,
        )
        val_metrics = validate(
            model,
            loader_val,
            clip_model,
            amp_autocast,
            world_size,
            local_rank,
            distributed,
        )

        if lr_scheduler:
            lr_scheduler.step(epoch + 1, val_metrics["loss"])

        if output_dir:
            update_summary(
                epoch,
                train_metrics,
                val_metrics,
                os.path.join(output_dir, "summary.csv"),
                write_header=(best_metric is None),
            )
        if saver:
            best_metric, best_epoch = saver.save_checkpoint(
                epoch, metric=val_metrics["loss"]
            )

    if rank == 0 and best_metric is not None:
        _logger.info(f"✓ Best: {best_metric:.4f} @ epoch {best_epoch}")


if __name__ == "__main__":
    main()
