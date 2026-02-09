from typing import Union
from contextlib import suppress
import logging

from timm.utils import *
from timm.loss import *


import torch
import torch.nn as nn
import torchvision.utils
import wandb
import time 
from collections import OrderedDict

from misc.utils import dump_images, save_image

_logger = logging.getLogger("train")


def evaluate_zeroshot(eval_ctx, model, device, channels_last=None, amp_autocast=suppress, distributed=False, rank=0, world_size=1):
    """
    Run zero-shot evaluation using FastViT + projector
    as image encoder and CLIP text features as class prototypes.

    Returns top-1 and top-5 accuracy in %.
    """
    loader = eval_ctx["loader"]
    text_features = eval_ctx["text_features"]  # [C, D]

    top1_m = AverageMeter()
    top5_m = AverageMeter()
    backbone = model
    was_training = model.training
    model.eval()
    batch_size = loader.batch_size
    if rank  == 0:
        _logger.info(f"Evaluating.... {len(loader)} Iteratiopns on a B={loader.batch_size}, with {len(loader) * loader.batch_size} datapoints")
        print(type(model))
    with torch.no_grad():
        for images, targets in loader:
            # print(images.shape)
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if channels_last:
                images = images.contiguous(memory_format=torch.channels_last)

            images = images.float()
            feats, _, _ = backbone(images)

            if isinstance(feats, (tuple, list)):
                feats = feats[0]
            if feats.ndim == 4:
                feats = feats.mean(dim=[2, 3])  # global average pool if spatial

            feats = feats.float()
            feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-6)  # [B, D]

            if feats.dtype != text_features.dtype:
                text_features = text_features.to(feats.dtype)

            logits = 100.0 * feats @ text_features.T  # [B, num_classes]

            acc1, acc5 = accuracy(logits, targets, topk=(1, 5))

            # print("1... ", acc1, world_size)
            if distributed:
                acc1 = reduce_tensor(acc1, world_size)
                acc5 = reduce_tensor(acc5, world_size)
            
            # print("2... ", acc1, world_size)
            top1_m.update(acc1.item(), batch_size)
            top5_m.update(acc5.item(), batch_size)

    if was_training:
        model.train()

    acc1 = top1_m.avg
    acc5 = top5_m.avg

    return acc1, acc5

def run_zeroshot_eval(
    eval_ctx,
    args,
    model,
    when: str,
    epoch: Union[int, None] = None,
    batch_idx: Union[int, None] = None,
    has_wandb=False, 
    ):
    acc1, acc5 = evaluate_zeroshot(
        eval_ctx, model=model, device=args.device, channels_last=args.channels_last, distributed=args.distributed,
        world_size=args.world_size, rank=args.rank, 
    )

    # ----- logging strings -----
    if when == "before_train":
        msg = "Zero-shot before training"
    elif when == "epoch":
        msg = f"Zero-shot after epoch {epoch}"
    elif when == "batch":
        msg = f"Zero-shot at epoch {epoch}, batch {batch_idx}"
    else:
        msg = "Zero-shot"

    if args.rank == 0:
        _logger.info(f"{msg}: Acc@1 = {acc1:.2f}%")
        _logger.info(f"{msg}: Acc@5 = {acc5:.2f}%")

    # ----- wandb logging -----
    if args.log_wandb and has_wandb:
        log_dict = {}
        if when == "before_train":
            log_dict["zeroshot/top1_before_train"] = acc1
            log_dict["zeroshot/top5_before_train"] = acc5
            log_dict["epoch"] = -1
        elif when == "epoch":
            log_dict["zeroshot/top1"] = acc1
            log_dict["zeroshot/top5"] = acc5
            if epoch is not None:
                log_dict["epoch"] = epoch
        elif when == "batch":
            log_dict["zeroshot/top1_batch"] = acc1
            log_dict["zeroshot/top5_batch"] = acc5
            if epoch is not None:
                log_dict["epoch"] = epoch
            if batch_idx is not None:
                log_dict["batch"] = batch_idx

        wandb.log(log_dict)

    return acc1, acc5



############ LOGIT BASED EVAL 
def validate(model, loader, loss_fn, args, amp_autocast=suppress, log_suffix=""):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            if not args.prefetcher:
                input = input.cuda()
                target = target.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[1]

            # augmentation reduction
            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0 : target.size(0) : reduce_factor]

            loss = loss_fn(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if args.local_rank == 0 and (
                last_batch or batch_idx % args.log_interval == 0
            ):
                log_name = "Test" + log_suffix
                _logger.info(
                    "{0}: [{1:>4d}/{2}]  "
                    "Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  "
                    "Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  "
                    "Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  "
                    "Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})".format(
                        log_name,
                        batch_idx,
                        last_idx,
                        batch_time=batch_time_m,
                        loss=losses_m,
                        top1=top1_m,
                        top5=top5_m,
                    )
                )

    metrics = OrderedDict(
        [("loss", losses_m.avg), ("top1", top1_m.avg), ("top5", top5_m.avg)]
    )

    return metrics

