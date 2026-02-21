import time
from contextlib import suppress
from collections import OrderedDict
import logging

from timm.utils import *
from timm.loss import *


import torch
import torchvision.utils

from timm.models import model_parameters
from Functions.losses import LossManager

from models import VANILLA_MODELS

from train import log_output

_logger = logging.getLogger("train")

# the only change is pulling the logit scale from the model and sending it to loss manager

def train_one_epoch_with_logic_scaling(
    epoch, model, loader, optimizer, loss_manager: LossManager, args, lr_scheduler=None,
    saver=None, output_dir=None, amp_autocast=suppress, loss_scaler=None,
    model_ema=None, mixup_fn=None, wd_scheduler=None,
    zeroshot_eval_ctx=None, clip_model=None,
):

    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if args.prefetcher and loader.mixup_enabled:
            loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    second_order = hasattr(optimizer, "is_second_order") and optimizer.is_second_order
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    model.train()

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)

    if args.rank == 0:
        _logger.info(
            f"Training.... {len(loader)} Iterations on a "
            f"B={loader.loader.batch_size}, "
            f"with {len(loader) * loader.loader.batch_size} datapoints"
        )

    for batch_idx, (input, target) in enumerate(loader):

        if args.debug and batch_idx % 100 == 0 and batch_idx != 0:
            break

        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)

        if not args.prefetcher:
            input, target = input.cuda(), target.cuda()
            if mixup_fn is not None:
                input, target = mixup_fn(input, target)

        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        with amp_autocast():
            if args.model in VANILLA_MODELS:
                output = model(input)
                projected_embed = None
                logit_scale = None
            else:
                projected_embed, output, x, logit_scale = model(input)

            # Compute CLIP image features if needed for MSE loss
            clip_image_features = None
            if clip_model is not None and projected_embed is not None:
                with torch.no_grad():
                    clip_image_features = clip_model.encode_image(input)

            loss, loss_dict = loss_manager.compute(
                output,
                target,
                projected_embed,
                clip_image_features,
                logit_scale,
            )

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

        if model_ema is not None:
            model_ema.update(model)

        # Optional zero-shot eval inside epoch
        if (
            zeroshot_eval_ctx is not None
            and args.rank == 0
            and getattr(args, "zeroshot_eval_interval", 0) > 0
            and (batch_idx + 1) % args.zeroshot_eval_interval == 0
        ):
            run_zeroshot_eval(
                zeroshot_eval_ctx,
                args,
                model,
                when="batch",
                epoch=epoch,
                batch_idx=batch_idx + 1,
            )

        torch.cuda.synchronize()
        num_updates += 1
        batch_time_m.update(time.time() - end)

        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group["lr"] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)
            param_groups = list(optimizer.param_groups)
            wd0 = param_groups[0]["weight_decay"]
            wd1 = param_groups[1]["weight_decay"] if len(param_groups) > 1 else wd0

            if args.local_rank == 0:
                log_output(
                    epoch,
                    batch_idx,
                    total_len=len(loader),
                    last_idx=last_idx,
                    input_size=input.size(0),
                    world_size=args.world_size,
                    batch_time_m=batch_time_m,
                    lr=lr,
                    wd0=wd0,
                    wd1=wd1,
                    data_time_m=data_time_m,
                    losses_m=losses_m,
                    loss_dict=loss_dict,
                )

                if args.save_images and output_dir:
                    torchvision.utils.save_image(
                        input,
                        os.path.join(output_dir, f"train-batch-{batch_idx}.jpg"),
                        padding=0,
                        normalize=True,
                    )

        if (
            saver is not None
            and args.recovery_interval
            and (last_batch or (batch_idx + 1) % args.recovery_interval == 0)
        ):
            saver.save_recovery(epoch, batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        if wd_scheduler is not None:
            wd_scheduler.update_weight_decay(optimizer)

        end = time.time()

    if hasattr(optimizer, "sync_lookahead"):
        optimizer.sync_lookahead()

    return OrderedDict([("loss", losses_m.avg)])