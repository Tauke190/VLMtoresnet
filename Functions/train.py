import time
from contextlib import suppress
from typing import Union
from collections import OrderedDict
import logging

from timm.utils import *
from timm.loss import *


import torch
import torch.nn as nn
import torchvision.utils

from timm.models import model_parameters

_logger = logging.getLogger("train")


def log_output(epoch, batch_idx, total_len, last_idx, input_size, world_size, batch_time_m, 
    lr, wd0, wd1, data_time_m, losses_m=0, base_loss_val=0, clip_loss_val=0):
    percent = 100.0 * batch_idx / last_idx
    rate= input_size * world_size / batch_time_m.val
    rate_avg=input_size * world_size / batch_time_m.avg

    str = f"Train: {epoch} [{batch_idx:>4d}/{total_len} ({percent:>3.0f}%)]\t"
    str += f"Loss: {losses_m.val:#.4g} ({losses_m.avg:#.3g})\t" 
    str += f"Base Loss: {base_loss_val:.6f}\t"
    str += f"CLIP Loss: {clip_loss_val:.6f}\t" 
    str += f"Time: {batch_time_m.val:.3f}s, {rate:>7.2f}/s ({batch_time_m.avg:.3f}s, {rate_avg:>7.2f}/s)\t"  
    str += f"LR: {lr:.3e}, WD0: {wd0:.6e}, WD1: {wd1:.6e}  Data: {data_time_m.val:.3f} ({data_time_m.avg:.3f})"
    _logger.info(str)
        




def train_one_epoch(
    epoch, model, loader, optimizer, loss_fn, args, lr_scheduler=None,
    saver=None, output_dir=None, amp_autocast=suppress, loss_scaler=None,
    model_ema=None, mixup_fn=None, wd_scheduler=None,
    clip_text_features=None, clip_logit_scale=None, clip_loss_fn=None,
    zeroshot_eval_ctx=None,
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
    
    if args.rank  == 0:
        _logger.info(f"Training.... {len(loader)} Iteratiopns on a B={loader.loader.batch_size}, with {len(loader) * loader.loader.batch_size} datapoints")
    for batch_idx, (input, target) in enumerate(loader):
        if args.debug and batch_idx % 100 ==0 and batch_idx != 0 :
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
            # stage_ins = model.get_stage_inputs(input)
            # if batch_idx == 0 and args.rank == 0:
            #     print("==== Stage input debug (batch 0) ====")
            #     print("Number of stage inputs:", len(stage_ins))
            #     for i, feat in enumerate(stage_ins):
            #         print(f"[Stage {i}] "
            #             f"shape={tuple(feat.shape)}, "
            #             f"dtype={feat.dtype}, device={feat.device}, "
            #             f"min={feat.min().item():.4f}, "
            #             f"max={feat.max().item():.4f}, "
            #             f"mean={feat.mean().item():.4f}, "
            #             f"std={feat.std().item():.4f}, "
            #             f"nonzero_frac={(feat != 0).float().mean().item()*100:.2f}%")
            #     print("=====================================")
            if args.model == 'fastvit_sa36':
                output = model(input)
            else:
                projected_embed, output, x = model(input)
            # base_loss = loss_fn(input, output, target)
            base_loss = loss_fn(output, target)

            loss = base_loss


        #-------------Added by avinash gyawali-------------------------#
        # --- CLIP feature alignment loss ----------------------------#
        clip_loss = torch.tensor(0.0, device=base_loss.device)  # Ensure tensor for .item()
        if args.clip_loss_weight > 0.0:
            '''
            if anything fails after this, trying to compute clip loss unnecessarily   
            '''
            # and clip_loss_fn is not None and clip_text_features is not None \
            # and clip_logit_scale is not None and isinstance(target, torch.Tensor) and target.dtype == torch.long):    
            feats = projected_embed
            if isinstance(feats, (tuple, list)):
                feats = feats[0]
            if feats.ndim == 4 and feats.shape[2] > 1 and feats.shape[3] > 1:
                feats = feats.mean(dim=[2, 3])  # Global average pooling
            
            clip_loss = clip_loss_fn(feats, clip_text_features[target], clip_logit_scale)
            loss += args.clip_loss_weight * clip_loss
        
        
        #-------------------------------------------------------------------------#
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

        if model_ema is not None:
            model_ema.update(model)

        # -----------------------------------------------------------------
        # Optional generic zero-shot eval every N batches inside epoch
        if ( zeroshot_eval_ctx is not None and args.rank == 0 and getattr(args, "zeroshot_eval_interval", 0) > 0 and (batch_idx + 1) % args.zeroshot_eval_interval == 0):
            run_zeroshot_eval( zeroshot_eval_ctx, args, model,
                when="batch", epoch=epoch, batch_idx=batch_idx + 1)

        # -----------------------------------------------------------------

        torch.cuda.synchronize()
        num_updates += 1
        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group["lr"] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)
            param_groups = list(optimizer.param_groups)
            wd0 = param_groups[0]["weight_decay"]
            wd1 = param_groups[1]["weight_decay"] if len(param_groups) > 1 else wd0

            # --- Add CLIP loss and base loss to log ---
            clip_loss_val = clip_loss.item() if isinstance(clip_loss, torch.Tensor) else float(clip_loss)
            base_loss_val = base_loss.item() if isinstance(base_loss, torch.Tensor) else float(base_loss)

            if args.local_rank == 0:
                log_output(epoch, batch_idx, total_len=len(loader), last_idx=last_idx, input_size=input.size(0), world_size=args.world_size, batch_time_m=batch_time_m, 
                        lr=lr, wd0=wd0, wd1=wd1, data_time_m=data_time_m, losses_m=losses_m, base_loss_val=base_loss_val, clip_loss_val=clip_loss_val)

                if args.save_images and output_dir:
                    torchvision.utils.save_image(
                        input,
                        os.path.join(output_dir, "train-batch-%d.jpg" % batch_idx),
                        padding=0,
                        normalize=True,
                    )

        if (saver is not None and args.recovery_interval and (last_batch or (batch_idx + 1) % args.recovery_interval == 0)):
            saver.save_recovery(epoch, batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)
        if wd_scheduler is not None:
            wd_scheduler.update_weight_decay(optimizer)

        end = time.time()
        # end for

    if hasattr(optimizer, "sync_lookahead"):
        optimizer.sync_lookahead()

    return OrderedDict([("loss", losses_m.avg)])

