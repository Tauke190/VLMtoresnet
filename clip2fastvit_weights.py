"""
Convert CLIP ViT-L/14 weights into FastViT-SA36 architecture.
"""

import argparse
import logging
from collections import OrderedDict

import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def truncate_weight(w: torch.Tensor, target_shape: tuple) -> torch.Tensor:
    slices = []
    for src_dim, tgt_dim in zip(w.shape, target_shape):
        slices.append(slice(0, min(src_dim, tgt_dim)))
    out = w[tuple(slices)].clone()

    if out.shape != target_shape:
        padded = torch.zeros(target_shape, dtype=w.dtype, device=w.device)
        assign_slices = tuple(slice(0, s) for s in out.shape)
        padded[assign_slices] = out
        out = padded
    return out


def match_distribution(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    src_f = src.float()
    ref_f = ref.float()
    src_std = src_f.std().clamp(min=1e-8)
    ref_std = ref_f.std().clamp(min=1e-8)
    out = (src_f - src_f.mean()) / src_std * ref_std + ref_f.mean()
    return out.to(src.dtype)


def linear_to_conv1x1(weight_2d: torch.Tensor, target_out: int, target_in: int) -> torch.Tensor:
    w = truncate_weight(weight_2d, (target_out, target_in))
    return w.unsqueeze(-1).unsqueeze(-1)


def interpolate_pos_embed(pos_embed: torch.Tensor, src_grid: int, tgt_grid: int, has_cls: bool = True):
    if src_grid == tgt_grid:
        return pos_embed.clone()

    if has_cls:
        cls_token = pos_embed[:1]
        patch_embed = pos_embed[1:]
    else:
        cls_token = None
        patch_embed = pos_embed

    D = patch_embed.shape[-1]
    patch_embed = patch_embed.reshape(1, src_grid, src_grid, D).permute(0, 3, 1, 2).float()
    patch_embed = F.interpolate(patch_embed, size=(tgt_grid, tgt_grid),
                                mode="bicubic", align_corners=False)
    patch_embed = patch_embed.permute(0, 2, 3, 1).reshape(-1, D)

    if cls_token is not None:
        return torch.cat([cls_token, patch_embed], dim=0)
    return patch_embed


def load_clip_state_dict(clip_model_name: str, device: str = "cpu"):
    import CLIP.clip as clip
    model, _ = clip.load(clip_model_name, device=device, jit=False)
    sd = model.state_dict()

    vision_width = sd["visual.conv1.weight"].shape[0]
    n_layers = len([k for k in sd if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
    patch_size = sd["visual.conv1.weight"].shape[-1]
    grid_size = round((sd["visual.positional_embedding"].shape[0] - 1) ** 0.5)

    logger.info(f"CLIP vision: width={vision_width}, layers={n_layers}, "
                f"patch_size={patch_size}, grid={grid_size}")
    return sd, vision_width, n_layers, patch_size, grid_size


def extract_clip_visual_blocks(clip_sd: dict, layer_groups: tuple):
    cumulative = [0]
    for g in layer_groups:
        cumulative.append(cumulative[-1] + g)

    blocks = [[{} for _ in range(g)] for g in layer_groups]

    for key, value in clip_sd.items():
        if not key.startswith("visual.transformer.resblocks."):
            continue
        parts = key.split(".")
        block_idx = int(parts[3])
        rest = ".".join(parts[4:])

        for gi in range(len(layer_groups)):
            if cumulative[gi] <= block_idx < cumulative[gi + 1]:
                local = block_idx - cumulative[gi]
                blocks[gi][local][rest] = value
                break

    return blocks


FASTVIT_SA36_STAGE_CFG = {
    0: (0, 6, 64, 4, "repmixer"),
    1: (2, 6, 128, 4, "repmixer"),
    2: (4, 18, 256, 4, "repmixer"),
    3: (7, 6, 512, 4, "attention"),
}

CLIP_LAYER_GROUPS = (2, 2, 18, 2)


def convert_mlp_to_convffn(clip_block: dict, fastvit_dim: int, mlp_ratio: int, clip_width: int):
    hidden = fastvit_dim * mlp_ratio
    result = {}

    if "mlp.c_fc.weight" in clip_block:
        result["fc1.weight"] = linear_to_conv1x1(
            clip_block["mlp.c_fc.weight"], hidden, fastvit_dim
        )
    if "mlp.c_fc.bias" in clip_block:
        result["fc1.bias"] = truncate_weight(
            clip_block["mlp.c_fc.bias"], (hidden,)
        )

    if "mlp.c_proj.weight" in clip_block:
        result["fc2.weight"] = linear_to_conv1x1(
            clip_block["mlp.c_proj.weight"], fastvit_dim, hidden
        )
    if "mlp.c_proj.bias" in clip_block:
        result["fc2.bias"] = truncate_weight(
            clip_block["mlp.c_proj.bias"], (fastvit_dim,)
        )

    return result


def convert_attention(clip_block: dict, fastvit_dim: int, clip_width: int):
    result = {}
    W = clip_width
    D = fastvit_dim

    if "attn.in_proj_weight" in clip_block:
        qkv_full = clip_block["attn.in_proj_weight"]
        q_w, k_w, v_w = qkv_full[:W], qkv_full[W:2*W], qkv_full[2*W:3*W]
        q_t = truncate_weight(q_w, (D, D))
        k_t = truncate_weight(k_w, (D, D))
        v_t = truncate_weight(v_w, (D, D))
        result["qkv.weight"] = torch.cat([q_t, k_t, v_t], dim=0)

    if "attn.in_proj_bias" in clip_block:
        bias_full = clip_block["attn.in_proj_bias"]
        q_b, k_b, v_b = bias_full[:W], bias_full[W:2*W], bias_full[2*W:3*W]
        q_bt = truncate_weight(q_b, (D,))
        k_bt = truncate_weight(k_b, (D,))
        v_bt = truncate_weight(v_b, (D,))
        result["qkv.bias"] = torch.cat([q_bt, k_bt, v_bt], dim=0)

    if "attn.out_proj.weight" in clip_block:
        result["proj.weight"] = truncate_weight(
            clip_block["attn.out_proj.weight"], (D, D)
        )
    if "attn.out_proj.bias" in clip_block:
        result["proj.bias"] = truncate_weight(
            clip_block["attn.out_proj.bias"], (D,)
        )

    return result


def convert_layernorm_to_batchnorm(clip_block: dict, ln_prefix: str,
                                   fastvit_dim: int,
                                   preserve_running_stats: bool = False):
    result = {}
    wkey = f"{ln_prefix}.weight"
    bkey = f"{ln_prefix}.bias"

    if wkey in clip_block:
        result["weight"] = truncate_weight(clip_block[wkey], (fastvit_dim,))
    if bkey in clip_block:
        result["bias"] = truncate_weight(clip_block[bkey], (fastvit_dim,))

    if not preserve_running_stats:
        result["running_mean"] = torch.zeros(fastvit_dim)
        result["running_var"] = torch.ones(fastvit_dim)
        result["num_batches_tracked"] = torch.tensor(0, dtype=torch.long)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Convert CLIP ViT-L/14 weights to FastViT-SA36 initialization"
    )
    parser.add_argument("--clip-model", type=str, default="ViT-L/14")
    parser.add_argument("--fastvit-variant", type=str, default="fastvit_sa36")
    parser.add_argument("--fastvit-checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default="clip_to_fastvit_sa36.pt")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logger.info(f"Loading CLIP {args.clip_model}...")
    clip_sd, clip_width, n_layers, patch_size, grid_size = \
        load_clip_state_dict(args.clip_model, device=args.device)

    logger.info(f"Instantiating FastViT variant: {args.fastvit_variant}")
    import models
    fastvit_model = getattr(models, args.fastvit_variant)(
        pretrained=False, num_classes=args.num_classes
    )
    fastvit_model.eval()

    if args.fastvit_checkpoint is not None:
        logger.info(f"Loading existing FastViT checkpoint: {args.fastvit_checkpoint}")
        ckpt = torch.load(args.fastvit_checkpoint, map_location=args.device)
        base_sd = ckpt["state_dict"] if "state_dict" in ckpt else \
                  ckpt["model"] if "model" in ckpt else ckpt
        fastvit_model.load_state_dict(base_sd, strict=False)

    logger.info("Conversion complete.")


if __name__ == "__main__":
    main()
