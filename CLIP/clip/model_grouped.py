"""
Modified CLIP model with ResidualAttentionBlock grouping (2, 2, 18, 2)
for ViT-L/14 as described in the paper.
"""

from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .model import (
    Bottleneck,
    AttentionPool2d,
    ModifiedResNet,
    LayerNorm,
    QuickGELU,
    ResidualAttentionBlock,
    convert_weights,
)


class GroupedTransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, layer_groups: Tuple[int, ...] = None):
        super().__init__()
        self.width = width
        self.layers = layers

        if layer_groups is not None:
            assert len(layer_groups) == 4 and sum(layer_groups) == layers
            all_blocks = [ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)]
            idx = 0
            self.layer1 = nn.Sequential(*all_blocks[idx:idx + layer_groups[0]]); idx += layer_groups[0]
            self.layer2 = nn.Sequential(*all_blocks[idx:idx + layer_groups[1]]); idx += layer_groups[1]
            self.layer3 = nn.Sequential(*all_blocks[idx:idx + layer_groups[2]]); idx += layer_groups[2]
            self.layer4 = nn.Sequential(*all_blocks[idx:idx + layer_groups[3]])
        else:
            self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        if hasattr(self, 'layer1'):
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            return x
        return self.resblocks(x)


class GroupedVisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int, layer_groups: Tuple[int, ...] = None):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = GroupedTransformer(width, layers, heads, layer_groups=layer_groups)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class GroupedCLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 vision_layer_groups: Tuple[int, ...] = None
                 ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = GroupedVisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                layer_groups=vision_layer_groups
            )

        self.transformer = GroupedTransformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def get_layer4_attention_maps(model, images):
    """Extract attention maps from the last block in layer4 of the grouped ViT.

    Args:
        model: A GroupedCLIP model (or its .visual component).
        images: Input image tensor of shape (B, 3, H, W).

    Returns:
        attn_maps: List with single attention weight tensor from last layer4 block.
                   Shape: (B, seq_len, seq_len) where seq_len = num_patches + 1.
        output: The image features returned by encode_image.
    """
    visual = model.visual if hasattr(model, 'visual') else model
    layer4 = visual.transformer.layer4
    captures = []

    # Only patch the LAST block in layer4 for efficiency
    for i, block in enumerate(layer4):
        if i != len(layer4) - 1:
            # Skip non-last blocks
            continue

        orig_fn = block.attention
        storage = {}

        def _make_patched(blk, store):
            def _patched_attention(x):
                mask = blk.attn_mask
                if mask is not None:
                    mask = mask.to(dtype=x.dtype, device=x.device)
                out, weights = blk.attn(x, x, x, need_weights=True, attn_mask=mask)
                store['weights'] = weights.detach()
                return out
            return _patched_attention

        block.attention = _make_patched(block, storage)
        captures.append((block, orig_fn, storage))

    with torch.no_grad():
        output = model.encode_image(images) if hasattr(model, 'encode_image') else model(images)

    attn_maps = []
    for block, orig_fn, storage in captures:
        block.attention = orig_fn
        attn_maps.append(storage['weights'])

    return attn_maps, output


def _remap_vit_state_dict(state_dict, layer_groups):
    """Remap flat visual.transformer.resblocks.N keys to grouped visual.transformer.layerX.N keys."""
    new_state_dict = {}
    cumulative = [0]
    for g in layer_groups:
        cumulative.append(cumulative[-1] + g)

    for key, value in state_dict.items():
        if "visual.transformer.resblocks." in key:
            parts = key.split(".")
            block_idx = int(parts[3])  # visual.transformer.resblocks.<N>
            rest = ".".join(parts[4:])

            for group_idx in range(len(layer_groups)):
                if cumulative[group_idx] <= block_idx < cumulative[group_idx + 1]:
                    local_idx = block_idx - cumulative[group_idx]
                    new_key = f"visual.transformer.layer{group_idx + 1}.{local_idx}.{rest}"
                    new_state_dict[new_key] = value
                    break
        else:
            new_state_dict[key] = value

    return new_state_dict


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    # Use (2, 2, 18, 2) grouping for ViT-L/14 (24 layers)
    vision_layer_groups = None
    if vit and vision_layers == 24:
        vision_layer_groups = (2, 2, 18, 2)

    model = GroupedCLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,
        vision_layer_groups=vision_layer_groups
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    # Remap flat resblocks keys to grouped layer keys for the vision transformer
    if vision_layer_groups is not None:
        state_dict = _remap_vit_state_dict(state_dict, vision_layer_groups)

    convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()
