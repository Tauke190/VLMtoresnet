
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn as nn
import torch.nn.functional as F
from models.fastvit import AttentionBlock, RepMixerBlock


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        self.apply(_init_weights)


# class ConvAdapter(nn.Module):
#     """
#     Convolutional Adapter module for efficient parameter tuning.
#     Uses a bottleneck architecture: down-projection -> activation -> up-projection.
#     """
#     def __init__(self, in_channels, reduction_factor=4):
#         super().__init__()
#         hidden_dim = max(in_channels // reduction_factor, 8)

#         self.adapter = nn.Sequential(
#             nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=True),
#             nn.GELU(),
#             nn.Conv2d(hidden_dim, in_channels, kernel_size=1, bias=True)
#         )

#         self.adapter[0].weight.data.normal_(mean=0.0, std=0.02)
#         self.adapter[0].bias.data.normal_(mean=0.0)
#         self.adapter[2].weight.data.normal_(mean=0.0, std=0.02)
#         self.adapter[2].bias.data.normal_(mean=0.0)


#     def forward(self, x):
#         return x + self.adapter(x)  # Residual connection 


class AttentionBlock_Adapter(AttentionBlock):
    def __init__(self, dim=-1, reduction_factor=-1, **kwargs):
        super().__init__(dim=dim, **kwargs)
        hidden_dim = max(dim // reduction_factor, 8)

        self.adapter1 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=True)
        )

        self.adapter2 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=True)
        )

        self.adapter1[0].weight.data.normal_(mean=0.0, std=0.02)
        self.adapter1[0].bias.data.normal_(mean=0.0, std=0)
        self.adapter1[2].weight.data.normal_(mean=0.0, std=0.02)
        self.adapter1[2].bias.data.normal_(mean=0.0, std=0)

        self.adapter2[0].weight.data.normal_(mean=0.0, std=0.02)
        self.adapter2[0].bias.data.normal_(mean=0.0, std=0)
        self.adapter2[2].weight.data.normal_(mean=0.0, std=0.02)
        self.adapter2[2].bias.data.normal_(mean=0.0, std=0)

    def forward(self, x):
        if self.use_layer_scale:
            z = self.token_mixer(self.norm(x))
            z = self.adapter1(z)
            z = self.layer_scale_1 * z
            x = x + self.drop_path( z )

            z = self.convffn(x)
            z = self.adapter2(z)
            z = self.layer_scale_2 * z
            x = x + self.drop_path( z )
        else:
            z = self.token_mixer(self.norm(x))
            z = self.adapter1(z)
            x = x + self.drop_path( z )
            
            z =  self.convffn(x)
            z = self.adapter2(z)
            x = x + self.drop_path(z)
        return x


class RepMixerBlock_Adapter(RepMixerBlock):
    
    def __init__(self, dim=-1, reduction_factor=-1, **kwargs):
        super().__init__(dim=dim, **kwargs)

        hidden_dim = max(dim // reduction_factor, 8)

        self.adapter1 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=True)
        )

        self.adapter1[0].weight.data.normal_(mean=0.0, std=0.02)
        self.adapter1[0].bias.data.normal_(mean=0.0, std=0)
        self.adapter1[2].weight.data.normal_(mean=0.0, std=0.02)
        self.adapter1[2].bias.data.normal_(mean=0.0, std=0)

    def forward(self, x):
        if self.use_layer_scale:
            x = self.token_mixer(x)
            z = self.convffn(x)
            z = self.adapter1(z)
            z = self.layer_scale * z
            x = x + self.drop_path( z )
        else:
            x = self.token_mixer(x)
            z = self.convffn(x)
            z = self.adapter1(z)
            x = x + self.drop_path( z )
        return x


class ConvLoRA(nn.Module):
    """
    LoRA (Low-Rank Adaptation) for convolutional layers.
    Applies low-rank decomposition: output = Conv(x) + scale * (B * A)(x)
    where A projects down to rank r, and B projects back up.
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, rank=8, alpha=16, stride=1, padding=0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank  # Scaling factor for LoRA

        # Low-rank matrices: A (down-projection) and B (up-projection)
        # A: [in_channels, rank, kernel_size, kernel_size]
        # B: [out_channels, rank, 1, 1]
        self.lora_A = nn.Conv2d(in_channels, rank, kernel_size=kernel_size,
                                stride=stride, padding=padding, bias=False)
        self.lora_B = nn.Conv2d(rank, out_channels, kernel_size=1, bias=False)

        # Initialize A with small random values (Kaiming), B with zeros
        # This ensures LoRA starts as identity (no effect initially)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        # LoRA path: x -> A (down) -> B (up)
        lora_out = self.lora_B(self.lora_A(x))
        return x + self.scaling * lora_out  # Residual connection with scaling

