
from einops import rearrange, repeat
from typing import Optional, List, Tuple, Type, Any
import math 
import torch 
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
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





class ConvAdapter(nn.Module):
    """
    Convolutional Adapter module for efficient parameter tuning.
    Uses a bottleneck architecture: down-projection -> activation -> up-projection.
    """
    def __init__(self, in_channels, reduction_factor=4):
        super().__init__()
        hidden_dim = max(in_channels // reduction_factor, 8)

        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_dim, in_channels, kernel_size=1, bias=True)
        )

        self.adapter[0].weight.data.normal_(mean=0.0, std=1e-7)
        self.adapter[0].bias.data.normal_(mean=0.0, std=1e-7)
        self.adapter[2].weight.data.normal_(mean=0.0, std=1e-7)
        self.adapter[2].bias.data.normal_(mean=0.0, std=1e-7)


    def forward(self, x):
        return x + self.adapter(x)  # Residual connection 




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

        self.adapter1[0].weight.data.normal_(mean=0.0, std=1e-7)
        self.adapter1[0].bias.data.normal_(mean=0.0, std=1e-7)
        self.adapter1[2].weight.data.normal_(mean=0.0, std=1e-7)
        self.adapter1[2].bias.data.normal_(mean=0.0, std=1e-7)

        self.adapter2[0].weight.data.normal_(mean=0.0, std=1e-7)
        self.adapter2[0].bias.data.normal_(mean=0.0, std=1e-7)
        self.adapter2[2].weight.data.normal_(mean=0.0, std=1e-7)
        self.adapter2[2].bias.data.normal_(mean=0.0, std=1e-7)

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

        self.adapter1[0].weight.data.normal_(mean=0.0, std=1e-7)
        self.adapter1[0].bias.data.normal_(mean=0.0, std=1e-7)
        self.adapter1[2].weight.data.normal_(mean=0.0, std=1e-7)
        self.adapter1[2].bias.data.normal_(mean=0.0, std=1e-7)

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





class LoRALayer():
    def __init__(
            self,
            r: int,
            lora_alpha: int,
            lora_dropout: float,
            merge_weights: bool,
        ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights

class StreightConv_LoRA(nn.Conv2d, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.,
            groups = 1, 
            bias=False, 
            **kwargs
        ):

        nn.Conv2d.__init__(self, in_features, out_features, groups=groups, bias=bias, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=False )
        
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Conv2d(in_features, r, bias=False, **kwargs)
            self.lora_B = nn.Conv2d(r, out_features, kernel_size=1, bias=False)
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False
            if self.bias is not None:
                self.bias.requires_grad = False 
        self.reset_parameters()
         
    def reset_parameters(self):
        nn.Conv2d.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)

    def train(self, mode: bool = True):
        # kernel_size
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Conv2d.train(self, mode)
        self.lora_A.train(mode)
        self.lora_B.train(mode)
        
    def forward(self, x: torch.Tensor):
        result = F.conv2d(x, self.weight, bias=self.bias, stride=self.stride, padding=self.padding, groups=self.groups)
        result += self.lora_B( self.lora_A( self.lora_dropout(x) ) ) 
        return result
        
        

class MergedLinear_LoRA(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.,
            enable_lora: List[bool] = [False],
            fan_in_fan_out: bool = False,
            merge_weights: bool = True,
            bias=False, 
            **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, bias=bias, **kwargs)
        # will call reset_parameters via nn.Linear first 
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        assert out_features % len(enable_lora) == 0, \
            'The length of enable_lora must divide out_features'
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0 and any(enable_lora):
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r * sum(enable_lora), in_features)))
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r))
            )  # weights for Conv1D with groups=sum(enable_lora)
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            if bias:
                self.bias.requires_grad = False 
            # Compute the indices
            self.lora_ind = self.weight.new_zeros(
                (out_features,), dtype=torch.bool
            ).view(len(enable_lora), -1)  # (3d,) -> (3,d)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)  # (3d,)
        self.reset_parameters()
        # this will be second reset_parameters call after intial nn.Linear
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def zero_pad(self, x):
        # x -> (B,N,2d)
        # print(x.shape)
        result = x.new_zeros((*x.shape[:-1], self.out_features)) # result -> B,N,3d
        # result = x.new_zeros((self.out_features, *x.shape[1:]))
        result = result.view(-1, self.out_features)  # BN, 3d
        # result = result.view(self.out_features, -1)
        # print(result.shape)
        # print(self.out_features // len(self.enable_lora) * sum(self.enable_lora))
        result[:, self.lora_ind] = x.reshape(
            -1, self.out_features // len(self.enable_lora) * sum(self.enable_lora)
        )  # (BN,2d)
        # result[self.lora_ind, :] = x.reshape(
        #      self.out_features // len(self.enable_lora) * sum(self.enable_lora), -1
        # )  # (BN,2d)
        # print(result.shape)
        return result.view((*x.shape[:-1], self.out_features))

        # print(sum(x - result[self.lora_ind,:]))
        # print(result[1,:])
        # return result

    def zero_pad_weight(self,x):
        # x -> (B,N,2d)
        # print(x.shape)
        # result = x.new_zeros((*x.shape[:-1], self.out_features)) # result -> B,N,3d
        result = x.new_zeros((self.out_features, *x.shape[1:]))
        # result = result.view(-1, self.out_features)  # BN, 3d
        result = result.view(self.out_features, -1)
        # print(result.shape)
        # print(self.out_features // len(self.enable_lora) * sum(self.enable_lora))
        # result[:, self.lora_ind] = x.reshape(
        #     -1, self.out_features // len(self.enable_lora) * sum(self.enable_lora)
        # )  # (BN,2d)
        result[self.lora_ind, :] = x.reshape(
            self.out_features // len(self.enable_lora) * sum(self.enable_lora), -1
        )  # (BN,2d)
        # print(result.shape)
        # return result.view((*x.shape[:-1], self.out_features))

        # print(sum(x - result[self.lora_ind,:]))
        # print(result[1,:])
        return result

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0 and any(self.enable_lora):
                    delta_w = F.conv1d(
                        self.lora_A.data.unsqueeze(0),
                        self.lora_B.data.unsqueeze(-1),
                        groups=sum(self.enable_lora)
                    ).squeeze(0)
                    self.weight.data -= self.zero_pad_weight(T(delta_w * self.scaling))
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0 and any(self.enable_lora):
                    delta_w = F.conv1d(
                        self.lora_A.data.unsqueeze(0),
                        self.lora_B.data.unsqueeze(-1),
                        groups=sum(self.enable_lora)
                    ).squeeze(0)
                    self.weight.data += self.zero_pad_weight(T(delta_w * self.scaling))
                self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.merged:
            return F.linear(x, T(self.weight), bias=self.bias)
        else:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                after_A = F.linear(self.lora_dropout(x), self.lora_A)
                after_B = F.conv1d(
                    after_A.transpose(-2, -1),
                    self.lora_B.unsqueeze(-1),
                    groups=sum(self.enable_lora)
                ).transpose(-2, -1)
                result += self.zero_pad(after_B) * self.scaling
            return result

