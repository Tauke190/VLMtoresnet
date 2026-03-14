import torch
from torch import nn
from torch.nn import functional as F
from functools import reduce
from operator import mul


class NonLocalBlock2d(nn.Module):
    """Single-head Non-Local Block using 1x1 Conv2d projections."""

    def __init__(self, in_channels, inter_channels=None, bn_layer=True, name=None):
        super().__init__()
        self.name = name or "NonLocalBlock2d"

        self.in_channels = in_channels
        self.inter_channels = inter_channels or max(in_channels // 2, 1)
        self.scale = self.inter_channels ** -0.5

        self.theta = nn.Conv2d(in_channels, self.inter_channels, 1)
        self.theta_bn = nn.BatchNorm2d(self.inter_channels)
        self.phi   = nn.Conv2d(in_channels, self.inter_channels, 1)
        self.phi_bn = nn.BatchNorm2d(self.inter_channels)
        self.g     = nn.Conv2d(in_channels, self.inter_channels, 1)

        if bn_layer:
            self.W_z = nn.Sequential(
                nn.Conv2d(self.inter_channels, in_channels, 1),
                nn.BatchNorm2d(in_channels),
            )
            nn.init.constant_(self.W_z[1].weight, 0)
            nn.init.constant_(self.W_z[1].bias, 0)
        else:
            self.W_z = nn.Conv2d(self.inter_channels, in_channels, 1)
            nn.init.constant_(self.W_z.weight, 0)
            nn.init.constant_(self.W_z.bias, 0)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        theta_proj = self.theta_bn(self.theta(x))  # shape: (B, inter_channels, H, W)
        theta_x = theta_proj.view(B, self.inter_channels, N).permute(0, 2, 1)

        phi_proj = self.phi_bn(self.phi(x))  # shape: (B, inter_channels, H, W)
        phi_x = phi_proj.view(B, self.inter_channels, N)
        g_x   = self.g(x).view(B, self.inter_channels, N).permute(0, 2, 1)

        with torch.cuda.amp.autocast(enabled=False):
            attn = torch.bmm(theta_x.float(), phi_x.float()) * self.scale
            
            # NaN Debug Logging
            if self.training:
                with torch.no_grad():
                    a_min, a_max = attn.min().item(), attn.max().item()
                    if torch.isnan(attn).any() or torch.isinf(attn).any() or a_max > 60000:
                        print(f"[NAN_DEBUG] {self.name} | ATTN STATS | min: {a_min:.2f}, max: {a_max:.2f}")
            
            attn = F.softmax(attn, dim=-1)
        attn = attn.to(g_x.dtype)

        y = torch.bmm(attn, g_x)
        y = y.permute(0, 2, 1).contiguous().view(B, self.inter_channels, H, W)

        return self.W_z(y) + x


class NonLocalBlockLinear(nn.Module):
    """Single-head Non-Local Block using nn.Linear projections."""

    def __init__(self, in_channels, inter_channels=None, bn_layer=True, name=None):
        super().__init__()
        self.name = name or "NonLocalBlockLinear"

        self.in_channels = in_channels
        self.inter_channels = inter_channels or max(in_channels // 2, 1)
        self.scale = self.inter_channels ** -0.5

        self.theta = nn.Linear(in_channels, self.inter_channels)
        self.theta_bn = nn.BatchNorm1d(self.inter_channels)
        self.phi   = nn.Linear(in_channels, self.inter_channels)
        self.phi_bn = nn.BatchNorm1d(self.inter_channels)
        self.g     = nn.Linear(in_channels, self.inter_channels)

        self.W_z_linear = nn.Linear(self.inter_channels, in_channels)

        if bn_layer:
            self.bn = nn.BatchNorm2d(in_channels)
            nn.init.constant_(self.bn.weight, 0)
            nn.init.constant_(self.bn.bias, 0)
        else:
            self.bn = None
            nn.init.constant_(self.W_z_linear.weight, 0)
            nn.init.constant_(self.W_z_linear.bias, 0)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        x_flat = x.view(B, C, N).permute(0, 2, 1)

        Q_proj = self.theta(x_flat)  # shape: (B, N, inter_channels)
        Q = self.theta_bn(Q_proj.transpose(1, 2)).transpose(1, 2)  # BatchNorm1d expects (B, C, N)
        K_proj = self.phi(x_flat)
        K = self.phi_bn(K_proj.transpose(1, 2)).transpose(1, 2)
        V = self.g(x_flat)

        with torch.cuda.amp.autocast(enabled=False):
            attn = torch.bmm(Q.float(), K.float().transpose(1, 2)) * self.scale

            # NaN Debug Logging
            if self.training:
                with torch.no_grad():
                    a_min, a_max = attn.min().item(), attn.max().item()
                    if torch.isnan(attn).any() or torch.isinf(attn).any() or a_max > 60000:
                        print(f"[NAN_DEBUG] {self.name} | ATTN STATS | min: {a_min:.2f}, max: {a_max:.2f}")

            attn = F.softmax(attn, dim=-1)
        attn = attn.to(V.dtype)

        y = self.W_z_linear(torch.bmm(attn, V))
        y = y.permute(0, 2, 1).contiguous().view(B, C, H, W)

        if self.bn is not None:
            y = self.bn(y)

        return y + x


class MultiHeadNonLocalBlock2d(nn.Module):
    """Multi-head Non-Local Block with parallel attention heads."""

    def __init__(self, in_channels, inter_channels=None, num_heads=4,
                 bn_layer=True, name=None):
        super().__init__()
        self.name = name or "MultiHeadNonLocalBlock2d"

        self.in_channels = in_channels
        self.inter_channels = inter_channels or max(in_channels // 2, 1)
        self.num_heads = num_heads

        assert self.inter_channels % num_heads == 0, \
            "inter_channels must be divisible by num_heads"

        self.head_dim = self.inter_channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.theta = nn.Conv2d(in_channels, self.inter_channels, 1)
        self.theta_bn = nn.BatchNorm2d(self.inter_channels)
        self.phi   = nn.Conv2d(in_channels, self.inter_channels, 1)
        self.phi_bn = nn.BatchNorm2d(self.inter_channels)
        self.g     = nn.Conv2d(in_channels, self.inter_channels, 1)

        if bn_layer:
            self.W_z = nn.Sequential(
                nn.Conv2d(self.inter_channels, in_channels, 1),
                nn.BatchNorm2d(in_channels),
            )
            nn.init.constant_(self.W_z[1].weight, 0)
            nn.init.constant_(self.W_z[1].bias, 0)
        else:
            self.W_z = nn.Conv2d(self.inter_channels, in_channels, 1)
            nn.init.constant_(self.W_z.weight, 0)
            nn.init.constant_(self.W_z.bias, 0)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        K = self.num_heads
        D = self.head_dim

        Q_proj = self.theta_bn(self.theta(x))  # shape: (B, inter_channels, H, W)
        Q = Q_proj.view(B, K, D, N).permute(0, 1, 3, 2).reshape(B * K, N, D)

        K_proj_raw = self.phi_bn(self.phi(x))  # shape: (B, inter_channels, H, W)
        K_proj = K_proj_raw.view(B, K, D, N).reshape(B * K, D, N)
        V = self.g(x).view(B, K, D, N).permute(0, 1, 3, 2).reshape(B * K, N, D)

        with torch.cuda.amp.autocast(enabled=False):
            attn = torch.bmm(Q.float(), K_proj.float()) * self.scale

            # NaN Debug Logging
            if self.training:
                with torch.no_grad():
                    a_min, a_max = attn.min().item(), attn.max().item()
                    if torch.isnan(attn).any() or torch.isinf(attn).any() or a_max > 60000:
                        print(f"[NAN_DEBUG] {self.name} | ATTN STATS | min: {a_min:.2f}, max: {a_max:.2f}")

            attn = F.softmax(attn, dim=-1)
        attn = attn.to(V.dtype)

        y = torch.bmm(attn, V)
        y = y.view(B, K, N, D).permute(0, 1, 3, 2).reshape(B, self.inter_channels, H, W)

        return self.W_z(y) + x


class NN_down_layer(nn.Module):
    """
    Helper module used by Non_local to project and (optionally) downsample
    feature maps into a sequence of tokens.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        scale: float = 1.0,
        linear: bool = False,
        kernel: int = 3,
        add_norm=None,
    ):
        super().__init__()
        self.linear = linear
        self.scale = scale
        self.kernel = kernel

        if linear:
            self.proj = nn.Linear(in_dim, out_dim)
            self.norm = add_norm(out_dim) if add_norm is not None else None
        else:
            padding = kernel // 2
            # Use strided conv for spatial downsampling.
            self.proj = nn.Conv2d(
                in_dim,
                out_dim,
                kernel_size=kernel,
                stride=kernel,
                padding=padding,
            )
            self.norm = add_norm(out_dim) if add_norm is not None else None

    def forward(self, x):
        """
        Args:
            x: tensor of shape (B, C, H, W)
        Returns:
            tokens: (B, N, out_dim)
            new_size: (H', W')
        """
        B, C, H, W = x.shape

        if self.linear:
            # No spatial downsampling, only channel projection.
            N = H * W
            x_flat = x.view(B, C, N).transpose(1, 2)  # (B, N, C)
            tokens = self.proj(x_flat)  # (B, N, out_dim)
            if self.norm is not None:
                # Treat tokens as (B*N, C) for norm if needed.
                tokens = self.norm(tokens.view(B * N, -1)).view(B, N, -1)
            tokens = tokens * self.scale
            return tokens, (H, W)

        # Convolutional downsampling path.
        y = self.proj(x)  # (B, out_dim, H', W')
        if self.norm is not None:
            y = self.norm(y)
        _, out_c, H_new, W_new = y.shape
        tokens = y.view(B, out_c, H_new * W_new).transpose(1, 2)  # (B, N, out_dim)
        tokens = tokens * self.scale
        return tokens, (H_new, W_new)


class NN_UPscale_layer(nn.Module):
    """
    Helper module used by Non_local to map token sequences back to feature maps
    at a desired spatial resolution.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        linear: bool = False,
        kernel: int = 3,
        add_norm=None,
    ):
        super().__init__()
        self.linear = linear

        if linear:
            self.proj = nn.Linear(in_dim, out_dim)
            self.norm = add_norm(out_dim) if add_norm is not None else None
        else:
            self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)
            self.norm = add_norm(out_dim) if add_norm is not None else None

    def forward(self, tokens, target_size, curr_size):
        """
        Args:
            tokens: (B, N, in_dim) where N = curr_H * curr_W
            target_size: (H_target, W_target)
            curr_size: (curr_H, curr_W)
        Returns:
            upsampled feature map of shape (B, out_dim, H_target, W_target)
        """
        B, N, C = tokens.shape
        curr_H, curr_W = curr_size
        assert N == curr_H * curr_W, "Token count does not match curr_size."

        if self.linear:
            tokens = self.proj(tokens)  # (B, N, out_dim)
            if self.norm is not None:
                tokens = self.norm(tokens.view(B * N, -1)).view(B, N, -1)
            out_c = tokens.shape[-1]
            y = tokens.transpose(1, 2).view(B, out_c, curr_H, curr_W)
        else:
            y = tokens.transpose(1, 2).view(B, C, curr_H, curr_W)
            y = self.proj(y)
            if self.norm is not None:
                y = self.norm(y)

        H_t, W_t = target_size
        if (curr_H, curr_W) != (H_t, W_t):
            y = F.interpolate(y, size=(H_t, W_t), mode="bilinear", align_corners=False)
        return y


class Non_local(nn.Module):
    """
    Multi-scale non-local block that operates jointly over a list of feature maps.
    """

    def __init__(
        self,
        linear_version: bool = False,
        kernel: int = 3,
        num_layers: int = 1,
        embed_dim: int = 96,
        add_norm=None,
    ):
        super().__init__()

        self.Query = nn.ModuleList()
        self.Key = nn.ModuleList()
        self.Value = nn.ModuleList()
        self.Out = nn.ModuleList()

        self.linear_version = linear_version
        self.kernel = kernel
        self.num_layers = num_layers

        for i_layer in range(num_layers):
            dim = int(embed_dim * 2 ** i_layer)
            scale = dim ** -0.5

            self.Query.append(
                NN_down_layer(
                    dim,
                    embed_dim // 2,
                    scale=scale,
                    linear=self.linear_version,
                    kernel=self.kernel,
                    add_norm=add_norm,
                )
            )
            self.Key.append(
                NN_down_layer(
                    dim,
                    embed_dim // 2,
                    linear=self.linear_version,
                    kernel=self.kernel,
                    add_norm=add_norm,
                )
            )
            self.Value.append(
                NN_down_layer(
                    dim,
                    self.kernel * self.kernel * (embed_dim // 2),
                    linear=self.linear_version,
                    kernel=self.kernel,
                    add_norm=add_norm,
                )
            )

            # self.Out[0] is ignored in enhancer
            if i_layer != 0:
                self.Out.append(
                    NN_UPscale_layer(
                        embed_dim // 2,
                        dim,
                        linear=self.linear_version,
                        kernel=self.kernel,
                        add_norm=add_norm,
                    )
                )
            else:
                self.Out.append(nn.Identity())

    def convert_to_KQV(self, y, i_layer):
        query_y, new_size = self.Query[i_layer](y)
        key_y, _ = self.Key[i_layer](y)
        value_y, _ = self.Value[i_layer](y)
        return query_y, key_y, value_y, new_size

    def compute_attention(self, Query, Key, Value):
        Query = torch.cat(Query, 1)
        Key = torch.cat(Key, 1)
        Value = torch.cat(Value, 1)

        Atten = Query @ Key.permute(0, 2, 1)
        Atten = Atten.softmax(-1)
        y = Atten @ Value
        return y

    def forward(self, outs):
        """
        Args:
            outs: list of feature maps, one per scale, each of shape (B, C_i, H_i, W_i)
        Returns:
            Updated list of feature maps with non-local enhancements applied.
        """
        Query = []
        Key = []
        Value = []
        spatial_res = []
        new_spatial_res = []

        for i in range(self.num_layers):
            out = outs[i]
            H, W = out.shape[2:]
            spatial_res.append((H, W))

            query_y, key_y, value_y, new_size = self.convert_to_KQV(out, i)
            new_spatial_res.append(new_size)
            Query.append(query_y)
            Key.append(key_y)
            Value.append(value_y)

        out = self.compute_attention(Query, Key, Value)
        curr = 0
        for i_layer in range(self.num_layers):
            spatial_dim = reduce(mul, new_spatial_res[i_layer])
            if i_layer != 0:
                chunk = out[:, curr : curr + spatial_dim, :]
                output = self.Out[i_layer](
                    chunk,
                    target_size=spatial_res[i_layer],
                    curr_size=new_spatial_res[i_layer],
                )
                outs[i_layer] = (outs[i_layer] + output).contiguous()
            curr += spatial_dim
        return outs

