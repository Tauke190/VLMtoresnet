"""
Non-Local Block for 2D feature maps (images).

Implements the non-local operation from:
    "Non-local Neural Networks" - Wang et al., CVPR 2018
    Paper: https://arxiv.org/abs/1711.07971
    Reference: https://github.com/facebookresearch/video-nonlocal-net

Adapted from the original 3D (video) implementation to 2D (image) feature maps
for integration with FastViT stages.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NonLocalBlock2d(nn.Module):
    """
    Non-Local Block for 2D spatial feature maps.

    Computes:
        z_i = x_i + W_z * ( (1/C(x)) * sum_j f(x_i, x_j) * g(x_j) )

    Using the Embedded Gaussian instantiation (default in the paper):
        f(x_i, x_j) = exp(theta(x_i)^T * phi(x_j))
        C(x) = sum_j f(x_i, x_j)   (i.e., softmax normalization)

    Architecture (following FB's reference implementation):
        theta: 1x1 Conv2d, C -> C_inner  (query, no spatial subsampling)
        phi:   1x1 Conv2d, C -> C_inner  (key, optional maxpool subsampling)
        g:     1x1 Conv2d, C -> C_inner  (value, optional maxpool subsampling)
        W_z:   1x1 Conv2d, C_inner -> C  (output projection)
        BN:    BatchNorm2d with gamma initialized to 0 (identity at init)

    Args:
        in_channels (int): Number of input/output channels (C).
        inter_channels (int or None): Bottleneck channels (C_inner).
            Default: in_channels // 2 (as in the paper/FB code: int(dim_in / 2)).
        use_maxpool (bool): Subsample phi and g spatially with 2x2 MaxPool.
            Default: True (cfg.NONLOCAL.USE_MAXPOOL = True in FB code).
        use_softmax (bool): Use embedded Gaussian (softmax) normalization.
            Default: True (cfg.NONLOCAL.USE_SOFTMAX = True in FB code).
        use_scale (bool): Scale attention logits by dim^{-0.5}.
            Default: True (cfg.NONLOCAL.USE_SCALE = True in FB code).
        use_bn (bool): Apply BatchNorm after the output 1x1 conv.
            Default: True (cfg.NONLOCAL.USE_BN = True in FB code).
        bn_init_gamma (float): Initial value for BN gamma (weight).
            Default: 0.0 (cfg.NONLOCAL.BN_INIT_GAMMA = 0.0 in FB code).
            Setting to 0 makes the block an identity at initialization.
        conv_init_std (float): Std for Gaussian init of 1x1 conv weights.
            Default: 0.01 (cfg.NONLOCAL.CONV_INIT_STD = 0.01 in FB code).
        max_pool_stride (int): Stride for spatial subsampling MaxPool.
            Default: 2 (max_pool_stride=2 in FB code).
    """

    def __init__(
        self,
        in_channels: int,
        inter_channels: int = None,
        use_maxpool: bool = True,
        use_softmax: bool = True,
        use_scale: bool = True,
        use_bn: bool = True,
        bn_init_gamma: float = 0.0,
        conv_init_std: float = 0.01,
        max_pool_stride: int = 2,
    ):
        super().__init__()

        self.in_channels = in_channels
        # Bottleneck: inter_channels = in_channels // 2 (paper default)
        self.inter_channels = inter_channels if inter_channels is not None else in_channels // 2
        # Ensure at least 1 channel
        if self.inter_channels == 0:
            self.inter_channels = 1

        self.use_maxpool = use_maxpool
        self.use_softmax = use_softmax
        self.use_scale = use_scale
        self.use_bn = use_bn

        # ---- Projection layers (1x1 convolutions) ----
        # theta: query projection (applied to original spatial resolution)
        # Matches: model.ConvNd(cur, prefix + '_theta', dim_in, dim_inner, [1,1,1], ...)
        self.theta = nn.Conv2d(
            in_channels, self.inter_channels,
            kernel_size=1, stride=1, padding=0, bias=True
        )

        # phi: key projection (applied after optional maxpool)
        # Matches: model.ConvNd(max_pool, prefix + '_phi', dim_in, dim_inner, [1,1,1], ...)
        self.phi = nn.Conv2d(
            in_channels, self.inter_channels,
            kernel_size=1, stride=1, padding=0, bias=True
        )

        # g: value projection (applied after optional maxpool)
        # Matches: model.ConvNd(max_pool, prefix + '_g', dim_in, dim_inner, [1,1,1], ...)
        self.g = nn.Conv2d(
            in_channels, self.inter_channels,
            kernel_size=1, stride=1, padding=0, bias=True
        )

        # W_z: output projection back to original channel dimension
        # Matches: model.ConvNd(blob_out, prefix + '_out', dim_inner, dim_out, [1,1,1], ...)
        self.W_z = nn.Conv2d(
            self.inter_channels, in_channels,
            kernel_size=1, stride=1, padding=0, bias=True
        )

        # ---- Optional BatchNorm with zero-init gamma ----
        # Matches: model.SpatialBN(...) followed by ConstantFill(BN_INIT_GAMMA=0.0)
        if self.use_bn:
            self.bn = nn.BatchNorm2d(in_channels)
            # Zero-init gamma so the block starts as identity via residual
            nn.init.constant_(self.bn.weight, bn_init_gamma)
            nn.init.constant_(self.bn.bias, 0.0)

        # ---- Optional MaxPool for spatial subsampling of phi and g ----
        # Matches: model.MaxPool(cur, prefix + '_pool', kernels=[1, 2, 2], strides=[1, 2, 2])
        if self.use_maxpool:
            self.pool = nn.MaxPool2d(
                kernel_size=max_pool_stride,
                stride=max_pool_stride,
            )

        # ---- Weight initialization ----
        # Matches: weight_init=('GaussianFill', {'std': CONV_INIT_STD})
        # and bias_init=('ConstantFill', {'value': 0.})
        self._init_weights(conv_init_std)

    def _init_weights(self, std: float = 0.01):
        """Initialize conv weights with Gaussian and biases with zeros.
        
        Follows FB code:
            weight_init=('GaussianFill', {'std': cfg.NONLOCAL.CONV_INIT_STD})
            bias_init=('ConstantFill', {'value': 0.})
        """
        for conv in [self.theta, self.phi, self.g, self.W_z]:
            nn.init.normal_(conv.weight, mean=0.0, std=std)
            if conv.bias is not None:
                nn.init.constant_(conv.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Non-Local Block.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Output tensor of shape (B, C, H, W)
            z = x + W_z(NonLocal(x))  [residual connection]
        """
        B, C, H, W = x.shape

        # ---- Step 1: Compute theta (query) ----
        # theta: (B, C, H, W) -> (B, C_inner, H, W)
        # Then flatten spatial: (B, C_inner, H*W)
        theta = self.theta(x)
        theta = theta.view(B, self.inter_channels, -1)  # (B, C_inner, N)

        # ---- Step 2: Compute phi (key) with optional spatial subsampling ----
        # phi: (B, C, H, W) -> maxpool -> (B, C, H', W') -> conv -> (B, C_inner, H', W')
        # Then flatten spatial: (B, C_inner, H'*W')
        if self.use_maxpool:
            phi_input = self.pool(x)
        else:
            phi_input = x
        phi = self.phi(phi_input)
        phi = phi.view(B, self.inter_channels, -1)  # (B, C_inner, N')

        # ---- Step 3: Compute g (value) with optional spatial subsampling ----
        # g: same subsampling as phi
        # (B, C_inner, N')
        g = self.g(phi_input)
        g = g.view(B, self.inter_channels, -1)  # (B, C_inner, N')

        # ---- Step 4: Compute affinity / attention ----
        # theta^T @ phi: (B, N, C_inner) @ (B, C_inner, N') => (B, N, N')
        # Matches: model.net.BatchMatMul([theta, phi], ..., trans_a=1)
        theta_phi = torch.bmm(theta.transpose(1, 2), phi)  # (B, N, N')

        if self.use_softmax:
            # Embedded Gaussian: softmax normalization
            if self.use_scale:
                # Scale by dim^{-0.5} as in the paper
                # Matches: model.Scale(theta_phi, theta_phi, scale=dim_inner**-.5)
                theta_phi = theta_phi * (self.inter_channels ** -0.5)
            # Softmax along last dim (over key positions)
            # Matches: model.Softmax(theta_phi_sc, ..., axis=2)
            p = F.softmax(theta_phi, dim=-1)  # (B, N, N')
        else:
            # Dot product: normalize by count
            # Matches the else branch in FB code with ConstantFill + ReduceBackSum + Div
            N_prime = theta_phi.shape[-1]
            p = theta_phi / N_prime

        # ---- Step 5: Compute attention-weighted values ----
        # g @ p^T: (B, C_inner, N') @ (B, N', N) => (B, C_inner, N)
        # Matches: model.net.BatchMatMul([g, p], ..., trans_b=1)
        t = torch.bmm(g, p.transpose(1, 2))  # (B, C_inner, N)

        # ---- Step 6: Reshape back to spatial ----
        # (B, C_inner, N) -> (B, C_inner, H, W)
        t = t.view(B, self.inter_channels, H, W)

        # ---- Step 7: Output projection ----
        # W_z: (B, C_inner, H, W) -> (B, C, H, W)
        z = self.W_z(t)

        # ---- Step 8: Optional BatchNorm (with gamma=0 init) ----
        if self.use_bn:
            z = self.bn(z)

        # ---- Step 9: Residual connection ----
        # Matches: blob_out = model.net.Sum([blob_in, blob_out], prefix + "_sum")
        out = x + z

        return out
