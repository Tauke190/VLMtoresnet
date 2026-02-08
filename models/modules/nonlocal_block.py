import torch
import torch.nn as nn
import torch.nn.functional as F


class NonLocalBlock2d(nn.Module):
    """
    Non-local block (Wang et al., CVPR 2018) for 2D feature maps.
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

        self.inter_channels = inter_channels or max(1, in_channels // 2)
        self.use_maxpool = use_maxpool
        self.use_softmax = use_softmax
        self.use_scale = use_scale
        self.use_bn = use_bn

        self.theta = nn.Conv2d(in_channels, self.inter_channels, 1, bias=True)
        self.phi = nn.Conv2d(in_channels, self.inter_channels, 1, bias=True)
        self.g = nn.Conv2d(in_channels, self.inter_channels, 1, bias=True)
        self.W_z = nn.Conv2d(self.inter_channels, in_channels, 1, bias=True)

        if self.use_bn:
            self.bn = nn.BatchNorm2d(in_channels)
            nn.init.constant_(self.bn.weight, bn_init_gamma)
            nn.init.constant_(self.bn.bias, 0.0)

        if self.use_maxpool:
            self.pool = nn.MaxPool2d(kernel_size=max_pool_stride, stride=max_pool_stride)

        self._init_weights(conv_init_std)

    def _init_weights(self, std: float):
        for conv in [self.theta, self.phi, self.g, self.W_z]:
            nn.init.normal_(conv.weight, mean=0.0, std=std)
            if conv.bias is not None:
                nn.init.constant_(conv.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape

        theta = self.theta(x).reshape(B, self.inter_channels, -1)

        phi_input = self.pool(x) if self.use_maxpool else x
        phi = self.phi(phi_input).reshape(B, self.inter_channels, -1)
        g = self.g(phi_input).reshape(B, self.inter_channels, -1)

        attn = torch.bmm(theta.transpose(1, 2), phi)

        if self.use_softmax:
            if self.use_scale:
                attn = attn * (self.inter_channels ** -0.5)
            attn = F.softmax(attn, dim=-1)
        else:
            attn = attn / attn.shape[-1]

        y = torch.bmm(g, attn.transpose(1, 2)).reshape(B, self.inter_channels, H, W)

        z = self.W_z(y)
        if self.use_bn:
            z = self.bn(z)

        return x + z
