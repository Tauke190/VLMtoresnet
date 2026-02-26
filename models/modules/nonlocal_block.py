import torch
from torch import nn
from torch.nn import functional as F


class NonLocalBlock2d(nn.Module):
    """Single-head Non-Local Block using 1x1 Conv2d projections."""

    def __init__(self, in_channels, inter_channels=None, bn_layer=True):
        super().__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels or max(in_channels // 2, 1)
        self.scale = self.inter_channels ** -0.5

        self.theta = nn.Conv2d(in_channels, self.inter_channels, 1)
        self.phi   = nn.Conv2d(in_channels, self.inter_channels, 1)
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

        theta_x = self.theta(x).view(B, self.inter_channels, N).permute(0, 2, 1)

        phi_x = self.phi(x).view(B, self.inter_channels, N)
        g_x   = self.g(x).view(B, self.inter_channels, N).permute(0, 2, 1)

        attn = torch.bmm(theta_x, phi_x) * self.scale
        attn = F.softmax(attn, dim=-1)

        y = torch.bmm(attn, g_x)
        y = y.permute(0, 2, 1).contiguous().view(B, self.inter_channels, H, W)

        return self.W_z(y) + x


class NonLocalBlockLinear(nn.Module):
    """Single-head Non-Local Block using nn.Linear projections."""

    def __init__(self, in_channels, inter_channels=None, bn_layer=True):
        super().__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels or max(in_channels // 2, 1)
        self.scale = self.inter_channels ** -0.5

        self.theta = nn.Linear(in_channels, self.inter_channels)
        self.phi   = nn.Linear(in_channels, self.inter_channels)
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

        Q = self.theta(x_flat)
        K = self.phi(x_flat)
        V = self.g(x_flat)

        attn = torch.bmm(Q, K.transpose(1, 2)) * self.scale
        attn = F.softmax(attn, dim=-1)

        y = self.W_z_linear(torch.bmm(attn, V))
        y = y.permute(0, 2, 1).contiguous().view(B, C, H, W)

        if self.bn is not None:
            y = self.bn(y)

        return y + x


class MultiHeadNonLocalBlock2d(nn.Module):
    """Multi-head Non-Local Block with parallel attention heads."""

    def __init__(self, in_channels, inter_channels=None, num_heads=4,
                 bn_layer=True):
        super().__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels or max(in_channels // 2, 1)
        self.num_heads = num_heads

        assert self.inter_channels % num_heads == 0, \
            "inter_channels must be divisible by num_heads"

        self.head_dim = self.inter_channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.theta = nn.Conv2d(in_channels, self.inter_channels, 1)
        self.phi   = nn.Conv2d(in_channels, self.inter_channels, 1)
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

        Q = self.theta(x).view(B, K, D, N).permute(0, 1, 3, 2).reshape(B * K, N, D)

        K_proj = self.phi(x).view(B, K, D, N).reshape(B * K, D, N)
        V = self.g(x).view(B, K, D, N).permute(0, 1, 3, 2).reshape(B * K, N, D)

        attn = torch.bmm(Q, K_proj) * self.scale
        attn = F.softmax(attn, dim=-1)

        y = torch.bmm(attn, V)
        y = y.view(B, K, N, D).permute(0, 1, 3, 2).reshape(B, self.inter_channels, H, W)

        return self.W_z(y) + x


if __name__ == '__main__':
    print("=== NonLocalBlock2d ===")
    img = torch.randn(2, 64, 14, 14)
    net = NonLocalBlock2d(64, 32)
    print(img.shape, "->", net(img).shape)

    print("\n=== NonLocalBlockLinear ===")
    img = torch.randn(2, 64, 14, 14)
    net = NonLocalBlockLinear(64, 32)
    print(img.shape, "->", net(img).shape)

    print("\n=== MultiHeadNonLocalBlock2d ===")
    for heads in [1, 2, 4, 8]:
        img = torch.randn(2, 64, 14, 14)
        net = MultiHeadNonLocalBlock2d(64, 32, num_heads=heads)
        print(img.shape, "->", net(img).shape)
