import torch
from torch import nn
from torch.nn import functional as F


# =============================================================================
# 1-Head Conv2d Non-Local Block
# =============================================================================
class NonLocalBlock2d(nn.Module):
    """Single-head Non-Local Block using Conv2d (1x1) projections.

    Implements Embedded Gaussian variant from:
        "Non-local Neural Networks" (Wang et al., CVPR 2018)

    Dimensions (paper notation):
        Input x:      (B, C, H, W)
        theta(x):     (B, C', HW)    — query   (C' = inter_channels)
        phi(x):       (B, C', HW)    — key
        g(x):         (B, C', HW)    — value
        attn:         (B, HW, HW)    — softmax(theta^T @ phi)
        y = attn @ g^T: (B, HW, C')
        W_z(y):       (B, C, H, W)   — project back + residual
    """

    def __init__(self, in_channels, inter_channels=None, bn_layer=True,
                 use_softmax=True, use_maxpool=False):
        """
        Args:
            in_channels:    C — input channel size
            inter_channels: C' — bottleneck channel size (default: C // 2)
            bn_layer:       add BatchNorm after output projection W_z
            use_softmax:    normalize attention with softmax (True per paper)
            use_maxpool:    subsample key/value spatially with 2x2 max-pool
        """
        super(NonLocalBlock2d, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels or max(in_channels // 2, 1)
        self.use_softmax = use_softmax
        self.use_maxpool = use_maxpool

        # Query, Key, Value projections — 1x1 convolutions
        self.theta = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1)  # query
        self.phi   = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1)  # key
        self.g     = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1)  # value

        # Optional spatial subsampling of key & value
        if self.use_maxpool:
            self.pool = nn.MaxPool2d(kernel_size=2)

        # Output projection W_z: C' -> C
        if bn_layer:
            self.W_z = nn.Sequential(
                nn.Conv2d(self.inter_channels, self.in_channels, kernel_size=1),
                nn.BatchNorm2d(self.in_channels),
            )
            # Section 4.1: init BN so block starts as identity
            nn.init.constant_(self.W_z[1].weight, 0)
            nn.init.constant_(self.W_z[1].bias, 0)
        else:
            self.W_z = nn.Conv2d(self.inter_channels, self.in_channels, kernel_size=1)
            # Section 3.3: init W_z=0 so block is identity at start
            nn.init.constant_(self.W_z.weight, 0)
            nn.init.constant_(self.W_z.bias, 0)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            z: (B, C, H, W)  — same shape, with non-local residual added
        """
        B, C, H, W = x.shape
        N = H * W  # number of spatial positions

        # Query: (B, C', HW) -> (B, HW, C')
        theta_x = self.theta(x).view(B, self.inter_channels, N).permute(0, 2, 1)  # (B, HW, C')

        # Key & Value (optionally subsampled)
        x_kv = self.pool(x) if self.use_maxpool else x
        N_kv = x_kv.shape[2] * x_kv.shape[3]

        phi_x = self.phi(x_kv).view(B, self.inter_channels, N_kv)                 # (B, C', N_kv)
        g_x   = self.g(x_kv).view(B, self.inter_channels, N_kv).permute(0, 2, 1)  # (B, N_kv, C')

        # Attention: (B, HW, C') @ (B, C', N_kv) -> (B, HW, N_kv)
        attn = torch.bmm(theta_x, phi_x)

        if self.use_softmax:
            attn = F.softmax(attn, dim=-1)
        else:
            attn = attn / N_kv

        # Aggregate: (B, HW, N_kv) @ (B, N_kv, C') -> (B, HW, C')
        y = torch.bmm(attn, g_x)

        # Reshape back: (B, HW, C') -> (B, C', H, W)
        y = y.permute(0, 2, 1).contiguous().view(B, self.inter_channels, H, W)

        # Project back to C and add residual
        z = self.W_z(y) + x
        return z


# =============================================================================
# 1-Head Linear Non-Local Block (uses nn.Linear instead of Conv2d)
# =============================================================================
class NonLocalBlockLinear(nn.Module):
    """Single-head Non-Local Block using nn.Linear projections.

    Functionally equivalent to NonLocalBlock2d but uses linear layers
    operating on flattened spatial tokens instead of 1x1 convolutions.

    Dimensions:
        Input x:         (B, C, H, W)
        x_flat:          (B, HW, C)     — spatial positions as tokens
        theta(x_flat):   (B, HW, C')    — query
        phi(x_flat):     (B, HW, C')    — key
        g(x_flat):       (B, HW, C')    — value
        attn:            (B, HW, HW)    — softmax(Q @ K^T)
        y = attn @ V:    (B, HW, C')
        W_z(y):          (B, HW, C) -> (B, C, H, W) + residual
    """

    def __init__(self, in_channels, inter_channels=None, bn_layer=True,
                 use_softmax=True, use_maxpool=False):
        """
        Args:
            in_channels:    C — input channel size
            inter_channels: C' — bottleneck channel size (default: C // 2)
            bn_layer:       add BatchNorm after output projection W_z
            use_softmax:    normalize attention with softmax (True per paper)
            use_maxpool:    subsample key/value spatially with 2x2 max-pool
        """
        super(NonLocalBlockLinear, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels or max(in_channels // 2, 1)
        self.use_softmax = use_softmax
        self.use_maxpool = use_maxpool

        # Query, Key, Value projections — nn.Linear (channel-wise)
        self.theta = nn.Linear(self.in_channels, self.inter_channels)   # query
        self.phi   = nn.Linear(self.in_channels, self.inter_channels)   # key
        self.g     = nn.Linear(self.in_channels, self.inter_channels)   # value

        # Optional spatial subsampling of key & value
        if self.use_maxpool:
            self.pool = nn.MaxPool2d(kernel_size=2)

        # Output projection W_z: C' -> C
        self.W_z_linear = nn.Linear(self.inter_channels, self.in_channels)
        if bn_layer:
            self.bn = nn.BatchNorm2d(self.in_channels)
            nn.init.constant_(self.bn.weight, 0)
            nn.init.constant_(self.bn.bias, 0)
        else:
            self.bn = None
            nn.init.constant_(self.W_z_linear.weight, 0)
            nn.init.constant_(self.W_z_linear.bias, 0)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            z: (B, C, H, W)
        """
        B, C, H, W = x.shape
        N = H * W

        # Flatten spatial dims: (B, C, H, W) -> (B, HW, C)
        x_flat = x.view(B, C, N).permute(0, 2, 1)  # (B, HW, C)

        # Query: (B, HW, C) -> (B, HW, C')
        Q = self.theta(x_flat)  # (B, HW, C')

        # Key & Value (optionally subsampled)
        if self.use_maxpool:
            x_kv = self.pool(x)
            N_kv = x_kv.shape[2] * x_kv.shape[3]
            x_kv_flat = x_kv.view(B, C, N_kv).permute(0, 2, 1)  # (B, N_kv, C)
        else:
            x_kv_flat = x_flat
            N_kv = N

        K = self.phi(x_kv_flat)  # (B, N_kv, C')
        V = self.g(x_kv_flat)    # (B, N_kv, C')

        # Attention: (B, HW, C') @ (B, C', N_kv) -> (B, HW, N_kv)
        attn = torch.bmm(Q, K.transpose(1, 2))

        if self.use_softmax:
            attn = F.softmax(attn, dim=-1)
        else:
            attn = attn / N_kv

        # Aggregate: (B, HW, N_kv) @ (B, N_kv, C') -> (B, HW, C')
        y = torch.bmm(attn, V)

        # Project back: (B, HW, C') -> (B, HW, C)
        y = self.W_z_linear(y)

        # Reshape: (B, HW, C) -> (B, C, H, W)
        y = y.permute(0, 2, 1).contiguous().view(B, C, H, W)

        if self.bn is not None:
            y = self.bn(y)

        # Residual connection
        z = y + x
        return z


# =============================================================================
# K Multi-Head Non-Local Block (Conv2d version)
# =============================================================================
class MultiHeadNonLocalBlock2d(nn.Module):
    """Multi-head Non-Local Block — K parallel attention heads.

    Splits inter_channels across K heads, computes independent attention
    per head, concatenates, and projects back. Analogous to multi-head
    attention in Transformers.

    Per-head dimensions (head_dim = inter_channels // num_heads):
        Q_k:  (B, HW, head_dim)
        K_k:  (B, head_dim, N_kv)
        V_k:  (B, N_kv, head_dim)
        attn_k: (B, HW, N_kv)    — softmax(Q_k @ K_k)
        y_k:    (B, HW, head_dim) — attn_k @ V_k

    After concat over K heads:
        y:    (B, HW, inter_channels)
        W_z:  (B, C, H, W) + residual
    """

    def __init__(self, in_channels, inter_channels=None, num_heads=4,
                 bn_layer=True, use_softmax=True, use_maxpool=False):
        """
        Args:
            in_channels:    C — input channel size
            inter_channels: C' — total bottleneck channels across all heads
                            (default: C // 2, must be divisible by num_heads)
            num_heads:      K — number of parallel attention heads
            bn_layer:       add BatchNorm after output projection W_z
            use_softmax:    normalize attention with softmax (True per paper)
            use_maxpool:    subsample key/value spatially with 2x2 max-pool
        """
        super(MultiHeadNonLocalBlock2d, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels or max(in_channels // 2, 1)
        self.num_heads = num_heads
        self.use_softmax = use_softmax
        self.use_maxpool = use_maxpool

        assert self.inter_channels % num_heads == 0, \
            f"inter_channels ({self.inter_channels}) must be divisible by num_heads ({num_heads})"
        self.head_dim = self.inter_channels // num_heads

        # Shared Q, K, V projections — project to all heads at once
        self.theta = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1)  # query
        self.phi   = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1)  # key
        self.g     = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1)  # value

        # Optional spatial subsampling
        if self.use_maxpool:
            self.pool = nn.MaxPool2d(kernel_size=2)

        # Output projection W_z: C' -> C
        if bn_layer:
            self.W_z = nn.Sequential(
                nn.Conv2d(self.inter_channels, self.in_channels, kernel_size=1),
                nn.BatchNorm2d(self.in_channels),
            )
            nn.init.constant_(self.W_z[1].weight, 0)
            nn.init.constant_(self.W_z[1].bias, 0)
        else:
            self.W_z = nn.Conv2d(self.inter_channels, self.in_channels, kernel_size=1)
            nn.init.constant_(self.W_z.weight, 0)
            nn.init.constant_(self.W_z.bias, 0)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            z: (B, C, H, W)
        """
        B, C, H, W = x.shape
        N = H * W
        K = self.num_heads
        D = self.head_dim

        # Query: (B, C', H, W) -> (B, K, D, N) -> (B*K, N, D)
        Q = self.theta(x).view(B, K, D, N).permute(0, 1, 3, 2).reshape(B * K, N, D)

        # Key & Value (optionally subsampled)
        x_kv = self.pool(x) if self.use_maxpool else x
        N_kv = x_kv.shape[2] * x_kv.shape[3]

        # Key: (B, C', H_kv, W_kv) -> (B*K, D, N_kv)
        K_proj = self.phi(x_kv).view(B, K, D, N_kv).reshape(B * K, D, N_kv)

        # Value: (B, C', H_kv, W_kv) -> (B*K, N_kv, D)
        V = self.g(x_kv).view(B, K, D, N_kv).permute(0, 1, 3, 2).reshape(B * K, N_kv, D)

        # Attention per head: (B*K, N, D) @ (B*K, D, N_kv) -> (B*K, N, N_kv)
        attn = torch.bmm(Q, K_proj)

        if self.use_softmax:
            attn = F.softmax(attn, dim=-1)
        else:
            attn = attn / N_kv

        # Aggregate: (B*K, N, N_kv) @ (B*K, N_kv, D) -> (B*K, N, D)
        y = torch.bmm(attn, V)

        # Merge heads: (B*K, N, D) -> (B, K, N, D) -> (B, K*D, H, W) = (B, C', H, W)
        y = y.view(B, K, N, D).permute(0, 1, 3, 2).reshape(B, self.inter_channels, H, W)

        # Project back to C and add residual
        z = self.W_z(y) + x
        return z


if __name__ == '__main__':
    import torch

    print("=== NonLocalBlock2d (1-head, Conv2d) ===")
    for pool in [False, True]:
        img = torch.randn(2, 64, 14, 14)
        net = NonLocalBlock2d(in_channels=64, inter_channels=32, use_softmax=True, use_maxpool=pool)
        out = net(img)
        print(f"  maxpool={pool}: {img.shape} -> {out.shape}")

    print("\n=== NonLocalBlockLinear (1-head, nn.Linear) ===")
    for pool in [False, True]:
        img = torch.randn(2, 64, 14, 14)
        net = NonLocalBlockLinear(in_channels=64, inter_channels=32, use_softmax=True, use_maxpool=pool)
        out = net(img)
        print(f"  maxpool={pool}: {img.shape} -> {out.shape}")

    print("\n=== MultiHeadNonLocalBlock2d (4-head, Conv2d) ===")
    for heads in [1, 2, 4, 8]:
        img = torch.randn(2, 64, 14, 14)
        net = MultiHeadNonLocalBlock2d(in_channels=64, inter_channels=32, num_heads=heads, use_softmax=True)
        out = net(img)
        print(f"  heads={heads}: {img.shape} -> {out.shape}")


