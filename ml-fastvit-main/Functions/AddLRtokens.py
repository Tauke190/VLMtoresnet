import torch
import torch.nn as nn
import torch.nn.functional as F

class AddLRtokens(nn.Module):
    def __init__(self, c1, c2, c3, c4, base_hw=(32, 32), mode="bicubic"):
        super().__init__()
        h0, w0 = base_hw
        # trainable prompts (batch dim = 1 so it broadcasts over B)
        self.p1 = nn.Parameter(torch.zeros(1, c1, h0, w0))
        self.p2 = nn.Parameter(torch.zeros(1, c2, h0 // 2, w0 // 2))
        self.p3 = nn.Parameter(torch.zeros(1, c3, h0 // 4, w0 // 4))
        self.p4 = nn.Parameter(torch.zeros(1, c4, h0 // 8, w0 // 8))
        self.mode = mode

    def _add_prompt(self, x, p):
        # x: [B,C,H,W], p: [1,C,h0,w0]
        H, W = x.shape[-2:]
        p_up = F.interpolate(p, size=(H, W), mode=self.mode,
                             align_corners=False if self.mode in ("bilinear", "bicubic") else None)
        return x + p_up  # broadcast over batch

    def forward(self, f1, f2, f3, f4):
        f1 = self._add_prompt(f1, self.p1)
        f2 = self._add_prompt(f2, self.p2)
        f3 = self._add_prompt(f3, self.p3)
        f4 = self._add_prompt(f4, self.p4)
        return f1, f2, f3, f4
