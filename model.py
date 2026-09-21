"""Fixed 2x Qwen adaptation of SesquiLSR (see LICENSE)."""

import torch
from torch import nn
from torch.nn import functional as F


class IRLayerNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(128))

    def forward(self, x):
        var, mean = torch.var_mean(x, dim=(1, 2, 3), unbiased=False, keepdim=True)
        std = (var + 1e-6).sqrt()
        return (x - mean) / std * self.weight.view(1, -1, 1, 1), std


class ResidualBlock(nn.Module):
    def __init__(self, gates, bottleneck):
        super().__init__()
        self.gates = gates
        self.norm = IRLayerNorm()
        self.up = nn.Conv2d(128, gates * 9, 1, bias=False)
        self.down = nn.Conv2d(gates * 8, bottleneck, 1, bias=False)
        self.mix = nn.Conv2d(
            bottleneck, 128, 3, padding=1, padding_mode="reflect", bias=False
        )
        self.act = nn.Mish()

    def forward(self, x):
        y, std = self.norm(x)
        values, gates = self.up(y).split([self.gates * 8, self.gates], dim=1)
        b, _, h, w = values.shape
        values = self.act(values).view(b, self.gates, 8, h, w)
        y = (values * (gates * 0.5).tanh()[:, :, None]).reshape(b, -1, h, w)
        return x + self.mix(self.down(y)) * std


class UpsampleHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.ps_expand = nn.Conv2d(128, 512, 1)
        self.lift_expand = nn.Conv2d(128, 256, 1, bias=False)
        self.lift_mix = nn.Conv2d(
            64, 128, 3, padding=1, padding_mode="reflect", bias=False
        )
        self.refine = ResidualBlock(32, 128)
        self.register_buffer("filters", torch.empty(2, 2, 128, 4))
        self.register_buffer("gains", torch.empty(2))

    def forward(self, x):
        x = F.pixel_shuffle(self.ps_expand(x), 2) + self.lift_mix(
            F.pixel_shuffle(self.lift_expand(x), 2)
        )
        x = self.refine(x)
        result = torch.zeros_like(x)
        # At exactly 2x, SesquiLSR's coordinate-conditioned filters are constant.
        for filters, gain in zip(self.filters, self.gains):
            y = F.pad(x, (1, 2, 0, 0), mode="replicate").unfold(3, 4, 1)
            y = (y * filters[1][None, :, None, None, :]).sum(-1)
            y = F.pad(y, (0, 0, 1, 2), mode="replicate").unfold(2, 4, 1)
            y = (y * filters[0][None, :, None, None, :]).sum(-1)
            result = result + gain * y
        return result


class LatentUpscaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.head_conv = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, padding_mode="reflect", bias=False),
            nn.Mish(),
            nn.Conv2d(64, 128, 3, padding=1, padding_mode="reflect"),
        )
        self.in_blocks = nn.Sequential(*(ResidualBlock(24, 96) for _ in range(8)))
        self.upsample = UpsampleHead()
        self.out_blocks = nn.Sequential(*(ResidualBlock(32, 128) for _ in range(6)))
        self.tail_conv = nn.Sequential(
            nn.Conv2d(128, 64, 1, bias=False),
            nn.Mish(),
            nn.Conv2d(64, 64, 3, padding=1, padding_mode="reflect"),
        )
        self.skip_conv = nn.Conv2d(
            64, 64, 5, padding=2, padding_mode="reflect", bias=False
        )

    def forward(self, x):
        h = self.out_blocks(self.upsample(self.in_blocks(self.head_conv(x))))
        skip = F.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False)
        return self.tail_conv(h) + self.skip_conv(skip)


class Qwen21Upscaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.empty(1, 64, 1, 1))
        self.register_buffer("std", torch.empty(1, 64, 1, 1))
        self.net = LatentUpscaler()

    def forward(self, raw):
        return self.net((raw - self.mean) / self.std) * self.std + self.mean
