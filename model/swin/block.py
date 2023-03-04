from typing import List
from dataclasses import dataclass
import itertools

import torch
import torch.nn as nn
from einops import rearrange  # type: ignore
from monai.networks.blocks.mlp import MLPBlock

from model.helper.padding import even_pad, unpad

from .window_attention import WindowAttention


class Block(nn.Module):
    """The Block module."""

    @dataclass(kw_only=True)
    class Options:
        """Options to build the model."""

        window_attention_options: WindowAttention.Options
        window_shift: int
        mlp_ratio: int

    def __init__(self, opt: Options):
        super().__init__()

        self.options = opt
        wopt = opt.window_attention_options

        self.window_attention = WindowAttention(wopt)

        self.norm1 = nn.LayerNorm(wopt.num_channel)
        self.norm2 = nn.LayerNorm(wopt.num_channel)
        self.mlp = MLPBlock(
            hidden_size=wopt.num_channel,
            mlp_dim=wopt.num_channel * opt.mlp_ratio,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        """Forward.

        Args:
            x (torch.Tensor): A tensor of shape (B, C, *D).

        Returns:
            torch.Tensor: A tensor of the same shape.
        """
        x = x + self._forward_1(x)
        x = x + self._forward_2(x)
        return x

    def _forward_1(self, x: torch.Tensor) -> torch.Tensor:
        opt = self.options
        wopt = opt.window_attention_options

        window_sizes = [wopt.window_size] * wopt.num_dim
        window_shifts = [opt.window_shift] * wopt.num_dim

        # pad to make window partition possible
        x, pad = even_pad(x, window_sizes)
        [_, _, *spatial_dims] = x.size()

        # shift
        if opt.window_shift > 0:
            shifted_x = torch.roll(
                x,
                shifts=[-s for s in window_shifts],
                dims=list(range(2, 2 + len(window_shifts))),
            )
            mask = window_mask(
                spatial_dims=spatial_dims,
                window_sizes=window_sizes,
                window_shifts=window_shifts,
                device=x.device,
            )
        else:
            shifted_x = x
            mask = None

        shifted_x = shifted_x.movedim(1, -1)
        shifted_x = self.norm1(shifted_x)
        shifted_x = shifted_x.movedim(-1, 1)

        windows = window_partition(shifted_x, window_sizes)
        windows = self.window_attention(windows, mask=mask)
        shifted_x = window_reunion(windows, window_sizes, spatial_dims)

        # unshift
        if opt.window_shift > 0:
            x = torch.roll(
                shifted_x,
                shifts=window_shifts,
                dims=list(range(2, 2 + wopt.num_dim)),
            )
        else:
            x = shifted_x
            mask = None

        # unpad
        x = unpad(x, pad)

        return x

    def _forward_2(self, x: torch.Tensor) -> torch.Tensor:
        x = x.movedim(1, -1)
        x = self.norm2(x)
        x = self.mlp(x)
        x = x.movedim(-1, 1)
        return x


def window_partition(x: torch.Tensor, window_sizes: List[int]) -> torch.Tensor:
    ds = " ".join([f"d{i}" for i in range(len(window_sizes))])
    ws = " ".join([f"w{i}" for i in range(len(window_sizes))])
    dws = " ".join([f"(d{i} w{i})" for i in range(len(window_sizes))])
    vs = {f"w{i}": w for i, w in enumerate(window_sizes)}

    x = rearrange(x, f"b c {dws} -> (b {ds}) ({ws}) c", **vs)

    return x


def window_reunion(
    x: torch.Tensor,
    window_sizes: List[int],
    spatial_dims: List[int],
) -> torch.Tensor:
    assert len(spatial_dims) == len(window_sizes)
    for [d, s] in zip(spatial_dims, window_sizes):
        assert d % s == 0
    n = len(spatial_dims)

    ds = " ".join([f"d{i}" for i in range(n)])
    ws = " ".join([f"w{i}" for i in range(n)])
    dws = " ".join([f"(d{i} w{i})" for i in range(n)])
    dvs = {f"d{i}": d // w for i, [d, w] in enumerate(zip(spatial_dims, window_sizes))}
    wvs = {f"w{i}": w for i, w in enumerate(window_sizes)}

    x = rearrange(x, f"(b {ds}) ({ws}) c -> b c {dws}", **wvs, **dvs)

    return x


def window_mask(
    spatial_dims: List[int],
    window_sizes: List[int],
    window_shifts: List[int],
    device: torch.device,
) -> torch.Tensor:
    assert len(window_sizes) == len(window_shifts)
    assert len(spatial_dims) == len(window_sizes)
    for [d, s] in zip(spatial_dims, window_sizes):
        assert d % s == 0

    ranges = [
        (
            slice(-size),
            slice(-size, -shift),
            slice(-shift, None),
        )
        for size, shift in zip(window_sizes, window_shifts)
    ]

    label = 0
    cluster = torch.zeros(*spatial_dims, device=device)
    for idx in itertools.product(*ranges):
        cluster[idx] = label
        label += 1

    ds = " ".join([f"d{i}" for i in range(len(window_sizes))])
    ws = " ".join([f"w{i}" for i in range(len(window_sizes))])
    dws = " ".join([f"(d{i} w{i})" for i in range(len(window_sizes))])
    vs = {f"w{i}": w for i, w in enumerate(window_sizes)}
    windows = rearrange(cluster, f"{dws} -> ({ds}) ({ws})", **vs)

    mask = windows[:, :, None] - windows[:, None, :]
    mask = mask.masked_fill(mask != 0, -999.0).masked_fill(mask == 0, 0.0)
    return mask
