from dataclasses import dataclass

import torch
import torch.nn as nn
from einops import rearrange  # type: ignore

from model.helper.padding import even_pad

from .block import Block
from .window_attention import WindowAttention


class Stage(nn.Module):
    """The Stage module."""

    @dataclass(kw_only=True)
    class Options:
        """Options to build the model."""

        window_attention_options: WindowAttention.Options
        depth: int
        block_window_shift: int
        block_mlp_ratio: int

    def __init__(self, opt: Options):
        super().__init__()

        self.options = opt
        wopt = opt.window_attention_options

        self.blocks = nn.ModuleList(
            [
                Block(
                    Block.Options(
                        window_attention_options=wopt,
                        window_shift=0 if i % 2 == 0 else opt.block_window_shift,
                        mlp_ratio=opt.block_mlp_ratio,
                    )
                )
                for i in range(opt.depth)
            ]
        )
        self.merge = PatchMerging(num_channel=wopt.num_channel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        for block in self.blocks:
            x = block(x)
        x = self.merge(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, num_channel: int) -> None:
        super().__init__()

        self.proj = nn.Linear(
            in_features=8 * num_channel,
            out_features=2 * num_channel,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        x, _ = even_pad(x, [2] * (len(x.shape) - 2))

        # TODO(hxu): generic

        # reshape since `nn.Linaer` acts on the last dimension.
        x = rearrange(x, "n c d h w -> n d h w c")

        x000 = x[:, 0::2, 0::2, 0::2, :]
        x001 = x[:, 0::2, 0::2, 1::2, :]
        x010 = x[:, 0::2, 1::2, 0::2, :]
        x100 = x[:, 1::2, 0::2, 0::2, :]
        x011 = x[:, 0::2, 1::2, 1::2, :]
        x101 = x[:, 1::2, 0::2, 1::2, :]
        x110 = x[:, 1::2, 1::2, 0::2, :]
        x111 = x[:, 1::2, 1::2, 1::2, :]
        x = torch.cat([x000, x001, x010, x100, x110, x011, x101, x111], -1)

        x = self.proj(x)
        x = rearrange(x, "n d h w c -> n c d h w")

        return x
