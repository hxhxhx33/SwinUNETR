from typing import List
from dataclasses import dataclass

import torch
import torch.nn as nn
from monai.networks.layers.factories import Conv
from monai.networks.layers.utils import get_norm_layer  # type: ignore

from model.helper.padding import even_pad

from .stage import Stage
from .window_attention import WindowAttention


class PatchEmbedding(nn.Module):
    """The PatchEmbedding module."""

    @dataclass(kw_only=True)
    class Options:
        """Options to build the model."""

        num_dim: int
        patch_size: int
        input_channel: int
        output_channel: int

    def __init__(self, opt: Options):
        super().__init__()

        self.options = opt

        # pylint: disable=not-callable
        self.proj = Conv["conv", opt.num_dim](
            in_channels=opt.input_channel,
            out_channels=opt.output_channel,
            kernel_size=opt.patch_size,
            stride=opt.patch_size,
        )

        # normalise along the output channel dimension,
        self.norm = get_norm_layer(
            name=("layer", {"normalized_shape": [opt.output_channel]})
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        opt = self.options

        x, _ = even_pad(x, [opt.patch_size] * opt.num_dim)
        x = self.proj(x)

        x = x.movedim(1, -1)
        x = self.norm(x)
        x = x.movedim(-1, 1)

        return x


class Swin(nn.Module):
    """The Swin module."""

    @dataclass(kw_only=True)
    class Options:
        """Options to build the model."""

        embed_options: PatchEmbedding.Options
        window_size: int
        stage_depths: List[int]
        stage_num_heads: List[int]
        block_mlp_ratio: int

    def __init__(self, opt: Options):
        super().__init__()

        self.options = opt
        eopt = opt.embed_options

        assert len(opt.stage_depths) == len(opt.stage_num_heads)

        self.embed = PatchEmbedding(eopt)
        self.stages = nn.ModuleList(
            [
                Stage(
                    Stage.Options(
                        window_attention_options=WindowAttention.Options(
                            num_dim=eopt.num_dim,
                            window_size=opt.window_size,
                            num_channel=eopt.output_channel * (2**i),
                            num_head=num_head,
                        ),
                        depth=depth,
                        block_window_shift=opt.window_size // 2,
                        block_mlp_ratio=opt.block_mlp_ratio,
                    )
                )
                for i, [depth, num_head] in enumerate(
                    zip(opt.stage_depths, opt.stage_num_heads)
                )
            ]
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:  # type: ignore
        outs: List[torch.Tensor] = []

        x = self.embed(x)
        outs.append(x)

        for stage in self.stages:
            x = stage(x)
            outs.append(x)

        return outs
