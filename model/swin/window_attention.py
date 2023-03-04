from typing import List, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
from einops import rearrange  # type: ignore
from monai.networks.layers.weight_init import trunc_normal_  # type: ignore


class WindowAttention(nn.Module):
    """The WindowAttention module."""

    @dataclass(kw_only=True)
    class Options:
        """Options to build the model."""

        num_dim: int
        window_size: int
        num_channel: int
        num_head: int

    def __init__(self, opt: Options):
        super().__init__()

        self.options = opt
        assert opt.num_channel % opt.num_head == 0

        self.scale = (opt.num_channel // opt.num_head) ** -0.5
        self.qkv = nn.Linear(
            in_features=opt.num_channel,
            out_features=opt.num_channel * 3,
            bias=False,
        )
        self.softmax = nn.Softmax(dim=-1)

        # relative position embedding
        self.relative_position_index = relative_position_index(
            dims=[opt.window_size] * opt.num_dim
        )
        self.relative_position_bias = nn.Parameter(
            torch.zeros(
                opt.num_head,
                (2 * opt.window_size - 1) ** opt.num_dim,
            )
        )
        trunc_normal_(self.relative_position_bias, std=0.02)

    def forward(  # type: ignore
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Forward.

        Args:
            x (torch.Tensor): A tensor of shape (V, C).
            mask (Optional[torch.Tensor]): N x (V, V) tensor with mask(i, j) = 0 when
                position i and j are relevant, and -inf otherwise.

        Returns:
            torch.Tensor: A tensor of shape (V, C).
        """
        opt = self.options

        qkv = self.qkv(x)
        qkv = rearrange(qkv, "n v (k h c) -> k n h v c", h=opt.num_head, k=3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q *= self.scale

        p = q @ k.transpose(-2, -1)
        p += self.relative_position_bias[:, self.relative_position_index]
        if mask is not None:
            p += mask.unsqueeze(1)

        a = self.softmax(p)
        x = a @ v
        x = rearrange(x, "n h v c -> n v (h c)", h=opt.num_head)
        return x


def relative_position_index(dims: List[int]) -> torch.Tensor:
    """Calculate relation position index.

    Args:
        dims (List[int]): The dimensionality.

    Returns:
        torch.Tensor: A (V, V) matrix in which entry (i, j) gives an integer uniquely
            identifies the relative position between voxel i and j. (i1, j1) and
            (i2, j2) have the same value if and only if the voxels they represent have
            the same spatial relative position.
    """
    coords = [torch.arange(d) for d in dims]

    # a (D, *dims) tensor where coords[k][d_1][...][d_D] = d_k.
    coords = torch.stack(torch.meshgrid(*coords, indexing="ij"))

    # a (D, V) tensor
    coords = torch.flatten(coords, 1)

    # a (D, V, V) tensor where relative_coords[k, i, i] tells the k-th coordinate
    # difference between position i and j
    relative_coords = coords[:, :, None] - coords[:, None, :]

    # a (V, V, D) tensor
    relative_coords = relative_coords.permute(1, 2, 0)

    # calculate the embedding
    for i, d in enumerate(dims):
        relative_coords[:, :, i] += d - 1

    s = 1
    for i, d in enumerate(reversed(dims), 1):
        relative_coords[:, :, -i] *= s
        s *= 2 * d - 1

    index = relative_coords.sum(-1)
    return index
