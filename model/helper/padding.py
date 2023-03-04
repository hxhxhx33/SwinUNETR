from typing import List, Tuple
import math

import torch
import torch.nn.functional as F


def even_pad(
    x: torch.Tensor,
    units: List[int],
) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
    """Pad a tensor to make its spatial dimensions divisible by given units. Zero
    paddings will be added evenly at head and tail.

    Args:
        x (torch.Tensor): The tensor to pad. Must have the shape (B, C, *) with spatial
            dimension at the tail.
        units (List[int]): The unit along each spatial dimension. Must have length equal
            to the length of spatial dimensions.

    Returns:
        Tuple[torch.Tensor, List[Tuple[int, int]]]:
            0: The padded tensor.
            1: A list of tuple telling the size of padding at the head and tail along
                each dimension.
    """
    [_, _, *dims] = x.shape
    assert len(dims) == len(units)

    pad: List[Tuple[int, int]] = []
    for d, u in zip(reversed(dims), reversed(units)):
        r = (u - d % u) % u
        l_pad = math.floor(r / 2)
        r_pad = math.ceil(r / 2)
        pad.append((l_pad, r_pad))
    padded = F.pad(x, pad=[v for p in pad for v in p])
    pad.reverse()

    return padded, pad


def unpad(x: torch.Tensor, pads: List[Tuple[int, int]]) -> torch.Tensor:
    """Unpad.

    Args:
        x (torch.Tensor): The tensor to pad. Must have the shape
            (B, C, *) with spatial dimension at the tail.
        pad (List[Tuple[int, int]]): A list of tuple telling the size of padding at the
            head and tail along each dimension.
    Returns:
        torch.Tensor: The unpadded tensor.
    """
    [_, _, *dims] = x.size()
    assert len(dims) == len(pads)

    rs = [slice(None), slice(None)]
    for d, (l, r) in zip(dims, pads):
        rs.append(slice(l, d - r))

    return x[rs]


def unpad_to_align(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Unpad the source to match the shape of the destination.

    Args:
        src (torch.Tensor): A tensor of shape (B, C, *) with spatial dimensions at the
            tail each of which is not less than that of `dst`.
        dst (torch.Tensor): A tensor of shape (B, C, *) with spatial dimensions at the
            tail.

    Returns:
        torch.Tensor: The unpadded tensor from the source.
    """
    [_, _, *src_dims] = src.size()
    [_, _, *dst_dims] = dst.size()
    assert len(src_dims) == len(dst_dims)

    rs = [slice(None), slice(None)]
    for [s, d] in zip(src_dims, dst_dims):
        assert s >= d
        r = s - d
        l_pad = math.floor(r / 2)
        r_pad = math.ceil(r / 2)
        rs.append(slice(l_pad, s - r_pad))

    return src[rs]
