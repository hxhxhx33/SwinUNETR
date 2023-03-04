from typing import List, Callable, Optional
from dataclasses import dataclass

import torch


@dataclass(kw_only=True)
class Options:
    """Option for data loader."""

    batch_size: int = 1
    roi_sizes: Optional[List[int]] = None
    image_included: bool = True
    label_included: bool = False
    distribtued: bool = False
    augment: bool = False
    device: Optional[torch.device] = None
    no_random_rotate: bool = False
    no_crop_random_center: bool = False
    no_crop_foreground: bool = False
    prediction_path_lookup: Optional[Callable[[str], str]] = None


Option = Callable[[Options], None]


def with_batch_size(size: int) -> Option:
    """Set batch size.

    Args:
        size (int): The batch size.

    Returns:
        Option: A dataset option.
    """

    def _(opt: Options):
        opt.batch_size = size

    return _


def with_roi_sizes(sizes: List[int]) -> Option:
    """Set RoI sizes.

    Args:
        sizes (List[int]): The RoI size along each dimension.

    Returns:
        Option: A dataset option.
    """

    def _(opt: Options):
        opt.roi_sizes = sizes

    return _


def with_distribtued(distribtued: bool) -> Option:
    """Set if is in distributed model.

    Args:
        distribtued (bool): To load data for distribution or not.

    Returns:
        Option: A dataset option.
    """

    def _(opt: Options):
        opt.distribtued = distribtued

    return _


def with_image_included(include: bool) -> Option:
    """Set if to include image.

    Args:
        include (bool): To include image or not.

    Returns:
        Option: A dataset option.
    """

    def _(opt: Options):
        opt.image_included = include

    return _


def with_label_included(include: bool) -> Option:
    """Set if to include label.

    Args:
        include (bool): To include label or not.

    Returns:
        Option: A dataset option.
    """

    def _(opt: Options):
        opt.label_included = include

    return _


def with_augment(augment: bool) -> Option:
    """Set if to perform data augmentation.

    Args:
        augment (bool): To perform data augmentation or not.

    Returns:
        Option: A dataset option.
    """

    def _(opt: Options):
        opt.augment = augment

    return _


def with_device(device: torch.device) -> Option:
    """Set a device to, say, run preprocessing like augmentation.

    Args:
        device (torch.device): The device to load data onto.

    Returns:
        Option: A dataset option.
    """

    def _(opt: Options):
        opt.device = device

    return _


def with_no_random_rotate(no_rotate: bool) -> Option:
    """Set if not to randomly rotate the data in augmentation.

    Args:
        no_rotate (bool): If not to randomly rotate.

    Returns:
        Option: A dataset option.
    """

    def _(opt: Options):
        opt.no_random_rotate = no_rotate

    return _


def with_no_crop_random_center(no_random: bool) -> Option:
    """Set if not to randomly pick center when cropping.

    Args:
        no_random (bool): If not to randomly pick center.

    Returns:
        Option: A dataset option.
    """

    def _(opt: Options):
        opt.no_crop_random_center = no_random

    return _


def with_no_crop_foreground(no_crop: bool) -> Option:
    """Set if not to crop foreground.

    Args:
        no_random (bool): If not to crop foreground.

    Returns:
        Option: A dataset option.
    """

    def _(opt: Options):
        opt.no_crop_foreground = no_crop

    return _


def with_prediction_path_lookup(lookup: Callable[[str], str]) -> Option:
    """Set a prediction path lookup function to load predictions.

    Args:
        lookup (Callable[[str], str]): A path-lookup function.

    Returns:
        Option: A dataset option.
    """

    def _(opt: Options):
        opt.prediction_path_lookup = lookup

    return _
