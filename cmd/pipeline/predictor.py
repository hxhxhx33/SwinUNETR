import os
from typing import List, Dict, Any, cast
from dataclasses import dataclass

import torch
from torch import nn

import nibabel as nib  # type: ignore
from monai.inferers.inferer import SlidingWindowInferer

from base.predictor import Predictor as BasePredictor
from dataset.loader import IMAGE_KEY


class Predictor(BasePredictor):
    """Class for running a model."""

    @dataclass(kw_only=True)
    class Options(BasePredictor.Options):
        """Options for Predictor."""

        sw_roi_sizes: List[int]
        sw_overlap_ratio: float
        sw_batch_size: int = 1

    def __init__(self, opt: Options) -> None:
        super().__init__(opt)

        self.options = opt
        self.inferer = SlidingWindowInferer(
            roi_size=opt.sw_roi_sizes,
            sw_batch_size=opt.sw_batch_size,
            overlap=opt.sw_overlap_ratio,
        )

    def forward(self, inferer: nn.Module, batch: Dict[str, Any]) -> None:
        opt = self.options

        image = batch[IMAGE_KEY]
        image = image.to(opt.device)

        probs = self.inferer(inputs=image, network=inferer)

        probs = cast(torch.Tensor, probs)

        pred = convert_from_multi_channel(probs)
        pred = pred.detach().cpu().numpy()
        metadata = batch[f"{IMAGE_KEY}_meta_dict"]
        for bidx, fpath in enumerate(metadata["filename_or_obj"]):
            case_name = os.path.basename(os.path.dirname(fpath))
            save_dir = os.path.join(opt.output_dir, case_name)
            save_path = os.path.join(save_dir, f"{case_name}_pred.nii.gz")
            os.makedirs(save_dir, exist_ok=True)

            affine = metadata["affine"][bidx]
            nifti = nib.Nifti1Image(pred[bidx], affine)  # type: ignore
            nib.save(nifti, save_path)  # type: ignore


def convert_from_multi_channel(probs: torch.Tensor) -> torch.Tensor:
    """Convert from the multi-channel binary label tensor used for training back to
        the single-channel multi-value label tensor used for visualisation.

    Args:
        x (torch.Tensor): A tensor of shape (B, C, *D) where (_, c, *p) is the
            probability of the voxel `p` being of (converted) label `c`.

    Returns:
        torch.Tensor: A uint8 tensor of shape (B, *D) where (_, *p) is the Brats label
            of the voxel `p`.
    """
    seg = (probs > 0.5).to(torch.uint8)

    [batch_size, _, *dims] = seg.shape
    seg_out = torch.zeros([batch_size, *dims])

    # The order is important!
    seg_out[seg[:, 1] == 1] = 2
    seg_out[seg[:, 0] == 1] = 1
    seg_out[seg[:, 2] == 1] = 4

    seg_out = seg_out.to(torch.uint8)

    return seg_out
