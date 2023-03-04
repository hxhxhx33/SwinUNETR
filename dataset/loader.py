import os
import sys
import logging
import math
from typing import List, Dict
from functools import partial

import torch
from torch.utils.data.distributed import DistributedSampler

from monai.data.dataset import Dataset
from monai.data.dataloader import DataLoader
from monai.transforms.transform import MapTransform
from monai.transforms.compose import Compose
from monai.transforms.croppad.dictionary import (
    CropForegroundd,
    RandSpatialCropd,
    SpatialPadd,
)
from monai.transforms.intensity.dictionary import NormalizeIntensityd
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.spatial.dictionary import RandRotated, RandFlipd
from monai.transforms.utility.dictionary import (
    ToTensord,
    ToDeviced,
    EnsureChannelFirstd,
    ConvertToMultiChannelBasedOnBratsClassesd,
)

from .option import Option, Options

IMAGE_KEY = "image"
LABEL_KEY = "label"
PREDICTION_KEY = "prediction"


def create_data_loader(
    data_root: str,
    fold_map: Dict[str, int],
    folds: List[int],
    *options: Option,
) -> DataLoader:
    """_summary_

    Args:
        data_root (str): the root directory from which to load data. The directory must
            be of the following structure:
                - [data_root]
                    - [sample]
                        - [sample]_flair.nii.gz
                        - [sample]_t1.nii.gz
                        - [sample]_t1ce.nii.gz
                        - [sample]_t2.nii.gz
                        - [sample]_seg.nii.gz (Optional)
                    - ...
        fold_map (Dict[str, int]): A map from [sample] to fold index.
        fold (int): The fold index to use.

    Returns:
        DataLoader: The data loader.
    """
    # Create a data loader for training.
    opt = Options()
    for option in options:
        option(opt)

    subdirs = _list_subdir(data_root)

    foldss = set(folds)
    sample_names = map(os.path.basename, subdirs)
    sample_names = [n for n in sample_names if n in fold_map and fold_map[n] in foldss]

    if (n := len(sample_names)) == 0:
        logging.critical("No samples are loaded")
        sys.exit(1)
    else:
        logging.info("Loaded %d samples", n)

    make_image_dict = partial(
        _make_image_dict,
        data_root=data_root,
        opt=opt,
    )
    samples = list(map(make_image_dict, sample_names))

    keys: List[str] = []
    if opt.image_included:
        keys.append(IMAGE_KEY)
    if opt.label_included:
        keys.append(LABEL_KEY)
    if opt.prediction_path_lookup is not None:
        keys.append(PREDICTION_KEY)

    # transformation
    trans: List[MapTransform] = [
        LoadImaged(keys),
        EnsureChannelFirstd(keys=keys),
    ]
    if opt.label_included:
        trans.append(ConvertToMultiChannelBasedOnBratsClassesd(keys=[LABEL_KEY]))
    if opt.prediction_path_lookup is not None:
        trans.append(ConvertToMultiChannelBasedOnBratsClassesd(keys=[PREDICTION_KEY]))

    if opt.device is not None:
        trans.append(ToDeviced(keys=keys, device=opt.device))

    if opt.augment:
        logging.info("Augment inputs")
        if opt.no_random_rotate:
            logging.info("No random rotatiton")
        else:
            trans.append(
                RandRotated(
                    keys=keys,
                    keep_size=False,
                    prob=1.0,
                    range_x=math.pi,
                    range_y=math.pi,
                    range_z=math.pi,
                    padding_mode="zeros",
                    # The default `bilinear` mode will introduce holes in labels.
                    mode="nearest",
                )
            )
        trans.extend(
            [
                RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
                RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
                RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
            ],
        )

    if opt.image_included and not opt.no_crop_foreground:
        trans.append(CropForegroundd(keys=keys, source_key=IMAGE_KEY))

    if opt.image_included:
        norm = NormalizeIntensityd(keys=[IMAGE_KEY], nonzero=True, channel_wise=True)
        trans.append(norm)

    if opt.roi_sizes is not None:
        logging.info("Crop inputs into %s", opt.roi_sizes)
        trans.extend(
            [
                SpatialPadd(
                    keys=keys,
                    spatial_size=opt.roi_sizes,
                ),
                RandSpatialCropd(
                    keys=keys,
                    roi_size=opt.roi_sizes,
                    random_size=False,
                    random_center=not opt.no_crop_random_center,
                ),
            ]
        )
    trans.extend(
        [
            ToTensord(keys=keys),
        ]
    )

    # dataset
    dataset = Dataset(samples, transform=Compose(transforms=trans))

    # sampler
    T = Dict[str, torch.Tensor]
    sampler = DistributedSampler[T](dataset, shuffle=True) if opt.distribtued else None

    # loader
    data_loader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        sampler=sampler,
        # can not set with `ToDevice`
        num_workers=1 if opt.device is None else 0,
        pin_memory=opt.device is None,
    )
    return data_loader


def _list_subdir(root: str) -> List[str]:
    rel_paths = os.listdir(root)
    abs_paths = map(lambda p: os.path.join(root, p), rel_paths)
    subdirs = filter(os.path.isdir, abs_paths)
    return list(subdirs)


def _make_image_dict(name: str, data_root: str, opt: Options):
    folder = os.path.join(data_root, name)
    sample = {
        f"{IMAGE_KEY}": [
            os.path.join(folder, f"{name}_flair.nii.gz"),
            os.path.join(folder, f"{name}_t1.nii.gz"),
            os.path.join(folder, f"{name}_t1ce.nii.gz"),
            os.path.join(folder, f"{name}_t2.nii.gz"),
        ],
    }
    if opt.label_included:
        sample[LABEL_KEY] = [
            os.path.join(folder, f"{name}_seg.nii.gz"),
        ]
    if opt.prediction_path_lookup is not None:
        path = opt.prediction_path_lookup(name)
        sample[PREDICTION_KEY] = [
            path,
        ]

    return sample
