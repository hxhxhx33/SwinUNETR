import json

import torch

from base.train import TrainApp, start_train
from base.util import setup_logging

from dataset import option as dataset_option
from dataset.loader import create_data_loader

from .argument import TrainArgument
from .common import create_model
from .pipeline.trainer import Trainer


def _model_factory(_: torch.device, args: TrainArgument):
    model = create_model(args)
    return model


def _data_loader_factory(device: torch.device, args: TrainArgument):
    with open(args.fold_map, encoding="utf-8") as file:
        fold_map = json.load(file)
    return create_data_loader(
        args.data_root,
        fold_map,
        args.folds,
        dataset_option.with_roi_sizes([args.spatial_size] * args.num_dim),
        dataset_option.with_device(device),
        dataset_option.with_distribtued(True),
        dataset_option.with_label_included(True),
        dataset_option.with_batch_size(args.batch_size),
        dataset_option.with_no_random_rotate(args.no_random_rotate),
        dataset_option.with_no_crop_random_center(args.no_crop_random_center),
        dataset_option.with_augment(not args.no_augment),
    )


def _trainer_factory(opt: Trainer.Options, _: TrainArgument):
    return Trainer(opt)


if __name__ == "__main__":
    xargs = TrainArgument().parse_args()
    setup_logging(level=xargs.log_level)
    train_opt = TrainApp.Options(
        model_factory=_model_factory,
        data_loader_factory=_data_loader_factory,
        trainer_factory=_trainer_factory,
    )
    start_train(xargs, train_opt)
