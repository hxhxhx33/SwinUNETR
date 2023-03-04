import json
from dataclasses import asdict

import torch

from base.predict import PredictApp
from base.predict import Predictor as BasePredictor

from dataset import option as dataset_option
from dataset.loader import create_data_loader

from .argument import PredictArgument
from .common import create_model
from .pipeline.predictor import Predictor

args = PredictArgument().parse_args()
with open(args.fold_map, encoding="utf-8") as file:
    fold_map = json.load(file)


def _model_factory(_: torch.device):
    return create_model(args)


def _data_loader_factory(_device: torch.device):
    return create_data_loader(
        args.data_root,
        fold_map,
        args.folds,
        dataset_option.with_batch_size(1),
        dataset_option.with_label_included(True),
        dataset_option.with_no_crop_foreground(True),
    )


def _predictor_factory(options: BasePredictor.Options):
    opt = Predictor.Options(
        sw_roi_sizes=[args.spatial_size] * args.num_dim,
        sw_batch_size=args.sw_batch_size,
        sw_overlap_ratio=args.sw_overlap_ratio,
        **asdict(options),
    )
    return Predictor(opt)


if __name__ == "__main__":
    app = PredictApp(
        args,
        PredictApp.Options(
            model_factory=_model_factory,
            data_loader_factory=_data_loader_factory,
            predictor_factory=_predictor_factory,
        ),
    )

    app.start()
