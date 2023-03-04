import os
import logging
from dataclasses import dataclass
from typing import Callable, Dict

import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader

from base.util import setup_logging

from .argument import PredictArgument
from .predictor import Predictor

ModelFactory = Callable[[torch.device], nn.Module]
DataLoaderFactory = Callable[[torch.device], DataLoader[Dict[str, torch.Tensor]]]
PredictorFactory = Callable[[Predictor.Options], Predictor]


class PredictApp:
    """predict a model."""

    @dataclass(kw_only=True)
    class Options:
        """Options for Predictor."""

        model_factory: ModelFactory
        data_loader_factory: DataLoaderFactory
        predictor_factory: PredictorFactory

    def __init__(self, args: PredictArgument, opt: Options) -> None:
        self.options = opt
        self.args = args

        # logging
        setup_logging(level=args.log_level)

        # check GPU
        ngpu = torch.cuda.device_count()
        assert ngpu > 0, "GPU is required to run this script"
        logging.info("Found %d GPUs", ngpu)
        self.ngpu = ngpu

    def start(self) -> None:
        """Start predicting."""

        # Do not utilize multiple GPUs.
        _start_process(0, self)


def _start_process(rank: int, app: PredictApp) -> None:
    args = app.args
    opt = app.options

    # device
    device = torch.device(f"cuda:{rank}")
    logging.info("Start predicting on device %s", device)

    # create output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # model
    model = opt.model_factory(device)

    # data loader
    data_loader = opt.data_loader_factory(device)

    # predictor
    predictor_opt = Predictor.Options(
        model=model,
        data_loader=data_loader,
        device=device,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
    )
    predictor = opt.predictor_factory(predictor_opt)
    predictor.start()
