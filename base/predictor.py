import abc
import logging
import time
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader

from .trainer import CKPT_MODEL_KEY, TensorDict


class Predictor:
    """Class for running model predictions."""

    @dataclass(kw_only=True)
    class Options:
        """Options for Predictor."""

        model: nn.Module
        data_loader: DataLoader[TensorDict]
        device: torch.device
        checkpoint: str
        output_dir: str

    def __init__(self, opt: Options) -> None:
        self.options = opt
        self._load_checkpoint()

    @abc.abstractmethod
    def forward(self, inferer: nn.Module, batch: TensorDict) -> None:
        """Forward and compute loss.

        Args:
            inferer (nn.Module): The inferer used to perform forward computaiton.
            batch (TensorDict): The input batch.
        """
        raise NotImplementedError

    def start(self):
        """Start predicting."""

        opt = self.options

        logging.info("Start inference")
        opt.model.eval()

        with torch.no_grad():
            for bidx, batch in enumerate(opt.data_loader, 1):
                start_time = time.time()

                self.forward(opt.model, batch)
                logging.info(
                    "Finished batch %d/%d with duration %.2fs",
                    bidx,
                    len(opt.data_loader),
                    time.time() - start_time,
                )

    def _load_checkpoint(self) -> None:
        opt = self.options
        ckpt = torch.load(opt.checkpoint, map_location=opt.device)  # type: ignore
        opt.model.load_state_dict(ckpt[CKPT_MODEL_KEY])
        opt.model.to(opt.device)  # type: ignore
