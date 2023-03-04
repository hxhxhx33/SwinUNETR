import torch
from torch import nn

from monai.losses.dice import DiceLoss

from base.trainer import Trainer as BaseTrainer, TensorDict
from dataset.loader import IMAGE_KEY, LABEL_KEY


class Trainer(BaseTrainer):
    """Class for training a model."""

    def __init__(self, opt: BaseTrainer.Options) -> None:
        super().__init__(opt)
        self.loss_func = DiceLoss()

    def forward_loss(self, inferer: nn.Module, batch: TensorDict) -> torch.Tensor:
        opt = self.options

        image = batch[IMAGE_KEY]
        label = batch[LABEL_KEY]

        image = image.to(opt.device)
        label = label.to(opt.device)

        probs = inferer(image)
        loss = self.loss_func(probs, label)
        return loss
