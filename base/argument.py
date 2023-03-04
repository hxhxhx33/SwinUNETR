import os
import random
from typing import Optional, List, Any, Dict

from tap import Tap


class BaseArgument(Tap):
    """The class supposed to be the base of all argument class."""

    def __init__(
        self,
        *args: List[Any],
        underscores_to_dashes: bool = False,
        explicit_bool: bool = False,
        config_files: Optional[List[str]] = None,
        **kwargs: Dict[str, Any],
    ):
        config_files = config_files or []
        if (env := os.getenv("ARGS_FILES")) is not None:
            extras = env.split(" ")
            config_files.extend(extras)

        super().__init__(  # type: ignore
            *args,
            underscores_to_dashes=underscores_to_dashes,
            explicit_bool=explicit_bool,
            config_files=config_files,
            **kwargs,
        )


class CommonArgument(BaseArgument):
    """Common commend line arguments."""

    log_level: str = "INFO"
    """log level"""

    data_root: str
    """dataset root directory"""

    fold_map: str
    """path to the fold map json file"""

    folds: List[int]
    """fold index to use"""

    seed: int = random.randrange(2**32)
    """a seed for random generation for reproducibility"""


class DatasetArgument(CommonArgument):
    """Commend line arguments for data set configurations."""

    batch_size: int = 1
    """batch size to train (effective batch size is batch_size * acc_batch * #gpus)"""


class TrainArgument(CommonArgument):
    """Commend line arguments for training."""

    num_epoch: int
    """number of epochs to train"""

    max_time_sec: Optional[int] = None
    """max time in seconds to to train"""

    acc_batch: int = 1
    """number of batches to accumulate gradients as a scale of effective batch size"""

    lr: float = 1e-4
    """learning rate"""

    weight_decay: float = 1e-5
    """weight decay"""

    checkpoint_dir: str
    """directory to save checkpoints"""

    checkpoint_prefix: str = "model"
    """prefix to save checkpoints"""

    no_use_amp: bool = False
    """do not use amp"""


class PredictArgument(CommonArgument):
    """Commend line arguments for prediction."""

    checkpoint: str
    """path to the checkpoint to be loaded"""

    output_dir: str
    """directory to save output"""
