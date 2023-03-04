import os
import random
import logging
import socket
from contextlib import closing

import numpy
import torch


def ensure_reproducibility(seed: int):
    """Provide a seed for reproducibility.

    Args:
        seed (Optional[int]): The seed to use.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)  # type: ignore
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore


def setup_logging(level: str, prefix: str = ""):
    """Setup the logging.

    Args:
        level (str): The log level.
        prefix (str, optional): A prefix append to the logging message. Defaults to "".
    """
    numeric_level = getattr(logging, level.upper(), None)
    assert isinstance(numeric_level, int)

    components = ["%(asctime)s.%(msecs)03d", "%(levelname)s"]
    if prefix != "":
        components.append(prefix)
    components.append("%(message)s")

    logging.basicConfig(
        level=numeric_level,
        datefmt="%Y-%m-%d %H:%M:%S",
        format=" ".join(components),
        # Importing from `monai` causes logging configurations here take no effect,
        # presumbly because it calls `logging.basicConfig` first.
        # To rescue, set `force` to True.
        force=True,
    )


def get_open_port():
    """Get an open port."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
