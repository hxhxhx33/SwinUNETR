import os
import socket
import logging
from dataclasses import dataclass
from typing import Dict, Callable, TypeVar, Generic

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.utils.data.dataloader import DataLoader


from .argument import TrainArgument
from .trainer import Trainer
from .util import ensure_reproducibility, setup_logging, get_open_port

T = TypeVar("T", bound=TrainArgument)
S = TypeVar("S", bound=TrainArgument)

ModelFactory = Callable[[torch.device, T], nn.Module]
DataLoaderFactory = Callable[[torch.device, T], DataLoader[Dict[str, torch.Tensor]]]
TrainerFactory = Callable[[Trainer.Options, T], Trainer]


class TrainApp(Generic[T]):
    """Train a model."""

    @dataclass(kw_only=True)
    class Options(Generic[S]):
        """Options for Trainer."""

        model_factory: ModelFactory[S]
        data_loader_factory: DataLoaderFactory[S]
        trainer_factory: TrainerFactory[S]

    def __init__(self, args: T, opt: Options[T]) -> None:
        self.options = opt
        self.args = args

        # set host and port for pytorch DDP
        torch_port = os.getenv("MASTER_PORT")
        if torch_port is None:
            torch_port = str(get_open_port())
            os.environ["MASTER_PORT"] = torch_port
        logging.info("Use port %s for PyTorch DDP", torch_port)

        torch_addr = os.getenv("MASTER_ADDR")
        if torch_addr is None:
            torch_addr = "127.0.0.1"
            os.environ["MASTER_ADDR"] = torch_addr
        host = socket.gethostbyname(socket.gethostname())
        logging.info("Use addr %s(%s) for PyTorch DDP", torch_addr, host)

        # check GPU
        ngpu = torch.cuda.device_count()
        assert ngpu > 0, "GPU is required to run this script"
        logging.info("Found %d GPUs", ngpu)
        self.ngpu = ngpu

        # log args
        args_dict = {k: v for k, v in args.as_dict().items() if v is not None}
        logging.info("arguments: %s", args_dict)

        # ddp
        node_size = 1
        if (env := os.getenv("NODE_SIZE")) is not None:
            node_size = int(env)
        node_rank = 0
        if (env := os.getenv("NODE_RANK")) is not None:
            node_rank = int(env)
        self.node_size = node_size
        self.node_rank = node_rank
        logging.info("Run on node %d/%d", node_rank + 1, node_size)

    def start(self) -> None:
        """Start training."""
        mp.spawn(_start_process, nprocs=self.ngpu, args=(self,))  # type: ignore


def start_train(args: T, opt: TrainApp.Options[T]):
    """Start training.

    Args:
        args (T): The command line arguments.
        opt (TrainApp.Options[T]): The TrainApp options.
    """

    app = TrainApp[T](args, opt)
    app.start()


def _start_process(local_rank: int, app: TrainApp[T]) -> None:
    args = app.args
    opt = app.options

    setup_logging(level=args.log_level, prefix=f"Process-{local_rank}")
    logging.info("Start training on GPU %d", local_rank)

    # seed
    ensure_reproducibility(args.seed)
    logging.info("Using seed %d", args.seed)

    # distribtued data parallel
    ngpu = torch.cuda.device_count()
    world_size = ngpu * app.node_size
    global_rank = ngpu * app.node_rank + local_rank
    dist.init_process_group(  # type: ignore
        backend="nccl",
        world_size=world_size,
        rank=global_rank,
    )

    # device
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # model
    model = opt.model_factory(device, args)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("Number of params: %.2fM", (n_params / 1.0e6))
    model.to(device)  # type: ignore

    # data loader
    data_loader = opt.data_loader_factory(device, args)

    # trainer
    trainer_opt = Trainer.Options(
        model=model,
        num_epoch=args.num_epoch,
        data_loader=data_loader,
        device=device,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        global_rank=global_rank,
        local_rank=local_rank,
        acc_batch=args.acc_batch,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_prefix=args.checkpoint_prefix,
        max_time_sec=args.max_time_sec,
        use_amp=not args.no_use_amp,
    )

    trainer = opt.trainer_factory(trainer_opt, args)
    trainer.start()
