from typing import List

from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.trainer.callback import Callback
from pipegoose.trainer.logger import DistributedLogger
from pipegoose.trainer.state import TrainerState


class Trainer:
    def __init__(
        self,
        module: nn.Module,
        train_loader: DataLoader,
        eval_loader: DataLoader,
        optim: Optimizer,
        num_epochs: int,
        callbacks: List[Callback] = [],
        loggers: List[DistributedLogger] = [],
        parallel_context: ParallelContext = None,
    ):
        # NOTE: based on the data_parallel_size, tensor_parallel_size, and pipeline_parallel_size
        # in the parallel_context, we do the correspond parallel model.
        self.state = TrainerState()

    def fit(self):
        # NOTE: both train and validation
        pass

    def train(self):
        # NOTE: only train
        pass
