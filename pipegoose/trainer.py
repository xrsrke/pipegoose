from transformers import Trainer as BaseTrainer
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from transformers.modeling_utils import PreTrainedModel
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch import nn
from pipegoose.nn import DataParallel, TensorParallel
from pipegoose.distributed import ParallelContext

class Trainer(BaseTrainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        **kwargs
    ):
        # TODO: make dataloader work with accelerate for distributed sampler to work
        parallel_context = ParallelContext.from_torch(
            data_parallel_size=2,
            tensor_parallel_size=2,
            pipeline_parallel_size=2
        )

        model = TensorParallel(model, parallel_context).parallelize()
        model = DataParallel(model, parallel_context).parallelize()

        super().__init__(model=model, train_dataset=train_dataset, eval_dataset=eval_dataset, optimizers=optimizers, **kwargs)