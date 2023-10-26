from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union, Type
import torch
from transformers.modeling_utils import PreTrainedModel
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch import nn
from pipegoose.nn import DataParallel, TensorParallel
from pipegoose.distributed import ParallelContext, ParallelMode
from transformers.utils import is_datasets_available
from torch.utils.data import DataLoader, Dataset
from transformers.data.data_collator import DataCollator
from transformers.trainer_utils import seed_worker
from torch.utils.data.distributed import DistributedSampler
from transformers.training_args  import TrainingArguments
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
if is_datasets_available():
    import datasets
def parallelize_model(model, parallel_context):
    if parallel_context.tensor_parallel_size > 1:
        model = TensorParallel(model, parallel_context).parallelize()
    if parallel_context.data_parallel_size > 1:
        model = DataParallel(model, parallel_context).parallelize()
    if parallel_context.pipeline_parallel_size > 1:
        raise NotImplemented("Pipeline parallel is not yet implemented")
    return model
def pipegoose_prepare_dataloader(dataset, parallel_context, dataloader_params):
    sampler = None
    if parallel_context.data_parallel_size > 1:
        dp_rank = parallel_context.get_local_rank(ParallelMode.DATA)
        sampler =  DistributedSampler(dataset, rank=dp_rank)
    dataloader_params["sampler"] = sampler
    return DataLoader(dataset, **dataloader_params)
class Trainer:
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        optimizer_cls: Type[torch.optim.Optimizer] = None,
        data_parallel_size: int = 2,
        tensor_parallel_size: int = 2,
        pipeline_parallel_size: int = 1,
        **kwargs
    ):
        self.parallel_context = ParallelContext.from_torch(
            data_parallel_size=data_parallel_size,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size
        )
        self.model = parallelize_model(model, self.parallel_context)
        self.model.to("cuda")
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.args = args
        self.data_collator = data_collator
        self.train_dataset = train_dataset
        self.optimizer = optimizer_cls(model.parameters(), lr=self.args.learning_rate)
    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
        return pipegoose_prepare_dataloader(train_dataset, parallel_context=self.parallel_context, dataloader_params=dataloader_params)
    def train(
        self,
        **kwargs,
    ):
        self._train_batch_size = self.args.train_batch_size
        train_dataloader = self.get_train_dataloader()
        for epoch in range(int(self.args.num_train_epochs)):
            for batch in train_dataloader:
                inputs = self.tokenizer(batch["text"], padding=True, truncation=True, max_length=1024, return_tensors="pt")
                for key in inputs:
                    inputs[key] = inputs[key].to(self.device)
                labels = inputs["input_ids"]
                outputs = self.model(**inputs, labels=labels)
                self.optimizer.zero_grad()
                outputs.loss.backward()
                self.optimizer.step()