from abc import ABC, abstractclassmethod
from dataclasses import dataclass
from typing import Union

import torch
from torch import nn

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.distributed.parallel_mode import ParallelMode
from pipegoose.nn.tensor_parallel._utils import VocabUtility
from pipegoose.nn.tensor_parallel.embedding import ParallelEmbedding
from pipegoose.nn.tensor_parallel.layer_norm import LayerNorm
from pipegoose.nn.tensor_parallel.linear import ColumnParallelLinear, RowParallelLinear
from pipegoose.nn.tensor_parallel.parallel_mapping import TensorParallelMapping


def _update_model_arguments(module: nn.Module, **kwargs):
    for key, value in kwargs.items():
        setattr(module, key, value)


def get_partition(data: torch.Tensor, parallel_context: ParallelContext, dim: int) -> torch.Tensor:
    rank = parallel_context.get_local_rank(ParallelMode.TENSOR)
    chunks = torch.chunk(data, parallel_context.get_world_size(ParallelMode.TENSOR), dim=dim)
    return chunks[rank].contiguous()


@dataclass
class ParallelMetadata:
    is_sliced: bool


class ModuleParallelizer(ABC):
    def __init__(self, module_name: str, module: nn.Module, model: nn.Module, parallel_context: ParallelContext):
        """Parallelize a module.

        Args:
            module_name (str): the name of the module
            module (nn.Module): the module to be parallelized. example: nn.Linear, nn.Embedding, nn.LayerNorm
            model (nn.Module): the transformer model that contains the module
            parallel_context (ParallelContext): parallel context
        """
        self.module_name = module_name
        self.module = module
        self.model = model
        self.parallel_context = parallel_context

    @abstractclassmethod
    def is_parallelizable(self):
        raise NotImplementedError

    @abstractclassmethod
    def parallelize(self):
        raise NotImplementedError

    @abstractclassmethod
    def deparallelize(self):
        raise NotImplementedError


class LinearParallelizer(ModuleParallelizer):
    @staticmethod
    def is_parallelizable(module_name: str, module: nn.Module) -> bool:
        return isinstance(module, nn.Linear) and TensorParallelMapping.is_lm_head(module_name) is False

    def parallelize(self) -> Union[ColumnParallelLinear, RowParallelLinear]:
        assert self.is_parallelizable(self.module_name, self.module), f"{self.module_name} can't be parallelized"

        if TensorParallelMapping.is_column_parallel(self.module_name):
            module = self._parallelize_column_linear(self.module)
        elif TensorParallelMapping.is_row_parallel(self.module_name):
            module = self._parallelize_row_linear(self.module)
        else:
            raise ValueError(f"module {self.module_name} is not supported")

        return module

    def deparallelize(self):
        pass

    def _parallelize_column_linear(self, module: nn.Module) -> ColumnParallelLinear:
        module.__class__ = ColumnParallelLinear
        module = self._slice_weight_and_bias(module, slice_bias=True, dim=0)

        _update_model_arguments(
            module=module,
            gather_output=True,
            parallel_context=self.parallel_context,
        )

        return module

    def _parallelize_row_linear(self, module: nn.Module) -> RowParallelLinear:
        module.__class__ = RowParallelLinear
        module = self._slice_weight_and_bias(module, slice_bias=False, dim=1)

        _update_model_arguments(
            module=module,
            parallel_context=self.parallel_context,
        )

        return module

    def _slice_weight_and_bias(self, module: nn.Module, slice_bias: bool, dim: int) -> nn.Module:
        module.weight.data = get_partition(module.weight, self.parallel_context, dim=dim)
        module.weight.parallel_metadata = ParallelMetadata(is_sliced=True)

        if module.bias is not None and slice_bias is True:
            module.bias.data = get_partition(module.bias, self.parallel_context, dim=0)

        return module


class EmbeddingParallelizer(ModuleParallelizer):
    @staticmethod
    def is_parallelizable(module_name: str, module: nn.Module) -> bool:
        return isinstance(module, nn.Embedding)

    def parallelize(self) -> ParallelEmbedding:
        """Parallelize nn.Embedding module."""
        assert self.is_parallelizable(self.module_name, self.module), f"{self.module_name} can't be parallelized"

        module = self.module
        self._resize_vocab_size(module)
        self._split_weight(module)

        return module

    def deparallelize(self):
        pass

    def _split_weight(self, module: nn.Module):
        """Split weight into chunks and assign to each process."""
        world_size = self.parallel_context.get_world_size(ParallelMode.TENSOR)
        rank = self.parallel_context.get_local_rank(ParallelMode.TENSOR)

        vocab_size = module.weight.shape[0]
        vocab_start_idx, vocab_end_idx = VocabUtility.get_vocab_range_from_global_vocab_size(world_size, rank, vocab_size)
        weight_chunks = torch.chunk(module.weight, world_size, dim=0)

        module.weight.data = weight_chunks[rank]
        module.weight.parallel_metadata = ParallelMetadata(is_sliced=True)
        module.__class__ = ParallelEmbedding

        _update_model_arguments(
            module=module,
            parallel_context=self.parallel_context,
            vocab_start_idx=vocab_start_idx,
            vocab_end_idx=vocab_end_idx,
            world_size=world_size,
        )

    def _resize_vocab_size(self, module: nn.Module):
        """Pad embedding size to make it splittable across GPUs"""
        padding_size = 0

        vocab_size, embedding_dim = module.weight.size()
        while not self._is_splitable(vocab_size + padding_size):
            padding_size += 1

        if padding_size > 0:
            padding = torch.zeros((padding_size, embedding_dim), device=module.weight.device)
            new_embeddings = torch.cat([module.weight, padding], dim=0)

            module.weight.data = new_embeddings

    def _is_splitable(self, size: int) -> bool:
        world_size = self.parallel_context.get_world_size(ParallelMode.TENSOR)
        return True if size % world_size == 0 else False


class LayerNormParallelizer(ModuleParallelizer):
    @staticmethod
    def is_parallelizable(module_name: str, module: nn.Module) -> bool:
        return isinstance(module, nn.LayerNorm)

    def parallelize(self) -> LayerNorm:
        assert self.is_parallelizable(self.module_name, self.module), f"{self.module_name} can't be parallelized"
        self.module.__class__ = LayerNorm

        _update_model_arguments(
            module=self.module,
            normalized_shape=self.module.normalized_shape,
            eps=self.module.eps,
            paralell_context=self.parallel_context,
        )

        return self.module

    def deparallelize(self):
        pass


class LMHeadParallelizer(ModuleParallelizer):
    @staticmethod
    def is_parallelizable(module_name: str, module: nn.Module) -> bool:
        return isinstance(module, nn.Linear) and TensorParallelMapping.is_lm_head(module_name) is True

    """Parallelize language model head."""

    def parallelize(self) -> ColumnParallelLinear:
        assert self.is_parallelizable(self.module_name, self.module), f"{self.module_name} can't be parallelized"
        module = self.module
        module.__class__ = ColumnParallelLinear

        # NOTE: in some models, the lm_head uses the same weight as the token embedding.
        # Because we split the token embedding before the lm_head, so if we already split
        # the token embedding, then we want to avoid splitting the weight again
        if module.weight is self.model.get_input_embeddings().weight:
            if not hasattr(module.weight, "parallel_metadata"):
                self._slice_weight(module, dim=0)
        else:
            self._slice_weight(module, dim=0)

        _update_model_arguments(
            module=module,
            gather_output=True,
            parallel_context=self.parallel_context,
        )

        return module

    def _slice_weight(self, module: nn.Module, dim: int) -> ColumnParallelLinear:
        module.weight.data = get_partition(module.weight, self.parallel_context, dim=dim)
        module.weight.parallel_metadata = ParallelMetadata(is_sliced=True)
        return module

    def deparallelize(self):
        pass
