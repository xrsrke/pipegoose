from abc import ABC, abstractclassmethod

import torch
from torch import nn

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.distributed.parallel_mode import ParallelMode
from pipegoose.nn.tensor_parallel._utils import VocabUtility, is_splitable
from pipegoose.nn.tensor_parallel.embedding import ParallelEmbedding


def _update_model_arguments(module: nn.Module, **kwargs):
    for key, value in kwargs.items():
        setattr(module, key, value)


class ParallelizeModule(ABC):
    def __init__(self, module: nn.Module, parallel_context: ParallelContext):
        self.module = module
        self.parallel_context = parallel_context

    @abstractclassmethod
    def parallelize(self):
        raise NotImplementedError

    @abstractclassmethod
    def deparallelize(self):
        raise NotImplementedError


class ParallelizeLinear(ParallelizeModule):
    def parallelize(self):
        pass

    def deparallelize(self):
        pass


class ParallelizeEmbedding(ParallelizeModule):
    # TODO: refactor to staticmethod
    def parallelize(self) -> nn.Module:
        assert isinstance(self.module, nn.Embedding), "only parallelize nn.Embedding"
        self._resize_vocab_size()
        self._split_weight()
        return self.module

    def deparallelize(self):
        pass

    def _split_weight(self):
        world_size = self.parallel_context.get_world_size(ParallelMode.TENSOR)
        rank = self.parallel_context.get_local_rank(ParallelMode.TENSOR)

        vocab_size = self.module.weight.shape[0]
        vocab_start_idx, vocab_end_idx = VocabUtility.get_vocab_range_from_global_vocab_size(world_size, rank, vocab_size)
        weight_chunks = torch.chunk(self.module.weight, world_size, dim=0)
        self.module.weight.data = weight_chunks[rank]

        self.module.__class__ = ParallelEmbedding

        _update_model_arguments(module=self.module, vocab_start_idx=vocab_start_idx, vocab_end_idx=vocab_end_idx)

    def _resize_vocab_size(self):
        """Make vocab size divisible by world size."""
        padding_size = 0

        vocab_size, embedding_dim = self.module.weight.size()
        while not is_splitable(vocab_size + padding_size, self.parallel_context):
            padding_size += 1

        if padding_size > 0:
            padding = torch.zeros((padding_size, embedding_dim))
            new_embeddings = torch.cat([self.module.weight, padding], dim=0)

            self.module.weight.data = new_embeddings

    # def _is_text_embedding(self, module: nn.Module) -> bool:
    #     return True if module is self.module.get_input_embeddings() else False


class ParallelizeLayerNorm(ParallelizeModule):
    pass


class ParallelizeAttention(ParallelizeModule):
    pass
