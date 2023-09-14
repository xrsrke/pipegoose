from abc import ABC, abstractclassmethod
from typing import List

from torch import nn

from pipegoose.distributed.parallel_context import ParallelContext


class BasePartitioner(ABC):
    @abstractclassmethod
    def split(self):
        raise NotImplementedError


class NaivePartitioner(BasePartitioner):
    def __init__(self, module: nn.Module, parallel_context: ParallelContext):
        self.module = module
        self.parallel_context = parallel_context

    def split(self) -> List[nn.Module]:
        module = self.module
        n_partitions = self.parallel_context.pipeline_parallel_size

        embedding_module = module.transformer.word_embeddings
        transformer_blocks = module.transformer.h
        lm_head = module.lm_head

        # NOTE: Calculate the number of transformer blocks per partition
        blocks_per_partition = len(transformer_blocks) // n_partitions
        partitions = []

        for i in range(n_partitions):
            start = i * blocks_per_partition
            # NOTE: if it's the last partition, get all remaining blocks
            end = start + blocks_per_partition if i < n_partitions - 1 else None
            partitions.append(nn.Sequential(*transformer_blocks[start:end]))

        partitions[0] = nn.Sequential(embedding_module, partitions[0])
        partitions[-1] = nn.Sequential(lm_head, partitions[-1])

        return partitions
