from abc import ABC, abstractclassmethod
from enum import Enum, auto
from typing import List

from torch import nn

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.distributed.parallel_mode import ParallelMode


class PartitionPolicy(Enum):
    UNIFORM = auto()


class BasePartitioner(ABC):
    """Base class for partitioning a model into multiple partitions."""

    @abstractclassmethod
    def split(self) -> List[nn.Module]:
        raise NotImplementedError


class _UniformPartitioner(BasePartitioner):
    def __init__(self, module: nn.Module, parallel_context: ParallelContext):
        self.module = module
        self.parallel_context = parallel_context

    def split(self) -> List[nn.Module]:
        module = self.module
        n_partitions = self.parallel_context.pipeline_parallel_size

        # BLOOM-560
        # embedding_module = module.transformer.word_embeddings
        # transformer_blocks = module.transformer.h
        # lm_head = module.lm_head

        # For sshleifer/tiny-gpt2
        embed_module = module.transformer.wte
        pos_embed_module = module.transformer.wpe
        drop_module = module.transformer.drop
        transformer_blocks = module.transformer.h
        ln_f = module.transformer.ln_f
        lm_head = module.lm_head

        # NOTE: Calculate the number of transformer blocks per partition
        blocks_per_partition = len(transformer_blocks) // n_partitions
        partitions = []

        for i in range(n_partitions):
            start = i * blocks_per_partition
            # NOTE: if it's the last partition, get all remaining blocks
            end = start + blocks_per_partition if i < n_partitions - 1 else None
            partitions.append(nn.Sequential(*transformer_blocks[start:end]))

        partitions[0] = nn.Sequential(embed_module, pos_embed_module, drop_module, partitions[0])
        partitions[-1] = nn.Sequential(ln_f, lm_head, partitions[-1])

        return partitions


def _get_partitioner(policy: PartitionPolicy) -> BasePartitioner:
    """Return the corresponding partitioner based on the policy."""
    policy_to_partitioner = {
        PartitionPolicy.UNIFORM: _UniformPartitioner,
    }

    return policy_to_partitioner[policy]


def get_model_partition(module: nn.Module, policy: PartitionPolicy, parallel_context: ParallelContext) -> nn.Module:
    """Get the corresponding partition of the current process."""
    partitioner = _get_partitioner(policy)
    partitions = partitioner(module, parallel_context).split()

    def _get_partition_idx():
        rank = parallel_context.get_local_rank(ParallelMode.PIPELINE)
        rank_per_group = len(parallel_context.get_ranks_in_group(ParallelMode.PIPELINE))
        partition_idx = rank // rank_per_group
        return partition_idx

    partition_idx = _get_partition_idx()
    return partitions[partition_idx]
