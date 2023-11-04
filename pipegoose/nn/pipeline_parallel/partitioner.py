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


class UniformPartitioner(BasePartitioner):
    def __init__(self, module: nn.Module, parallel_context: ParallelContext):
        self.module = module
        self.parallel_context = parallel_context

    def split(self) -> List[nn.Module]:
        n_partitions = self.parallel_context.pipeline_parallel_size
        module = self.module
        partitions = []
        start = 0
        end = 0

        def _flatten_model(model, parent_name=""):
            model_list = []
            for name, child_module in model.named_children():
                # Form the full name of the module
                full_name = f"{parent_name}.{name}" if parent_name else name
                if (
                    full_name == "transformer.h"
                ):  # Check if the module is the 'h' attribute
                    # If it's the 'h' ModuleList, append each of its blocks as a whole
                    for block in child_module:
                        model_list.append(block)
                elif len(list(child_module.children())) == 0:
                    # If it's a leaf node, append the module itself
                    model_list.append(child_module)
                else:
                    # Otherwise, continue flattening its children
                    model_list.extend(_flatten_model(child_module, full_name))
            return model_list

        prepared_model = _flatten_model(module)
        for p in range(n_partitions):
            end = start + len(prepared_model) // n_partitions
            partitions.append(nn.Sequential(*prepared_model[start:end]))
            start = end

        for partition in partitions:
            print("--------------------------------------------------")
            print(partition)
            print("--------------------------------------------------")

        return partitions


def _get_partitioner(policy: PartitionPolicy) -> BasePartitioner:
    """Return the corresponding partitioner based on the policy."""
    policy_to_partitioner = {
        PartitionPolicy.UNIFORM: UniformPartitioner,
    }

    return policy_to_partitioner[policy]


def get_model_partition(
    module: nn.Module, policy: PartitionPolicy, parallel_context: ParallelContext
) -> nn.Module:
    """Get the corresponding partition of the current process."""
    partitioner = _get_partitioner(policy)
    partitions = partitioner(module, parallel_context).split()

    # TODO: remove this, use pipeline_context instead
    def _get_partition_idx():
        rank = parallel_context.get_local_rank(ParallelMode.PIPELINE)
        rank_per_group = len(parallel_context.get_ranks_in_group(ParallelMode.PIPELINE))
        partition_idx = rank // rank_per_group
        return partition_idx

    partition_idx = _get_partition_idx()
    return partitions[partition_idx]
