from abc import ABC, abstractclassmethod

from torch import nn

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.distributed.parallel_mode import ParallelMode


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
        module = self.module
        parallel_context = self.parallel_context

        local_rank = parallel_context.get_local_rank(ParallelMode.TENSOR)
        local_world_size = parallel_context.get_local_world_size(ParallelMode.TENSOR)

        input_size, output_size = module.weight.size()
        output_per_partition = output_size // local_world_size

        start_element = local_rank * output_per_partition
        end_element = (local_rank + 1) * output_per_partition

        weight = module.weight.data
        module.weight.data = weight[:, start_element:end_element]

        if module.bias is not None:
            bias = module.bias.data
            module.bias.data = bias[start_element:end_element]

        return module

    def deparallelize(self):
        pass


class ParallelizeEmbedding(ParallelizeModule):
    def parallelize(self):
        vocab_size, embedding_size = self.module.weight.size()


class ParallelizeLayerNorm(ParallelizeModule):
    pass
