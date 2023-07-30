from abc import ABC, abstractclassmethod

from torch import nn

from pipegoose.distributed.context import ParallelContext
from pipegoose.distributed.mode import ParallelMode


class ColumnParallelLinear(nn.Module):
    pass


class RowParallelLinear(nn.Module):
    pass


class ParallelMLP(nn.Module):
    def __init__(self, parallel_context: ParallelContext):
        super().__init__()
        # world_size = parallel_context.get_world_size()
        self.dense_h_to_4h = ColumnParallelLinear()
        self.dense_4h_to_h = RowParallelLinear()
        self.activation_func = nn.GELU()


class ParallelTransformer(nn.Module):
    pass


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
        local_rank = self.parallel_context.get_local_rank(ParallelMode.TENSOR)
        local_world_size = self.parallel_context.get_local_world_size(ParallelMode.TENSOR)

        input_size, output_size = module.weight.size()
        output_per_partition = output_size // local_world_size

        start_element = local_rank * output_per_partition
        end_element = (local_rank + 1) * output_per_partition
        weight = module.weight.data[:, start_element:end_element]

        if module.bias is not None:
            bias = module.bias.data[start_element:end_element]

        # if is_linear_parallelizable(module) is True:
        #     pass

    def deparallelize(self):
        pass


class ParallelizeEmbedding(ParallelizeModule):
    def parallelize(self):
        vocab_size, embedding_size = self.module.weight.size()


class ParallelizeLayerNorm(ParallelizeModule):
    pass


class ParallelizeAttention(ParallelizeModule):
    pass
