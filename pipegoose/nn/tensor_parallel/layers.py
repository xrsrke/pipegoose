from torch import nn

from pipegoose.distributed import ParallelContext


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
