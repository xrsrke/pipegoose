from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.distributed.parallel_mode import ParallelMode


def get_num_local_experts(num_experts: int, parallel_context: ParallelContext) -> int:
    """Return the number of local experts per device."""
    tensor_parallel_size = parallel_context.get_world_size(ParallelMode.TENSOR)
    return num_experts // tensor_parallel_size
