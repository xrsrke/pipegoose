import os

import torch

from pipegoose.distributed._initializers.initialize_tensor import (
    TensorParallelGroupInitializer,
)

if __name__ == "__main__":
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    host = os.environ["MASTER_ADDR"]
    port = int(os.environ["MASTER_PORT"])

    TENSOR_PARALLEL_SIZE = 2
    PIPELINE_PARALLEL_SIZE = 2
    DATA_PARALLEL_SIZE = 2

    torch.distributed.init_process_group(
        rank=rank,
        world_size=world_size,
        backend="gloo",
    )

    result = TensorParallelGroupInitializer(
        rank,
        world_size,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
        data_parallel_size=DATA_PARALLEL_SIZE,
    ).init_dist_group()

    # assert isinstance(result["local_rank"], int)
    # assert isinstance(result["local_world_size"], int)
    # assert isinstance(result["process_group"], ProcessGroup)
    # assert isinstance(result["ranks_in_group"], list)
    # assert isinstance(result["parallel_mode"], ParallelMode.TENSOR)
