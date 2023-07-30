from pipegoose.distributed._initializers.initialize_tensor import (
    TensorParallelGroupInitializer,
)


def test_init_tensor_parallel_group():
    TensorParallelGroupInitializer()
