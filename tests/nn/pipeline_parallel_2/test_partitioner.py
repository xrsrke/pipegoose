import pytest
from transformers import AutoModelForCausalLM

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.nn.pipeline_parallel2.partitioner import NaivePartitioner
from pipegoose.testing.utils import spawn

MODEL_NAME = "bigscience/bloom-560m"


@pytest.fixture(scope="session")
def model():
    return AutoModelForCausalLM.from_pretrained(MODEL_NAME)


def init_parallel_context(rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size):
    parallel_context = ParallelContext(
        rank=rank,
        local_rank=rank,
        world_size=world_size,
        local_world_size=world_size,
        host="localhost",
        port=port,
        seed=69,
        backend="gloo",
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=data_parallel_size,
    )

    return parallel_context


def run_naive_partitioning(rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size, module):
    parallel_context = init_parallel_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )

    partitioner = NaivePartitioner(module, parallel_context)
    partitions = partitioner.split()

    assert len(partitions) == pipeline_parallel_size


@pytest.mark.parametrize("pipeline_parallel_size", [1, 2, 5])
def test_naive_partitioning(model, pipeline_parallel_size):
    TENSOR_PARALLEL_SIZE = 1
    DATA_PARALLEL_SIZE = 1

    spawn(
        run_naive_partitioning,
        world_size=pipeline_parallel_size,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=DATA_PARALLEL_SIZE,
        module=model,
    )
