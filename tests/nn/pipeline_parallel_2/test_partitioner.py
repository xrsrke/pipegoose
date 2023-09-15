import pytest
from transformers import AutoModelForCausalLM

from pipegoose.nn.pipeline_parallel2.partitioner import NaivePartitioner
from pipegoose.testing.utils import init_parallel_context, spawn


def run_naive_partitioning(rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size, module):
    parallel_context = init_parallel_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )
    MODEL_NAME = "bigscience/bloom-560m"
    module = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    partitioner = NaivePartitioner(module, parallel_context)
    partitions = partitioner.split()

    assert len(partitions) == pipeline_parallel_size


@pytest.mark.parametrize("pipeline_parallel_size", [1, 2, 5])
def test_naive_partitioning(pipeline_parallel_size):
    TENSOR_PARALLEL_SIZE = 1
    DATA_PARALLEL_SIZE = 1

    spawn(
        run_naive_partitioning,
        world_size=pipeline_parallel_size,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=DATA_PARALLEL_SIZE,
    )
