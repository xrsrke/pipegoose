import pytest

from torch import nn
from transformers import AutoModelForCausalLM

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.nn.pipeline_parallel2.pipeline_context import PipelineContext
from pipegoose.nn.pipeline_parallel2.scheduler import GPipeScheduler
from pipegoose.nn.pipeline_parallel2.partitioner import NaivePartitioner
from pipegoose.testing.utils import spawn

MODEL_NAME = "bigscience/bloom-560m"


@pytest.fixture(scope="session")
def module():
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


@pytest.fixture
def scheduler():
    N_MICROBATCHES = 4
    N_PARTITIONS = 3

    scheduler = GPipeScheduler(N_MICROBATCHES, N_PARTITIONS)

    return scheduler


def run_pipeline_context(rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size, module, scheduler):
    parallel_context = init_parallel_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )
    partitioner = NaivePartitioner(module, parallel_context)
    partitions = partitioner.split()

    pipeline_context = PipelineContext(partitions, scheduler, parallel_context)

    assert isinstance(pipeline_context.partition_idx, int)
    assert isinstance(pipeline_context.get_partition_forward(), nn.Module)

    assert isinstance(pipeline_context.current_clock_idx, int)
    assert isinstance(pipeline_context.current_schedule, list)
    assert isinstance(pipeline_context.schedules, list)


@pytest.mark.parametrize("pipeline_parallel_size", [1, 2])
def test_run_pipeline_context(module, scheduler, pipeline_parallel_size):
    TENSOR_PARALLEL_SIZE = 1
    DATA_PARALLEL_SIZE = 1

    spawn(
        run_pipeline_context,
        world_size=pipeline_parallel_size,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=DATA_PARALLEL_SIZE,
        module=module,
        scheduler=scheduler,
    )
