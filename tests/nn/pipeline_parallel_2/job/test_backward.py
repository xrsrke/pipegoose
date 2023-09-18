from copy import deepcopy

import pytest
import torch
from torch import nn

from pipegoose.nn.pipeline_parallel2._job.backward import BackwardJob
from pipegoose.nn.pipeline_parallel2._job.creator import schedule_backward_job
from pipegoose.nn.pipeline_parallel2._utils import sleep
from pipegoose.nn.pipeline_parallel2.pipeline_context import PipelineContext
from pipegoose.nn.pipeline_parallel2.queue import JobQueue
from pipegoose.nn.pipeline_parallel2.scheduler import SchedulerType, get_scheduler
from pipegoose.testing.utils import init_parallel_context, spawn

# NOTE: use for creating a forward job
function = nn.Linear(2, 4)


def run_create_a_backward_job_if_a_tensor_do_backprop(
    rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size, forward_package
):
    SRC = forward_package.metadata.src
    N_PARTITIONS = 3
    N_MICROBATCHES = 5
    parallel_context = init_parallel_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )

    scheduler = get_scheduler(SchedulerType.GPIPE)(N_MICROBATCHES, N_PARTITIONS)
    pipeline_context = PipelineContext(scheduler, parallel_context)
    rank = parallel_context.get_global_rank()

    # NOTE: both the forward job and backward job of the same package
    # execute on the same node
    if rank == SRC:
        ORIG_FORWARD_PACKAGE = deepcopy(forward_package)
        forward_package = schedule_backward_job(forward_package, pipeline_context)

        # NOTE: do forward job and save activations

        # NOTE: make sure we aren't change the package
        assert torch.equal(forward_package.data, ORIG_FORWARD_PACKAGE.data)
        assert forward_package.metadata == ORIG_FORWARD_PACKAGE.metadata

        data = forward_package.data
        data.sum().backward()

        sleep(2)

        # NOTE: since we don't launch any job selector workers in the background,
        # after triggering the creation of a backward job,
        # we expect the destination worker's job queue to have one job
        assert JobQueue.PENDING_JOBS.qsize() == 1

        backward_job = JobQueue.PENDING_JOBS.get()

        assert isinstance(backward_job, BackwardJob)


@pytest.mark.parametrize("pipeline_parallel_size", [2, 5])
def test_create_a_backward_job_if_a_tensor_do_backprop(forward_package, pipeline_parallel_size):
    TENSOR_PARALLEL_SIZE = 1
    DATA_PARALLEL_SIZE = 1

    spawn(
        run_create_a_backward_job_if_a_tensor_do_backprop,
        world_size=pipeline_parallel_size,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=DATA_PARALLEL_SIZE,
        forward_package=forward_package,
    )


@pytest.mark.skip
def test_execute_a_backward_job_and_send_the_output():
    pass


def test_execute_a_backward_job():
    pass
