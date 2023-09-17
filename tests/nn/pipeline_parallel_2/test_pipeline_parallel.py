import pytest
import torch

from pipegoose.nn.pipeline_parallel2.partitioner import PartitionPolicy
from pipegoose.nn.pipeline_parallel2.pipeline_parallel import PipelineParallel
from pipegoose.nn.pipeline_parallel2.scheduler import SchedulerType


class FakeParallelContext:
    pass


@pytest.mark.skip
def test_pipeline_parallel(model):
    parallel_context = FakeParallelContext()

    NUM_MICROBATCHES = 5

    input = torch.randn(NUM_MICROBATCHES, 4)

    parallelized_model = PipelineParallel(
        module=model,
        num_microbatches=NUM_MICROBATCHES,
        scheduler_type=SchedulerType.GPIPE,
        partition_policy=PartitionPolicy.UNIFORM,
        parallel_context=parallel_context,
    ).parallelize()

    output = parallelized_model(input)

    assert isinstance(output, torch.Tensor)
