import pytest
import torch

from pipegoose.nn.pipeline_parallel2.pipeline_parallel import PipelineParallel


class FakeParallelContext:
    pass


@pytest.mark.skip
def test_pipeline_parallel(model):
    parallel_context = FakeParallelContext()

    NUM_MICROBATCHES = 5
    NUM_COCURRENT = 10
    MAX_COCURRENT = 20

    input = torch.randn(NUM_MICROBATCHES, 4)

    parallelized_model = PipelineParallel(
        module=model,
        num_microbatches=NUM_MICROBATCHES,
        num_cocurrents=NUM_COCURRENT,
        max_cocurrent=MAX_COCURRENT,
        parallel_context=parallel_context,
    ).parallelize()

    output = parallelized_model(input)

    assert isinstance(output, torch.Tensor)
