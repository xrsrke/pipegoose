import pytest


class FakeParallelContext:
    pass


@pytest.mark.skip
def test_pipeline_engine(model):
    # BATCH_SIZE = 32
    # parallel_context = FakeParallelContext()
    # torch.randn(BATCH_SIZE, 4)

    # pipeline_engine = PipelineEngine(
    #     module=model,
    #     scheduler=GPipeScheduler(),
    #     worker_manager=WorkerManager(),
    #     parallel_context=parallel_context,
    # )
    pass
