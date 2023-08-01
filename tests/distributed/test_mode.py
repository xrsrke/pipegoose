from pipegoose.distributed.parallel_mode import ParallelMode


def test_parallel_mode():
    assert hasattr(ParallelMode, "GLOBAL")
    assert hasattr(ParallelMode, "TENSOR")
    assert hasattr(ParallelMode, "PIPELINE")
    assert hasattr(ParallelMode, "DATA")

    assert ParallelMode.GLOBAL == ParallelMode.GLOBAL
    assert ParallelMode.GLOBAL != ParallelMode.TENSOR
