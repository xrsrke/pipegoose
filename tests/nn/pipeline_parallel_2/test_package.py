import pytest

import torch

from pipegoose.nn.pipeline_parallel2._package import Package, Metadata, TrainingMetadata
from pipegoose.nn.pipeline_parallel2._job.job_type import JobType


@pytest.mark.parametrize("job_type", [JobType.FORWARD, JobType.BACKWARD])
def test_package(job_type):
    MICROBATCH_IDX = 1
    PARTITION_IDX = 2
    IS_TRAINING = True
    IS_GRAD_ENABLED = False

    SRC = 0
    DST = 1

    data = torch.randn(2, 4)
    metadata = Metadata(
        microbatch_idx=MICROBATCH_IDX,
        partition_idx=PARTITION_IDX,
        job_type=job_type,
        training=TrainingMetadata(
            is_training=IS_TRAINING,
            is_grad_enabled=IS_GRAD_ENABLED,
        ),
        src=SRC,
        dst=DST,
    )

    package = Package(data, metadata)

    assert package.metadata.microbatch_idx == MICROBATCH_IDX
    assert package.metadata.partition_idx == PARTITION_IDX

    assert package.metadata.job_type == job_type
    assert package.metadata.training.is_training == IS_TRAINING
    assert package.metadata.training.is_grad_enabled == IS_GRAD_ENABLED

    assert package.metadata.src == SRC
    assert package.metadata.dst == DST

    assert torch.allclose(package.data, data)
