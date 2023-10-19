import torch
from torch import nn

from pipegoose.nn.pipeline_parallel._job.backward import BackwardJob
from pipegoose.nn.pipeline_parallel._job.forward import (
    CreateForwardOutputPackageCallback,
    ForwardJob,
    SaveActivationIfTrainingCallback,
    SaveInputActivationsCallback,
)
from pipegoose.nn.pipeline_parallel._job.job_type import JobType
from pipegoose.nn.pipeline_parallel._package import Metadata, Package, TrainingMetadata
from pipegoose.testing.utils import init_pipeline_context

TENSOR_PARALLEL_SIZE = 1
PIPELINE_PARALLEL_SIZE = 1
DATA_PARALLEL_SIZE = 1
RANK = 0
WORLD_SIZE = 1
PORT = 12355

N_PARTITIONS = 3
N_MICROBATCHES = 5

pipeline_context, parallel_context = init_pipeline_context(
    rank=RANK,
    world_size=WORLD_SIZE,
    port=PORT,
    tensor_parallel_size=TENSOR_PARALLEL_SIZE,
    pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
    data_parallel_size=DATA_PARALLEL_SIZE,
    n_partitions=N_PARTITIONS,
    n_microbatches=N_MICROBATCHES,
)


BATCH_SIZE = 10
HIDDEN_SIZE = 20
OUTPUT_SIZE = 5


MICROBATCH_IDX = 0
PARTITION_IDX = 0
IS_TRAINING = True
IS_GRAD_ENABLED = True

SRC = 0
DST = 1

input = torch.randn((BATCH_SIZE, HIDDEN_SIZE), requires_grad=IS_GRAD_ENABLED)
linear = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)

metadata = Metadata(
    microbatch_idx=MICROBATCH_IDX,
    partition_idx=PARTITION_IDX,
    job_type=JobType.FORWARD,
    training=TrainingMetadata(
        is_training=IS_TRAINING,
        is_grad_enabled=IS_GRAD_ENABLED,
    ),
    src=SRC,
    dst=DST,
)

input_package = Package(input, metadata)

forward_callbacks = [
    CreateForwardOutputPackageCallback(parallel_context, pipeline_context),
    SaveInputActivationsCallback(),
    SaveActivationIfTrainingCallback(),
]
forward_job = ForwardJob(linear, input_package, forward_callbacks)
output = forward_job.compute()

INITIAL_GRADS = torch.ones_like(output.data)
grad_package = Package(INITIAL_GRADS, metadata)
grad_package.metadata.job_type = JobType.BACKWARD


class BackwardFunction:
    pass


backward_job = BackwardJob(BackwardFunction, grad_package, cbs=[])
backward_job.compute()

assert 1 == 1
