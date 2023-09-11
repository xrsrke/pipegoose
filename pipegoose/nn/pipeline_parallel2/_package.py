from dataclasses import dataclass

from pipegoose.nn.pipeline_parallel2._job.job_type import JobType


import torch


@dataclass
class TrainingMetadata:
    is_training: bool
    is_grad_enabled: bool


@dataclass
class Metadata:
    # pipeline
    microbatch_idx: int
    partition_idx: int

    job_type: JobType

    training: TrainingMetadata

    # global rank
    src: int
    dst: int


class Package:
    """A data package that will be sent from one pipeline stage to another."""
    def __init__(self, data: torch.Tensor, metadata: Metadata):
        self.data = data
        self.metadata = metadata
