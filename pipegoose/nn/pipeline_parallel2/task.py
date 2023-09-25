from dataclasses import dataclass

from pipegoose.nn.pipeline_parallel2._job.job_type import JobType


@dataclass
class Task:
    job_type: JobType
    microbatch_idx: int
    partition_idx: int
