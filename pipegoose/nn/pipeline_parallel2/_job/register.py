from queue import Queue

from pipegoose.distributed.parallel_context import ParallelContext

from pipegoose.nn.pipeline_parallel2._job.job import Job


class JobRegister:
    def __init__(self, queue: Queue, parallel_context: ParallelContext):
        self.queue = queue
        self.parallel_context = parallel_context

    def registry(self, job: Job):
        assert isinstance(job, Job), f"job must be an instance of Job, got {type(job)}"
        self.queue.put(job)
