from queue import Queue

from pipegoose.nn.pipeline_parallel._job.job import Job


class _JobRegister:
    def __init__(self, queue: Queue):
        self.queue = queue

    def registry(self, job: Job):
        assert isinstance(job, Job), f"job must be an instance of Job, got {type(job)}"
        self.queue.put(job)


def add_job_to_queue(job: Job, queue: Queue):
    job_register = _JobRegister(queue)
    job_register.registry(job)
