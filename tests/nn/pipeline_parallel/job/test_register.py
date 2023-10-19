from queue import Queue

from pipegoose.nn.pipeline_parallel._job.job import Job
from pipegoose.nn.pipeline_parallel._job.register import add_job_to_queue


class Dummyjob(Job):
    def run_compute(self):
        pass


def test_register_job():
    input = 1

    def function(input):
        return input

    job = Dummyjob(function, input)
    JOB_QUEUE = Queue()

    add_job_to_queue(job, JOB_QUEUE)

    assert JOB_QUEUE.qsize() == 1
