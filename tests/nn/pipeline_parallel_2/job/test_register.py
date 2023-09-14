from queue import Queue

import pytest

from pipegoose.nn.pipeline_parallel2._job.register import add_job_to_queue


@pytest.mark.parametrize("job", ["forward_job", "backward_job"])
def test_register_job(request, job):
    JOB_QUEUE = Queue()
    job = request.getfixturevalue(job)

    add_job_to_queue(job, JOB_QUEUE)

    assert JOB_QUEUE.qsize() == 1
