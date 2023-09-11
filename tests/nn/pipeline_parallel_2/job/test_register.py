from queue import Queue
from collections import OrderedDict

import pytest
from torch import nn

from pipegoose.nn.pipeline_parallel2._job.register import add_job_to_queue


@pytest.fixture
def model():
    model = nn.Sequential(OrderedDict([
        ('layer1', nn.Sequential(OrderedDict([
            ('fc', nn.Linear(4, 8)),
            ('relu', nn.ReLU())
        ]))),
        ('layer2', nn.Sequential(OrderedDict([
            ('fc', nn.Linear(8, 4)),
            ('relu', nn.ReLU())
        ]))),
        ('layer3', nn.Sequential(OrderedDict([
            ('fc', nn.Linear(4, 8)),
            ('relu', nn.ReLU())
        ]))),
    ]))
    return model


@pytest.mark.parametrize("job", ["forward_job", "backward_job"])
def test_register_job(request, job, parallel_context):
    JOB_QUEUE = Queue()
    job = request.getfixturevalue(job)

    add_job_to_queue(job, JOB_QUEUE, parallel_context)

    assert JOB_QUEUE.qsize() == 1
