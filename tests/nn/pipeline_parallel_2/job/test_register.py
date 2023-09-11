from queue import Queue
from collections import OrderedDict

import pytest
from torch import nn

from pipegoose.nn.pipeline_parallel2._job.creator import JobCreator
from pipegoose.nn.pipeline_parallel2._job.register import JobRegister


@pytest.fixture
def backward_job(backward_package, parallel_context):
    return JobCreator(parallel_context).create(backward_package)


@pytest.fixture
def forward_job(forward_package, parallel_context):
    return JobCreator(parallel_context).create(forward_package)


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
    job_register = JobRegister(JOB_QUEUE, parallel_context)

    job_register.registry(job)

    assert JOB_QUEUE.qsize() == 1
