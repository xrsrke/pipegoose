import pytest

from pipegoose.nn.pipeline_parallel._job.callback import Callback
from pipegoose.nn.pipeline_parallel._job.job import Job


# NOTE: We don't want to rely on the behavior of other jobs, like forward job
# and backward job. so we create a dummy job solely to test callbacks
class DummyJob(Job):
    def run_compute(self):
        return self.function(self.input.data)


@pytest.fixture
def job(forward_package):
    def function(*args, **kwargs):
        pass

    return DummyJob(function, forward_package)


@pytest.fixture(scope="function")
def cbs():
    class Callback1(Callback):
        pass

    class Callback2(Callback):
        pass

    class Callback3(Callback):
        pass

    return [Callback1(), Callback2, Callback3]


def test_a_callback_access_job_attributes(job):
    QUEUE = []

    class AccessJobAttributesCallback(Callback):
        def after_compute(self):
            QUEUE.append(self.job.key)

    job.add_cb(AccessJobAttributesCallback)
    job.compute()

    assert len(QUEUE) == 1
    assert QUEUE == [job.key]


def test_run_callbacks_by_order(job):
    QUEUE = []

    class Callback1(Callback):
        order = 0

        def after_compute(self):
            QUEUE.append(1)

    class Callback2(Callback):
        order = 1

        def after_compute(self):
            QUEUE.append(2)

    class Callback3(Callback):
        order = 2

        def after_compute(self):
            QUEUE.append(3)

    job.add_cbs([Callback3, Callback1, Callback2])
    job.compute()

    assert QUEUE == [1, 2, 3]


def test_create_and_run_a_callback(job):
    QUEUE = []

    class AddToQueueCallback(Callback):
        def after_compute(self):
            QUEUE.append(69)

    cb = AddToQueueCallback()

    assert isinstance(cb.order, int)

    job.add_cb(cb)
    job.compute()

    assert QUEUE == [69]


def test_add_and_remove_a_callback(job):
    class ToyCallback(Callback):
        pass

    N_ORIG_CBS = len(job.cbs)
    cb = ToyCallback()

    job.add_cb(cb)
    assert len(job.cbs) == 1 + N_ORIG_CBS

    job.remove_cb(cb)
    assert len(job.cbs) == N_ORIG_CBS


def test_add_and_remove_a_list_of_callback(job, cbs):
    N_ORIG_CBS = len(job.cbs)

    job.add_cbs(cbs)
    assert len(job.cbs) == 3 + N_ORIG_CBS

    job.remove_cbs(cbs)
    assert len(job.cbs) == N_ORIG_CBS
