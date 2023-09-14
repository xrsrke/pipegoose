import pytest

from pipegoose.nn.pipeline_parallel2._job.callback import Callback


@pytest.fixture(scope="function")
def cbs():
    class Callback1(Callback):
        pass

    class Callback2(Callback):
        pass

    class Callback3(Callback):
        pass

    return [Callback1(), Callback2, Callback3]


@pytest.fixture(scope="function")
def cb():
    class ToyCallback(Callback):
        pass

    return ToyCallback()


def test_create_and_run_a_callback(forward_job):
    QUEUE = []

    class AddToQueueCallback(Callback):
        def after_compute(self):
            QUEUE.append(69)

    forward_job.add_cb(AddToQueueCallback)
    forward_job.compute()

    assert QUEUE == [69]


def test_add_and_remove_a_callback(forward_job, cb):
    N_ORIG_CBS = len(forward_job.cbs)

    forward_job.add_cb(cb)
    assert len(forward_job.cbs) == 1 + N_ORIG_CBS

    forward_job.remove_cb(cb)
    assert len(forward_job.cbs) == N_ORIG_CBS


def test_add_and_remove_a_list_of_callback(forward_job, cbs):
    N_ORIG_CBS = len(forward_job.cbs)

    forward_job.add_cbs(cbs)
    assert len(forward_job.cbs) == 3 + N_ORIG_CBS

    forward_job.remove_cbs(cbs)
    assert len(forward_job.cbs) == N_ORIG_CBS


def test_a_callback_access_job_attributes(forward_job):
    QUEUE = []

    class AccessJobAttributesCallback(Callback):
        def after_compute(self):
            QUEUE.append(self.job.key)

    forward_job.add_cb(AccessJobAttributesCallback)
    forward_job.compute()

    assert len(QUEUE) == 1
    assert QUEUE == [forward_job.key]


def test_run_callbacks_by_order(forward_job):
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

    forward_job.add_cbs([Callback3, Callback1, Callback2])
    forward_job.compute()

    assert QUEUE == [1, 2, 3]
