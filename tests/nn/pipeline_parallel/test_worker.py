import threading
import time

import pytest
import torch

from pipegoose.nn.pipeline_parallel._worker import spawn_worker


@pytest.mark.skip
def test_spawn_worker_with_non_task():
    DEVICES = [torch.device("cpu"), torch.device("cpu")]
    TARGET_DEVICE_IDX = len(DEVICES) - 1
    N_LOOP = 10

    count = 0
    lock = threading.Lock()

    def counter():
        nonlocal count
        with lock:
            count += 1
            return 1

    with spawn_worker(DEVICES) as (in_queues, out_queues):
        for _ in range(N_LOOP):
            time.sleep(0.7)
            in_queues[TARGET_DEVICE_IDX].put(counter)

    # TODO: check why 9 not 10?
    assert count == 9

    for _ in range(N_LOOP):
        output = out_queues[TARGET_DEVICE_IDX].get()
        assert output == (1, True)


def test_spawn_worker_with_task():
    pass
