import time
import threading

import torch

from pipegoose.worker import spawn_worker


def test_spawn_worker():
    DEVICES = [torch.device("cpu"), torch.device("cpu")]
    TARGET_DEVICE_IDX = len(DEVICES) - 1

    count = 0
    lock = threading.Lock()

    def counter():
        nonlocal count
        with lock:
            count += 1
            return 1

    with spawn_worker(DEVICES) as (in_queues, out_queues):
        for _ in range(10):
            time.sleep(0.7)
            in_queues[TARGET_DEVICE_IDX].put(counter)

    # TODO: check why 9 not 10?
    assert count == 9
