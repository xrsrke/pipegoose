import torch
from pipegoose.worker import spawn_worker


def test_spawn_worker():
    DEVICES = [torch.device("cpu"), torch.device("cpu")]
    TARGET_DEVICE_IDX = 1

    count = 0

    def counter():
        nonlocal count
        count += 1
        return count

    with spawn_worker(DEVICES) as (in_queues, out_queues):
        for _ in range(10):
            in_queues[TARGET_DEVICE_IDX].put(counter)

    assert count == 10
