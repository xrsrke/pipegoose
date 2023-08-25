from random import randint

from pipegoose.nn.pipeline_parallel2._utils import sleep
from pipegoose.nn.pipeline_parallel2._worker import WorkerManager


def test_worker_manager():
    NUM_WORKERS = 5
    MIN_WORKERS = 2
    MAX_WORKERS = 7

    def kill_worker(worker_pool):
        randnum = randint(0, len(worker_pool) - 1)
        worker_pool.remove(worker_pool[randnum])

    worker_manager = WorkerManager(num_workers=NUM_WORKERS, min_workers=MIN_WORKERS, max_workers=MAX_WORKERS)
    worker_manager.spawn()

    # wait for workers to spawn
    sleep(1.69)

    assert worker_manager.num_workers == NUM_WORKERS
    assert worker_manager.min_workers == MIN_WORKERS
    assert worker_manager.max_workers == MAX_WORKERS
    assert len(worker_manager.worker_pool) >= MIN_WORKERS
    assert len(worker_manager.worker_pool) <= MAX_WORKERS

    for worker in worker_manager.worker_pool:
        assert worker.running is True
        assert worker.is_alive() is True

    # knock out some threads, and check if new threads are spawned
    kill_worker(worker_manager.worker_pool)
    sleep(1.69)

    assert len(worker_manager.worker_pool) <= MAX_WORKERS
