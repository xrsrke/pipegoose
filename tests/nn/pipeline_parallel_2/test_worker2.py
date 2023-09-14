from queue import Queue

from pipegoose.nn.pipeline_parallel2._worker import WorkerManager
from pipegoose.nn.pipeline_parallel2._utils import sleep


def test_worker_manager():
    NUM_WORKERS = 5
    MIN_WORKERS = 2
    MAX_WORKERS = 7

    worker_manager = WorkerManager(num_workers=NUM_WORKERS, min_workers=MIN_WORKERS, max_workers=MAX_WORKERS)
    worker_manager.spawn()

    # wait for workers to spawn
    sleep(1.69)

    assert worker_manager.num_workers == NUM_WORKERS
    assert worker_manager.min_workers == MIN_WORKERS
    assert worker_manager.max_workers == MAX_WORKERS

    assert len(worker_manager.worker_pool) >= MIN_WORKERS
    assert len(worker_manager.worker_pool) <= MAX_WORKERS
    assert isinstance(worker_manager.pending_jobs, Queue)
    assert isinstance(worker_manager.selected_jobs, Queue)

    # NOTE: since we don't have any jobs, all workers should be idle
    for worker in worker_manager.worker_pool:
        assert worker.is_running is False
        assert worker.is_alive() is True


def test_destroy_worker_manager():
    pass


def test_select_job_based_on_scheduler():
    pass


def test_execute_a_job_from_selected_job_queue():
    PENDING_JOBS = Queue()
    SELECTED_JOBS = Queue()
    QUEUE = []

    class FakeJob:
        def compute(self):
            QUEUE.append(1)

    worker_manager = WorkerManager(
        pending_jobs=PENDING_JOBS,
        selected_jobs=SELECTED_JOBS
    )
    job = FakeJob()
    worker_manager.spawn()

    PENDING_JOBS.put(job)
    assert PENDING_JOBS.qsize() == 1

    # NOTE: wait for job selector picks up the job
    sleep(2)

    assert QUEUE == [1]
    assert PENDING_JOBS.qsize() == 0
    assert SELECTED_JOBS.qsize() == 0


def test_execute_a_job_from_job_queue_and_send_the_output():
    pass
