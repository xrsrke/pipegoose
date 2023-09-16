import threading
from abc import ABC, abstractclassmethod
from queue import Queue
from typing import Callable, List

from pipegoose.constants import PIPELINE_MAX_WORKERS, PIPELINE_MIN_WORKERS
from pipegoose.nn.pipeline_parallel2._job.job import Job
from pipegoose.nn.pipeline_parallel2._utils import sleep
from pipegoose.nn.pipeline_parallel2.queue import JobQueue


class Worker(threading.Thread):
    """A worker that execute job."""

    def __init__(self, selected_jobs: Queue, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._selected_jobs = selected_jobs
        self._running = False
        # self._stop_event = threading.Event()

    @property
    def is_running(self) -> bool:
        return self._running

    # def stop(self):
    #     self._stop_event.set()

    # def stopped(self):
    #     return self._stop_event.is_set()

    def run(self):
        while True:
            job = self._selected_jobs.get()
            self._running = True
            job.compute()
            self._running = False


class WorkerPoolWatcher(threading.Thread):
    def __init__(self, worker_pool: List[Worker], min_workers: int, max_workers: int, spawn_worker: Callable, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.worker_pool = worker_pool
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.spawn_worker = spawn_worker

    def run(self):
        while True:
            num_working = self._num_working_workers()

            # NOTE: only spawn new workers if
            # all the current workers are working (num_working == num_all_workers)
            # and the number of workers is less than the max_workers
            if num_working == len(self.worker_pool) and num_working < self.max_workers:
                self.spawn_worker()

    def _num_working_workers(self) -> int:
        num_working = 0

        for worker in self.worker_pool:
            num_working += 1 if worker.is_running is True else 0

        return num_working


class JobSelector(threading.Thread):
    def __init__(self, pending_jobs: Queue, selected_jobs: Queue, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pending_job = pending_jobs
        self._selected_jobs = selected_jobs

    def run(self):
        while True:
            job = self._select_job()
            self._selected_jobs.put(job)

    def _select_job(self) -> Job:
        while self._pending_job.empty():
            sleep()

        job = self._pending_job.get()
        return job


class BaseWorkerManager(ABC):
    @abstractclassmethod
    def spawn(self):
        raise NotImplementedError

    @abstractclassmethod
    def destroy(self):
        raise NotImplementedError


class _WorkerManager(BaseWorkerManager, threading.Thread):
    def __init__(
        self,
        num_workers: int = PIPELINE_MIN_WORKERS,
        min_workers: int = PIPELINE_MIN_WORKERS,
        max_workers: int = PIPELINE_MAX_WORKERS,
        pending_jobs: Queue = JobQueue.PENDING_JOBS,
        selected_jobs: Queue = JobQueue.SELECTED_JOBS,
    ):
        # job just created but not yet selected
        self._pending_jobs = pending_jobs
        # job selected to be executed
        self._selected_jobs = selected_jobs
        self._worker_pool = []
        self.num_workers = num_workers
        self.min_workers = min_workers
        self.max_workers = max_workers

    @property
    def pending_jobs(self) -> Queue:
        return self._pending_jobs

    @property
    def selected_jobs(self) -> Queue:
        return self._selected_jobs

    @property
    def worker_pool(self) -> List[Worker]:
        return self._worker_pool

    def spawn(self):
        self._spawn_job_selector()
        self._spawn_initial_workers()
        self._spawn_pool_watcher()

    def _spawn_job_selector(self):
        job_selector = JobSelector(self._pending_jobs, self._selected_jobs)
        job_selector.setDaemon(True)
        job_selector.start()

        self.job_selector = job_selector

    def _spawn_pool_watcher(self):
        pool_watcher = WorkerPoolWatcher(
            self.worker_pool, min_workers=self.min_workers, max_workers=self.max_workers, spawn_worker=self._spawn_a_worker
        )
        pool_watcher.setDaemon(True)
        pool_watcher.start()

        # TODO: delete after testing
        self.pool_watcher = pool_watcher

    def _spawn_a_worker(self):
        worker = Worker(selected_jobs=self._selected_jobs)
        worker.setDaemon(True)
        worker.start()
        self._worker_pool.append(worker)

    def _spawn_initial_workers(self):
        for _ in range(self.num_workers):
            self._spawn_a_worker()

    def destroy(self):
        # Create a copy of the worker pool to iterate over
        worker_pool_copy = self.worker_pool.copy()

        for worker in worker_pool_copy:
            # Terminate the worker thread
            # worker.stop()
            worker.join()

            # Remove the worker from the original worker pool
            self.worker_pool.remove(worker)


def WorkerManager(
    num_workers: int = PIPELINE_MIN_WORKERS,
    min_workers: int = PIPELINE_MIN_WORKERS,
    max_workers: int = PIPELINE_MAX_WORKERS,
    pending_jobs: Queue = JobQueue.PENDING_JOBS,
    selected_jobs: Queue = JobQueue.SELECTED_JOBS,
):
    worker_manager = _WorkerManager(
        num_workers=num_workers,
        min_workers=min_workers,
        max_workers=max_workers,
        pending_jobs=pending_jobs,
        selected_jobs=selected_jobs,
    )

    return worker_manager
