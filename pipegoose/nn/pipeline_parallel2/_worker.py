from abc import ABC, abstractclassmethod
from queue import Queue
from threading import Thread
from typing import Callable, List

from pipegoose.nn.pipeline_parallel2._utils import sleep

_MIN_WORKERS = 16
_MAX_WORKERS = 32


class Worker(Thread):
    def __init__(self, selected_jobs: Queue, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._selected_jobs = selected_jobs
        # self._running = False

    @property
    def running(self) -> bool:
        # return self._running
        return self.is_alive()

    def run(self):
        while True:
            job = self._selected_jobs.get()
            # self._running = True

            job.compute()

            # self._running = False


class WorkerPoolWatcher(Thread):
    def __init__(self, worker_pool: List[Worker], min_workers: int, max_workers: int, spawn_worker: Callable, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.worker_pool = worker_pool
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.spawn_worker = spawn_worker

    def run(self):
        while True:
            num_working = self._num_working_workers()

            # TODO: should we spawn more than initial workers?
            # TODO: fix race condition in some idle threads
            if num_working < self.max_workers:
                self.spawn_worker()

    def _num_working_workers(self) -> int:
        num_working = 0
        for worker in self.worker_pool:
            num_working += 1 if worker.running is True else 0
        return num_working


class JobSelector(Thread):
    def __init__(self, pending_jobs: Queue, selected_jobs: Queue, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pending_job = pending_jobs
        self._selected_jobs = selected_jobs

    def run(self):
        while True:
            job = self._select_job()
            self._selected_jobs.put(job)

    def _select_job(self):
        while len(self._pending_job) < 1:
            sleep()

            job = list(sorted(self._pending_job))[0]
            self._pending_job.remove(job)
            return job


class BaseWorkerManager(ABC, Thread):
    @abstractclassmethod
    def pending_jobs(self):
        raise NotImplementedError("not implemented")

    @abstractclassmethod
    def selected_jobs(self):
        raise NotImplementedError("not implemented")

    @abstractclassmethod
    def worker_pool(self):
        raise NotImplementedError("not implemented")

    @abstractclassmethod
    def spawn(self):
        raise NotImplementedError("not implemented")


class WorkerManager(BaseWorkerManager):
    def __init__(
        self,
        num_workers: int = _MIN_WORKERS,
        min_workers: int = _MIN_WORKERS,
        max_workers: int = _MAX_WORKERS,
        pending_jobs: Queue = Queue(),
        selected_jobs: Queue = Queue(),
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

    def spawn(self):
        self._spawn_job_selector()
        self._spawn_initial_workers()
        self._spawn_pool_watcher()

    def destroy(self):
        for worker in self.worker_pool:
            worker.join()

        # self.worker_pool.remove(worker)
