from abc import ABC, abstractclassmethod
from typing import NewType, Tuple

import torch.distributed.rpc as rpc

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.distributed.parallel_mode import ParallelMode

# NOTE: (microbatch_idx, partition_idx)
Task = NewType("Task", Tuple[int, int])


class Handshake(ABC):
    master_rank = None

    parallel_context = None
    parallel_mode = None

    def __init__(self, master_rank: int, parallel_context: ParallelContext, parallel_mode: ParallelMode):
        Handshake.master_rank = master_rank
        Handshake.parallel_context = parallel_context
        Handshake.parallel_mode = parallel_mode

    @abstractclassmethod
    def initiate(self):
        raise NotImplementedError

    @abstractclassmethod
    def confirm(self):
        raise NotImplementedError

    @abstractclassmethod
    def is_initiated(self) -> bool:
        raise NotImplementedError

    @abstractclassmethod
    def is_confirmed(self, clock_idx: int) -> bool:
        raise NotImplementedError

    @abstractclassmethod
    def is_all_confirmed(self, clock_idx: int) -> bool:
        raise NotImplementedError


class ProgressTracker(Handshake):
    """Pipeline parallelism's progress tracker."""

    progress = None
    clock_idx = None

    def is_initiated(self) -> bool:
        return self.progress is not None

    def initiate(self, progress):
        INITIAL_CLOCK_IDX = 0
        ProgressTracker._broadcast_tasks(progress, clock_idx=INITIAL_CLOCK_IDX)
        ProgressTracker._recv_tasks(progress, clock_idx=INITIAL_CLOCK_IDX)

    @staticmethod
    def _broadcast_tasks(progress, clock_idx):
        parallel_context = ProgressTracker.parallel_context
        parallel_mode = ProgressTracker.parallel_mode

        local_rank = parallel_context.get_local_rank(parallel_mode)
        local_world_size = parallel_context.get_world_size(parallel_mode)

        for local_dst in range(local_world_size):
            if local_dst == local_rank:
                continue

            global_dst = parallel_context.get_global_rank_from_local_rank(local_dst, parallel_mode)
            worker_name = parallel_context.get_worker_name(global_dst)
            rpc.rpc_sync(to=worker_name, func=ProgressTracker._recv_tasks, args=(progress, clock_idx))

    @staticmethod
    def _recv_tasks(progress, clock_idx):
        ProgressTracker.progress = progress
        ProgressTracker.clock_idx = clock_idx

    def is_confirmed(self, task: Task, clock_idx: int) -> bool:
        return self.progress[clock_idx][task] is True

    @staticmethod
    def is_all_confirmed(clock_idx: int) -> bool:
        progress = ProgressTracker.progress
        return all([progress[clock_idx][task] is True for task in progress[clock_idx]])

    def confirm(self, task: Task):
        # TODO: only non-scheduler ranks should confirm
        global_master_rank = self.parallel_context.get_global_rank_from_local_rank(self.master_rank, self.parallel_mode)
        master_worker_name = self.parallel_context.get_worker_name(global_master_rank)
        # rank = self.parallel_context.get_local_rank(self.parallel_mode)
        rpc.rpc_sync(master_worker_name, func=ProgressTracker._recv_confirm_from_worker, args=(task,))

        # NOTE: a worker node should confirm itself
        ProgressTracker._update_progress(task)

    @staticmethod
    def _update_progress(task: Task):
        clock_idx = ProgressTracker.clock_idx
        progress = ProgressTracker.progress
        progress[clock_idx][task] = True

    @staticmethod
    def _recv_confirm_from_worker(task: Task):
        ProgressTracker._update_progress(task)

        clock_idx = ProgressTracker.clock_idx
        if ProgressTracker.is_all_confirmed(clock_idx) is True:
            NEXT_CLOCK_IDX = clock_idx + 1
            ProgressTracker.clock_idx = NEXT_CLOCK_IDX
            # broadcast the progress to all worker nodes
            ProgressTracker._broadcast_tasks(ProgressTracker.progress, clock_idx=NEXT_CLOCK_IDX)
