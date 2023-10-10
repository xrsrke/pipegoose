import threading
import time
from abc import ABC, abstractclassmethod
from queue import Queue
from time import sleep
from typing import Dict, List, NewType

import torch.distributed.rpc as rpc

from pipegoose.distributed.functional import barrier
from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.distributed.parallel_mode import ParallelMode
from pipegoose.nn.pipeline_parallel2.sync.callback import Callback
from pipegoose.nn.pipeline_parallel2.task import Task

ClockIdx = NewType("ClockIdx", int)
Progress = NewType("Progress", Dict[ClockIdx, Dict[Task, bool]])


_PROGRESS_TRACKER = None


def set_progress_tracker(progress_tracker):
    global _PROGRESS_TRACKER
    _PROGRESS_TRACKER = progress_tracker


def get_progress_tracker():
    global _PROGRESS_TRACKER
    return _PROGRESS_TRACKER


class Handshake(ABC):
    master_rank: int = None
    callbacks: List[Callback] = []

    parallel_context: ParallelContext = None
    parallel_mode: ParallelMode = None

    def __init__(
        self,
        master_rank: int,
        callbacks: List[Callback] = [],
        parallel_context: ParallelContext = None,
        parallel_mode: ParallelMode = None,
    ):
        assert isinstance(
            parallel_context, ParallelContext
        ), f"parallel_context must be an instance of ParallelContext, got {type(parallel_context)}"
        assert isinstance(
            parallel_mode, ParallelMode
        ), f"parallel_mode must be an instance of ParallelMode, got {type(parallel_mode)}"

        Handshake.master_rank = master_rank
        Handshake.callbacks = callbacks
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

    @staticmethod
    def _run_callback(event_name: str, *args, **kwargs):
        sorted_callbacks = sorted(Handshake.callbacks, key=lambda x: x.order)

        for callback in sorted_callbacks:
            event_method = getattr(callback, event_name, None)
            if event_method is not None:
                event_method(*args, **kwargs)


class ProgressTracker(Handshake):
    """Pipeline parallelism's progress tracker."""

    progress: Progress = None
    clock_idx: int = None
    update_progress_lock = threading.Lock()

    def is_initiated(self) -> bool:
        return self.progress is not None

    def initiate(self, progress: Progress):
        INITIAL_CLOCK_IDX = 0
        ProgressTracker._broadcast_tasks(progress, clock_idx=INITIAL_CLOCK_IDX, is_init=True)
        ProgressTracker.progress = progress
        ProgressTracker.clock_idx = INITIAL_CLOCK_IDX

    @staticmethod
    def _broadcast_tasks(progress, clock_idx, is_init=False):
        parallel_context = ProgressTracker.parallel_context
        parallel_mode = ProgressTracker.parallel_mode

        local_rank = parallel_context.get_local_rank(parallel_mode)
        local_world_size = parallel_context.get_world_size(parallel_mode)

        for local_dst in range(local_world_size):
            if local_dst == local_rank and is_init is False:
                # NOTE: since we skip the master node, we need to manually run the callback
                ProgressTracker._run_callback("after_new_clock_cycle", progress=progress, clock_idx=clock_idx)
                continue

            global_dst = parallel_context.get_global_rank_from_local_rank(local_dst, parallel_mode)
            worker_name = parallel_context.get_worker_name(global_dst)
            rpc.rpc_sync(to=worker_name, func=ProgressTracker._recv_tasks, args=(progress, clock_idx, is_init))

    @staticmethod
    def _recv_tasks(progress: Progress, clock_idx: int, is_init):
        with ProgressTracker.update_progress_lock:
            ProgressTracker.progress = progress
            ProgressTracker.clock_idx = clock_idx

            # NOTE: don't increase a new clock cycle if just initializing it
            # NOTE: after a worker node receives the progress, it should run the callback
            if is_init is False:
                ProgressTracker._run_callback("after_new_clock_cycle", progress=progress, clock_idx=clock_idx)

    def is_confirmed(self, task: Task, clock_idx: int) -> bool:
        return self.progress[clock_idx][task] is True

    @staticmethod
    def is_all_confirmed(clock_idx: int) -> bool:
        progress = ProgressTracker.progress
        return all([progress[clock_idx][task] is True for task in progress[clock_idx]])

    def confirm(self, task: Task):
        time.sleep(0.1)

        master_rank = self.parallel_context.get_global_rank_from_local_rank(self.master_rank, self.parallel_mode)
        rank = self.parallel_context.get_global_rank()

        print(f"confirm, clock_idx={self.clock_idx}, rank={rank}")

        if rank == master_rank:
            # NOTE: if master node confirm itself, then no need rpc call
            ProgressTracker._recv_confirm_from_worker(task)
        else:
            # NOTE: after a worker node confirms, it should tell the master node
            master_worker_name = self.parallel_context.get_worker_name(master_rank)
            rpc.rpc_sync(master_worker_name, func=ProgressTracker._recv_confirm_from_worker, args=(task,))

        time.sleep(0.1)

    @staticmethod
    def _recv_confirm_from_worker(task: Task):
        def _update_local_progress(task: Task):
            clock_idx = ProgressTracker.clock_idx
            progress = ProgressTracker.progress
            progress[clock_idx][task] = True

        _update_local_progress(task)

        clock_idx = ProgressTracker.clock_idx
        if ProgressTracker.is_all_confirmed(clock_idx) is True:
            NEXT_CLOCK_IDX = clock_idx + 1
            ProgressTracker.clock_idx = NEXT_CLOCK_IDX
            ProgressTracker._broadcast_tasks(ProgressTracker.progress, clock_idx=NEXT_CLOCK_IDX, is_init=False)


class ParallelGroupHandshake(Handshake):
    # NOTE: create a queue for storing confirmed ranks
    CONFIRMED_QUEUE: Dict[ParallelMode, List[int]] = {}
    # NOTE: wait until all ranks confirm
    CONTINUE_QUEUE: Dict[ParallelMode, Queue] = {}

    PARALLEL_CONTEXT: ParallelContext = None

    def __init__(self, parallel_context: ParallelContext, parallel_mode: ParallelMode):
        self.parallel_context = parallel_context
        self.parallel_mode = parallel_mode
        ParallelGroupHandshake.PARALLEL_CONTEXT = self.parallel_context

    def initiate(self):
        """Initiate the handshake."""
        ParallelGroupHandshake.CONFIRMED_QUEUE[self.parallel_mode] = []
        ParallelGroupHandshake.CONTINUE_QUEUE[self.parallel_mode] = Queue()

        # NOTE: make sure all ranks in the parallel mode are initiated
        # before continuing
        barrier(self.parallel_context, self.parallel_mode)

    def is_initiated(self):
        pass

    def is_confirmed(self):
        pass

    def is_all_confirmed(self):
        pass

    def confirm(self):
        master_rank = self._get_master_rank(self.parallel_mode)
        master_worker_name = self.parallel_context.get_worker_name(master_rank)
        rank = self.parallel_context.get_local_rank(self.parallel_mode)

        rpc.rpc_sync(
            to=master_worker_name,
            func=ParallelGroupHandshake._recv_confirm_from_worker_rank,
            args=(rank, self.parallel_mode),
        )

        print(f"rank={rank} confirm")

    @staticmethod
    def _recv_confirm_from_worker_rank(rank: int, parallel_mode: ParallelMode):
        confirmed_queue = ParallelGroupHandshake.CONFIRMED_QUEUE[parallel_mode]
        confirmed_queue.append(rank)

        print(f"master received confirm from {rank}")

        local_world_size = ParallelGroupHandshake.PARALLEL_CONTEXT.get_world_size(parallel_mode)
        num_worker_ranks = local_world_size
        if len(confirmed_queue) == num_worker_ranks:
            ParallelGroupHandshake.send_continue_to_worker_all_ranks(parallel_mode)

    @staticmethod
    def send_continue_to_worker_all_ranks(parallel_mode: ParallelMode):
        parallel_context = ParallelGroupHandshake.PARALLEL_CONTEXT
        ranks_in_group = parallel_context.get_ranks_in_group(parallel_mode)
        for other_rank in ranks_in_group:
            other_worker_name = parallel_context.get_worker_name(other_rank)
            rpc.rpc_sync(
                to=other_worker_name,
                func=ParallelGroupHandshake._recv_continue_from_master_rank,
                args=(parallel_mode,),
            )

    @staticmethod
    def _recv_continue_from_master_rank(parallel_mode):
        ParallelGroupHandshake.CONTINUE_QUEUE[parallel_mode].put(True)

    def _get_master_rank(self, parallel_mode: ParallelMode):
        # NOTE: the master rank is the first rank in the group
        ranks_in_group = self.parallel_context.get_ranks_in_group(parallel_mode)
        return ranks_in_group[0]

    def barrier(self):
        """Wait until all ranks in the parallel mode confirm."""
        while self.CONTINUE_QUEUE[self.parallel_mode].empty():
            sleep(0.5)

        # NOTE: block CPU execution until all ranks confirm
        _ = self.CONTINUE_QUEUE[self.parallel_mode].get()
