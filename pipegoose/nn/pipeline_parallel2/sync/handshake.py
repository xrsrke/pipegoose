import random
from abc import ABC, abstractclassmethod
from dataclasses import dataclass
from queue import Queue
from time import sleep
from typing import Dict, List

import torch.distributed.rpc as rpc

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.distributed.parallel_mode import ParallelMode

SessionId = str


class Handshake(ABC):
    def __init__(self, parallel_context: ParallelContext, parallel_mode: ParallelMode):
        self.parallel_context = parallel_context
        self.parallel_mode = parallel_mode

        self._session_id: str = None
        self._queue: Dict[SessionId, Queue] = {}
        self._ranks_confirmed: Dict[SessionId, List[int]] = set()

        self._data = None
        self._clock_idx = 0

    @abstractclassmethod
    def initiate(self):
        raise NotImplementedError

    def _generate_session_id(self) -> int:
        return random.randint(0, 9999)

    @abstractclassmethod
    def confirm(self):
        raise NotImplementedError

    @abstractclassmethod
    def is_initiated(self):
        raise NotImplementedError

    @abstractclassmethod
    def is_confirmed(self):
        raise NotImplementedError

    @abstractclassmethod
    def is_all_confirmed(self):
        raise NotImplementedError

    @abstractclassmethod
    def wait_until_all_confirmed(self):
        raise NotImplementedError


@dataclass
class SessionMetadata:
    clock_idx: int
    parallel_mode: ParallelMode


class SchedulerHandshake(Handshake):
    NUM_SECONDS_IDLE = 0.5
    # TODO: make this configurable
    MASTER_RANK = 0

    progress = None
    clock_idx = 0

    def is_initiated(self) -> bool:
        return self.progress is not None

    def initiate(self, data):
        rank = self.parallel_context.get_local_rank(self.parallel_mode)
        world_size = self.parallel_context.get_world_size(self.parallel_mode)
        for dst in range(world_size):
            if dst == rank:
                continue

            worker_name = self.parallel_context.get_worker_name(dst)
            rpc.rpc_sync(to=worker_name, func=SchedulerHandshake._recv_execution_plan, args=(data,))

        SchedulerHandshake._recv_execution_plan(data)

    @staticmethod
    def _recv_execution_plan(data):
        SchedulerHandshake.progress = data

    def is_confirmed(self, task, clock_idx: int) -> bool:
        return self.progress[clock_idx][task] is True

    @staticmethod
    def is_all_confirmed(clock_idx: int) -> bool:
        progress = SchedulerHandshake.progress
        return all([progress[clock_idx][task] is True for task in progress[clock_idx]])

    def confirm(self, task):
        # TODO: only non-scheduler ranks should confirm
        master_worker_name = self.parallel_context.get_worker_name(self.MASTER_RANK)
        # rank = self.parallel_context.get_local_rank(self.parallel_mode)
        rpc.rpc_sync(master_worker_name, func=SchedulerHandshake._recv_confirm_from_worker, args=(task,))

        # NOTE: a worker node should confirm itself
        SchedulerHandshake._update_progress(task)

    @staticmethod
    def _update_progress(task):
        clock_idx = SchedulerHandshake.clock_idx
        progress = SchedulerHandshake.progress
        progress[clock_idx][task] = True

    @staticmethod
    def _recv_confirm_from_worker(task):
        SchedulerHandshake._update_progress(task)

        clock_idx = SchedulerHandshake.clock_idx
        if SchedulerHandshake.is_all_confirmed(clock_idx) is True:
            SchedulerHandshake.clock_idx += 1

    def wait_until_all_confirmed(self):
        if self.parallel_context.is_first_rank() is True:
            while True:
                if self.is_all_confirmed() is True:
                    break
                else:
                    sleep(self.NUM_SECONDS_IDLE)
        else:
            pass
