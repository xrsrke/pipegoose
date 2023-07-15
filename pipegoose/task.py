from enum import Enum
from typing import Any, Callable

from pipegoose.microbatch import Batch


class BaseTask:
    def __call__(self, *args, **kwargs) -> Any:
        return self.compute(*args, **kwargs)


class TaskStatus(Enum):
    DONE = 1
    NOT_DONE = 2


class Task:
    def __init__(self, compute: Callable[[], Batch], is_done: TaskStatus = TaskStatus.NOT_DONE):
        self._compute = compute
        self.is_done = is_done

    def compute(self) -> Batch:
        return self._compute()
