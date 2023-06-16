from typing import Any, Callable


class BaseTask:
    def __call__(self, *args, **kwargs) -> Any:
        return self.func(*args, **kwargs)


class Task(BaseTask):
    def __init__(self, func: Callable) -> Any:
        self.func = func
