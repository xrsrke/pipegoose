from typing import Any, Callable


class BaseTask:
    def __call__(self, *args, **kwargs) -> Any:
        return self.compute(*args, **kwargs)


class Task(BaseTask):
    def __init__(self, compute: Callable) -> Any:
        self.compute = compute
