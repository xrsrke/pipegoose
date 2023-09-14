from enum import Enum
from abc import ABC


class CallbackEvent(Enum):
    BEFORE_COMPUTE = "before_compute"
    AFTER_COMPUTE = "after_compute"


class Callback(ABC):
    """Callback for a job."""

    order = 0

    @property
    def name(self) -> str:
        return self.__name__

    def before_compute(self):
        pass

    def after_compute(self):
        pass
