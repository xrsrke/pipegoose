from dataclasses import dataclass
from queue import Queue

# class Queue:
#     def __init__(self):
#         self._queue = Queue()


@dataclass
class JobQueue:
    """A queue for storing jobs."""

    PENDING_JOBS = Queue()
    SELECTED_JOBS = Queue()


ACTIVATIONS = {}
