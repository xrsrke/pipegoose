from dataclasses import dataclass
from queue import Queue


@dataclass
class JobQueue:
    """A queue for storing jobs."""

    PENDING_JOBS = Queue()
    SELECTED_JOBS = Queue()
