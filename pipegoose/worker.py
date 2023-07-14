from contextlib import contextmanager
from dataclasses import dataclass
from queue import Queue
from threading import Thread
from typing import Annotated, Any, Dict, Generator, List, NoReturn, Tuple

import torch

from pipegoose.task import Task


class InQueueTask:
    pass


class OutQueueTask:
    pass


@dataclass
class QueueOutput:
    task: Task
    output: Any
    is_done: bool = False


def wait_and_execute(device: torch.device, in_queue: Queue, out_queue: Queue) -> NoReturn:
    """Wait for a task and execute it."""
    while True:
        task = in_queue.get()

        if task.is_done is True:
            break

        try:
            output = task.compute()
        except Exception:
            out_queue.put(QueueOutput(task=task, output=None, is_done=False))
            continue

        out_queue.put(QueueOutput(task=task, output=output, is_done=True))


@contextmanager
def spawn_worker(
    devices: List[torch.device],
) -> Generator[
    Tuple[
        Annotated[List[Queue], "A list of tasks to be executed"],
        Annotated[List[Queue], "A list of tasks has been executed"],
    ],
    None,
    None,
]:
    """Spawn new worker threads."""
    in_queues: List[Queue] = []
    out_queues: List[Queue] = []

    workers: Dict[torch.device, Tuple[Queue, Queue]] = {}

    for device in devices:
        try:
            in_queue, out_queue = workers[device]
        except KeyError:
            in_queue = Queue()
            out_queue = Queue()
            workers[device] = (in_queue, out_queue)

            thread = Thread(target=wait_and_execute, args=(device, in_queue, out_queue), daemon=True)
            thread.start()

        in_queues.append(in_queue)
        out_queues.append(out_queue)

    yield (in_queues, out_queues)


# class Worker:
#     def __init__(self, devices: List[torch.device]):
#         self.devices = devices
#         self._streams = [torch.cuda.Stream(device=device) for device in devices]
#         self.in_queues: List[Queue] = []
#         self.out_queues: List[Queue] = []

#     @contextmanager
#     def spawn(self) -> Generator[Tuple[Queue, Queue]]:
#         pass
