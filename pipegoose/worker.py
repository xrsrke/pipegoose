from contextlib import contextmanager
from dataclasses import dataclass
from queue import Queue
from threading import Thread
from typing import Annotated, Any, Dict, Generator, List, Tuple

import torch


@dataclass
class QueueOutput:
    output: Any
    is_success: bool
    is_done: bool = False


def wait_and_execute_worker(device: torch.device, in_queue: Queue, out_queue: Queue) -> None:
    while True:
        task = in_queue.get()
        if task is None:
            break

        try:
            output = task()
        except Exception:
            out_queue.put(QueueOutput(output=None, is_success=False))
            continue

        out_queue.put(QueueOutput(output=output, is_success=True))


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

            thread = Thread(target=wait_and_execute_worker, args=(device, in_queue, out_queue), daemon=True)
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
