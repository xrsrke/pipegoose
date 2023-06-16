from contextlib import contextmanager
from queue import Queue
from threading import Thread
from typing import Dict, Generator, List, Tuple

import torch


def wait_and_execute_worker(device: torch.device, in_queue: Queue, out_queue: Queue):
    while True:
        func = in_queue.get()
        if func is None:
            break

        try:
            output = func()
        except Exception:
            # output, bool
            out_queue.put((None, False))
            continue

        out_queue.put((output, True))


@contextmanager
def spawn_worker(devices: List[torch.device]) -> Generator[Tuple[List[Queue], List[Queue]], None, None]:
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
