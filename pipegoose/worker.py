from contextlib import contextmanager
from queue import Queue
from typing import Dict, Generator, List, Tuple

import torch


@contextmanager
def spawn_worker(devices: List[torch.device]) -> Generator[Tuple[Queue, Queue]]:
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


# class Worker:
#     def __init__(self, devices: List[torch.device]):
#         self.devices = devices
#         self._streams = [torch.cuda.Stream(device=device) for device in devices]
#         self.in_queues: List[Queue] = []
#         self.out_queues: List[Queue] = []

#     @contextmanager
#     def spawn(self) -> Generator[Tuple[Queue, Queue]]:
#         pass
