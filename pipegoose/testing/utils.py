import random
import socket
from functools import partial
from typing import Callable

import torch.multiprocessing as mp


def find_free_port(min_port: int = 2000, max_port: int = 65000) -> int:
    while True:
        port = random.randint(min_port, max_port)
        try:
            with socket.socket() as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("localhost", port))
                return port
        except OSError:
            continue


def spawn(func: Callable, world_size: int = 1, **kwargs):
    if kwargs.get("port") is None:
        port = find_free_port()
    else:
        port = kwargs["port"]
        kwargs.pop("port")

    wrapped_func = partial(func, world_size=world_size, port=port, **kwargs)
    mp.spawn(wrapped_func, nprocs=world_size)
