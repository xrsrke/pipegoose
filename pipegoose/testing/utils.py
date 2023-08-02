import random
import socket
from functools import partial

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


def spawn(func, nprocs=1, **kwargs):
    # port = find_free_port()
    # wrapped_func = partial(func, world_size=nprocs, port=port, **kwargs)
    wrapped_func = partial(func, **kwargs)
    mp.spawn(wrapped_func, nprocs=nprocs)
