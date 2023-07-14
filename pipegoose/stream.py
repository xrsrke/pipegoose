from typing import Union

import torch


class CPUStreamType:
    pass


StreamType = Union[CPUStreamType, torch.cuda.Stream]


def wait_stream(source: StreamType, target: StreamType):
    """Source stream waits for target stream completes it queues."""
    if isinstance(target, torch.cuda.Stream):
        if isinstance(source, torch.cuda.Stream):
            source.wait_stream(target)
        else:
            torch.cuda.synchronize()
