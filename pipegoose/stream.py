from typing import Union

import torch


class CPUStreamType:
    pass


StreamType = Union[CPUStreamType, torch.cuda.Stream]


def is_cuda(stream: StreamType) -> bool:
    return True if isinstance(stream, torch.cuda.Stream) else False


def wait_stream(source: StreamType, target: StreamType):
    """Source stream waits for target stream completes it queues."""
    if isinstance(target, torch.cuda.Stream):
        if isinstance(source, torch.cuda.Stream):
            source.wait_stream(target)
        else:
            torch.cuda.synchronize()


def default_stream(device: torch.device) -> StreamType:
    """Get the default stream of the device."""
    if device.type == "cpu":
        return CPUStreamType()
    elif device.type == "cuda":
        return torch.cuda.default_stream(device)
    else:
        raise ValueError(f"Unknown device type: {device.type}")


def get_device(stream: StreamType) -> torch.device:
    """Get the device of the stream."""
    if isinstance(stream, torch.cuda.Stream):
        return stream.device
    return torch.device("cpu")


def use_stream(stream: StreamType):
    if not is_cuda(stream):
        yield
        return

    with torch.cuda.stream(stream):
        yield


def record_stream(tensor: torch.Tensor, stream: StreamType) -> None:
    if is_cuda(stream):
        tensor.record_stream(stream)
