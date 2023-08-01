import datetime

import torch


def init_dist():
    torch.distributed.init_process_group(
        backend="gloo",
        init_method="tcp://localhost:12349",
        rank=0,
        world_size=5,
        timeout=datetime.timedelta(seconds=60),
    )
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    init_dist()
    print("did")
