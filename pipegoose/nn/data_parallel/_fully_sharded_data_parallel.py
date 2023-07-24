from torch import nn


class FullyShardedDataParallel(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        # start_time = time.time()

        # params = []
        # for param_name, param in module.named_parameters():
