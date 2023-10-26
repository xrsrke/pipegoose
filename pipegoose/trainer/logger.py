from pipegoose.distributed import ParallelContext


class DistributedLogger:
    LEVELS = ["warning", ...]

    def __init__(self, parallel_context: ParallelContext):
        pass

    def set_level(self):
        pass

    def log(self):
        pass
