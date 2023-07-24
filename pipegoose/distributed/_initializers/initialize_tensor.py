from pipegoose.distributed._initializers.initializer import ProcessGroupInitializer


class TensorParallelGroupInitializer(ProcessGroupInitializer):
    def init_dist_group(self):
        pass
