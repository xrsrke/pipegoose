from pipegoose.distributed._initializers.initializer import ProcessGroupInitializer


class PipelineParallelGroupInitializer(ProcessGroupInitializer):
    def init_dist_group(self):
        pass
