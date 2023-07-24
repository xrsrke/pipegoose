from .initializer import ProcessGroupInitializer


class DataParallelGroupInitializer(ProcessGroupInitializer):
    def init_dist_group(self):
        pass
