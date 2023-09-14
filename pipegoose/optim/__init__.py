from abc import ABC, abstractclassmethod


class DistributedOptimizer(ABC):
    @abstractclassmethod
    def add_param_group(self):
        raise NotImplementedError("add_param_group is not implemented")

    @abstractclassmethod
    def load_state_dict(self):
        raise NotImplementedError("load_state_dict is not implemented")

    @abstractclassmethod
    def state_dict(self):
        raise NotImplementedError("state_dict is not implemented")

    @abstractclassmethod
    def step(self):
        raise NotImplementedError("step is not implemented")

    @abstractclassmethod
    def zero_grad(self):
        raise NotImplementedError("zero_grad is not implemented")
