from abc import ABC, abstractclassmethod


class BaseDistributedOptimizer(ABC):
    """A base class for distributed optimizer."""

    @abstractclassmethod
    def defaults(self):
        raise NotImplementedError("defaults is not implemented")

    @abstractclassmethod
    def param_groups(self):
        raise NotImplementedError("param_groups is not implemented")

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
