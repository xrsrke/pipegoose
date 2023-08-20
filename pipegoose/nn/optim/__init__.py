from abc import ABC, abstractclassmethod


class DistributedOptimizer(ABC):
    @abstractclassmethod
    def step(self):
        pass
