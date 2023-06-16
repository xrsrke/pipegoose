from abc import abstractclassmethod
from typing import Iterable, List, Tuple


class BaseScheduler:
    @abstractclassmethod
    def generate(self):
        raise NotImplementedError


class DetermisticScheduler(BaseScheduler):
    def __init__(self, n_patritions: int, n_microbatches: int):
        assert (
            n_patritions > 0
        ), "The number of partitions must be \
            greater than 0"
        assert (
            n_microbatches > 0
        ), "The number of microbatches must be \
            greater than 0"

        self.n_patritions = n_patritions
        self.n_microbatches = n_microbatches

    def generate(self) -> Iterable[List[Tuple[int, int]]]:
        # (microbatch_idx, partrition_idx)
        n_clock_cycles = self.n_patritions + self.n_microbatches - 1

        for clock_idx in range(n_clock_cycles):
            start_partrition = max(clock_idx + 1 - self.n_microbatches, 0)
            end_patrition = min(clock_idx + 1, self.n_patritions)

            tasks = []
            for partrition_idx in range(start_partrition, end_patrition):
                microbatch_idx = clock_idx - partrition_idx
                tasks.append((microbatch_idx, partrition_idx))

            yield tasks
