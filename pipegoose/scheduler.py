from abc import ABC, abstractclassmethod
from typing import Annotated, Iterable, List, Tuple


class BaseScheduler(ABC):
    @abstractclassmethod
    def generate(self):
        raise NotImplementedError


class DetermisticScheduler(BaseScheduler):
    """
    torchgpipe: On-the-fly Pipeline Parallelism for Training Giant Models
    https://arxiv.org/abs/2004.09910

    Section 3.2.1: Forward Dependency: Deterministic Clock-cycle
    """

    def generate(
        self,
        n_microbatches: int,
        n_partitions: int,
    ) -> Iterable[List[Tuple[Annotated[int, "batch_idx"], Annotated[int, "partition_idx"]]]]:
        assert (
            n_microbatches > 0
        ), "The number of microbatches must be \
            greater than 0"
        assert (
            n_partitions > 0
        ), "The number of partitions must be \
            greater than 0"

        self.n_partitions = n_partitions
        self.n_microbatches = n_microbatches
        n_clock_cycles = self.n_partitions + self.n_microbatches - 1

        for clock_idx in range(n_clock_cycles):
            start_partrition = max(clock_idx + 1 - self.n_microbatches, 0)
            end_partition = min(clock_idx + 1, self.n_partitions)

            tasks = []
            for partition_idx in range(start_partrition, end_partition):
                microbatch_idx = clock_idx - partition_idx
                tasks.append((microbatch_idx, partition_idx))

            yield tasks
