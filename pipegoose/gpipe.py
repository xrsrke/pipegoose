from typing import List, Optional

from torch import nn


class GPipe:
    def __init__(
        self,
        module: nn.Sequential,
        balances: Optional[List[int]],
        n_partritions: int,
    ) -> None:
        self.n_partritions = n_partritions
