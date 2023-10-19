from typing import Dict


class Callback:
    order = 0

    def after_new_clock_cycle(self, progress: Dict, clock_idx: int):
        raise NotImplementedError
