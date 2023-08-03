from typing import List


def map_rank_to_group(rank: int, groups: List[int]) -> List[int]:
    if len(groups) == 1:
        return groups
    else:
        rank_to_group = {r: g for g in groups for r in g}
        return rank_to_group[rank]
