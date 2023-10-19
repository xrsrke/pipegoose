from typing import List, TypedDict

import torch


class ModelInputs(TypedDict):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


def split(inputs: ModelInputs, n_microbatches: int) -> List[ModelInputs]:
    assert n_microbatches > 0, f"n_microbatches must be greater than 0, got {n_microbatches}"

    input_ids_microbatches = torch.split(inputs["input_ids"], 2)
    attention_mask_microbatches = torch.split(inputs["attention_mask"], 2)

    microbatches = []
    for input_ids, attention_mask in zip(input_ids_microbatches, attention_mask_microbatches):
        microbatches.append(ModelInputs(input_ids=input_ids, attention_mask=attention_mask))

    return microbatches
