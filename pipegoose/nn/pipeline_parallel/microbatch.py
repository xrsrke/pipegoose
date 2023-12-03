from typing import List, TypedDict

import torch


class ModelInputs(TypedDict):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


def split(inputs: ModelInputs, n_microbatches: int) -> List[ModelInputs]:
    assert n_microbatches > 0, f"n_microbatches must be greater than 0, got {n_microbatches}"
    assert "input_ids" in inputs, f"inputs must have 'input_ids' key, got {inputs.keys()}"
    assert "attention_mask" in inputs, f"inputs must have 'attention_mask' key, got {inputs.keys()}"
    assert (
        inputs["input_ids"].size(0) % n_microbatches == 0
    ), f"The batch size must be divisible by n_microbatches, got {inputs['input_ids'].size(0)} and {n_microbatches}"

    input_ids_microbatches = torch.split(inputs["input_ids"], n_microbatches)
    attention_mask_microbatches = torch.split(inputs["attention_mask"], n_microbatches)

    microbatches = []
    for input_ids, attention_mask in zip(input_ids_microbatches, attention_mask_microbatches):
        microbatches.append(ModelInputs(input_ids=input_ids, attention_mask=attention_mask))

    return microbatches
