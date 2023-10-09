# 🚧 PipeGoose: Training any 🤗 `transformers` in Megatron-LM 3D parallelism out of the box

[<img src="https://img.shields.io/badge/license-MIT-blue">](https://github.com/xrsrke/pipegoose) [![tests](https://github.com/xrsrke/pipegoose/actions/workflows/tests.yaml/badge.svg)](https://github.com/xrsrke/pipegoose/actions/workflows/tests.yaml) [<img src="https://img.shields.io/discord/767863440248143916?label=discord">](https://discord.gg/s9ZS9VXZ3p) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [<img alt="Codecov" src="https://img.shields.io/codecov/c/github/xrsrke/pipegoose">](https://app.codecov.io/gh/xrsrke/pipegoose) [![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/) [![Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40xariusrke)](https://twitter.com/xariusrke)

![pipeline](3d-parallelism.png)

<!-- [![docs](https://img.shields.io/github/deployments/Production?label=docs&logo=vercel)](https://docs.dev/) -->
<!-- [<img src="https://img.shields.io/youtube/channel/views/UCDdC6BIFRI0jvcwuhi3aI6w?style=social">](https://www.youtube.com/channel/UCDdC6BIFRI0jvcwuhi3aI6w/videos) -->
<!-- [<img src="https://img.shields.io/badge/%F0%9F%A4%97%20Models-Huggingface-F8D521">](https://huggingface.co) -->
<!-- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/blob/master/docs/get-started/CleanRL_Huggingface_Integration_Demo.ipynb) -->


Honk honk honk! This project is actively under development. Check out my learning progress [here](https://twitter.com/xariusrke/status/1667999818554413057).

⚠️ **The project is actively under development and not ready for use.**

⚠️ **The APIs is still a work in progress and could change at any time. None of the public APIs are set in stone until we hit version 0.6.9.**

```diff
import torch
import torch.nn.functional as F
from transformer import AutoModel, AutoTokenizer
from datasets import load_dataset
+ from pipegoose import DataParallel, TensorParallel, PipelineParalell, ParallelContext
+ from pipegoose.optim import DistributedOptimizer

model = AutoModel.from_pretrained("bloom")
tokenizer = AutoTokenizer.from_pretrained("bloom")

- device = "cuda"
- model = model.to(device)
+ parallel_context = ParallelContext(
+    tensor_parallel_size=2,
+    data_parallel_size=2,
+    pipeline_parallel_size=2
+ )
+ model = DataParallel(model, parallel_context).parallelize()
+ model = TensorParallel(model, parallel_context).parallelize()
+ model = PipelineParallel(model, parallel_context).parallelize()

optimizer = torch.optim.Adam(model.parameters())
+ optimizer = DistributedOptimizer(optimizer, parallel_context)

dataset = load_dataset('goose')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=42)

for epoch in range(69):
    for inputs, targets in dataloader:
-         inputs = inputs.to(device)
-         targets = targets.to(device)

        output = model(inputs)
        loss = F.cross_entropy(output, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```


**Implementation Details**

- Supports training `transformers` model in Megatron 3D parallelism and ZeRO-1 (write from scratch).
- Implements parallel compute and data transfer using separate CUDA streams.
- Gradient checkpointing will be implemented by enforcing virtual dependency in the backpropagation graph, ensuring that the activation for gradient checkpoint will be recomputed just in time for each (micro-batch, partition).
- Custom algorithms for model partitioning with two default partitioning models based on elapsed time and GPU memory consumption per layer.
- Potential support includes:
    - Callbacks within the pipeline: `Callback(function, microbatch_idx, partition_idx)` for before and after the forward, backward, and recompute steps (for gradient checkpointing).
    - Mixed precision training.

**Appreciation**

Big thanks to 🤗 [Hugging Face](https://huggingface.co/) for sponsoring this project with 8x A100 GPUs for testing! And [Zach Schrier](https://twitter.com/zach_schrier) for monthly twitch donations
