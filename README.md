# 🚧 PipeGoose: Training any 🤗 `transformers` in Megatron-LM 3D parallelism out of the box (write from scratch) - WIP

[<img src="https://img.shields.io/badge/license-MIT-blue">](https://github.com/xrsrke/pipegoose) [![tests](https://github.com/vwxyzjn/cleanrl/actions/workflows/tests.yaml/badge.svg)](https://github.com/xrsrke/pipegoose/actions/workflows/tests.yaml) [<img src="https://img.shields.io/discord/767863440248143916?label=discord">](https://discord.gg/s9ZS9VXZ3p) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/) 


![pipeline](3d-parallelism.png)

<!-- [![docs](https://img.shields.io/github/deployments/Production?label=docs&logo=vercel)](https://docs.dev/) -->
<!-- [<img src="https://img.shields.io/youtube/channel/views/UCDdC6BIFRI0jvcwuhi3aI6w?style=social">](https://www.youtube.com/channel/UCDdC6BIFRI0jvcwuhi3aI6w/videos) -->
<!-- [<img src="https://img.shields.io/badge/%F0%9F%A4%97%20Models-Huggingface-F8D521">](https://huggingface.co) -->
<!-- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/blob/master/docs/get-started/CleanRL_Huggingface_Integration_Demo.ipynb) -->

Honk honk honk! This project is actively under development. Check out my learning progress [here](https://twitter.com/xariusrke/status/1667999818554413057).

⚠️ **The APIs is still a work in progress and could change at any time. None of the public APIs are set in stone until we hit version 0.6.9.**


``` python
from transformer import AutoModel, AutoTokenizer
from pipegoose import Pipeline, ParallelContext

model = AutoModel.from_pretrained("bloom")
tokenizer = AutoTokenizer.from_pretrained("bloom")

parallel_context = ParallelContext(
    tensor_parallel_size=2,
    pipeline_parallel_size=2,
    data_parallel_size=2
)

pipeline = Pipeline(model, tokenizer, parallel_context)

pipeline.fit(dataloader)
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
