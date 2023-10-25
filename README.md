# 🚧 PipeGoose: Training 🤗 `transformers` in Megatron-LM 3D parallelism and ZeRO-1 out of the box

[<img src="https://img.shields.io/badge/license-MIT-blue">](https://github.com/xrsrke/pipegoose) [![tests](https://github.com/xrsrke/pipegoose/actions/workflows/tests.yaml/badge.svg)](https://github.com/xrsrke/pipegoose/actions/workflows/tests.yaml) [<img src="https://img.shields.io/discord/767863440248143916?label=discord">](https://discord.gg/s9ZS9VXZ3p) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [<img alt="Codecov" src="https://img.shields.io/codecov/c/github/xrsrke/pipegoose">](https://app.codecov.io/gh/xrsrke/pipegoose) [![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

![pipeline](3d-parallelism.png)

<!-- [![docs](https://img.shields.io/github/deployments/Production?label=docs&logo=vercel)](https://docs.dev/) -->


⚠️ **The project is actively under development, and we're actively seeking collaborators. Come join us: [[discord link]](https://discord.gg/s9ZS9VXZ3p) [[roadmap]](https://github.com/users/xrsrke/projects/5) [[good first issue]](https://github.com/xrsrke/pipegoose/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22)**

⚠️ **The APIs is still a work in progress and could change at any time. None of the public APIs are set in stone until we hit version 0.6.9.**

⚠️ **Currently only parallelize `bloom-560m` is supported. Support for hybrid 3D parallelism and distributed optimizer for 🤗 `transformers` will be available in the upcoming weeks (it's basically done, but it doesn't support 🤗 `transformers` yet)**

⚠️ **This library is underperforming when compared to Megatron-LM and DeepSpeed (and not even achieving reasonable performance). We're actively pushing it to reach 180 TFLOPs and go beyond Megatron-LM.**

```diff
from torch.utils.data import DataLoader
+ from torch.utils.data.distributed import DistributedSampler
from torch.optim import SGD
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

+ from pipegoose.distributed import ParallelContext, ParallelMode
+ from pipegoose.nn import DataParallel, TensorParallel

model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m")
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
tokenizer.pad_token = tokenizer.eos_token

BATCH_SIZE = 4
+ DATA_PARALLEL_SIZE = 2
+ parallel_context = ParallelContext.from_torch(
+    tensor_parallel_size=2,
+    data_parallel_size=2,
+    pipeline_parallel_size=1
+ )
+ model = TensorParallel(model, parallel_context).parallelize()
+ model = DataParallel(model, parallel_context).parallelize()
model.to("cuda")
+ device = next(model.parameters()).device

optim = SGD(model.parameters(), lr=1e-3)

dataset = load_dataset("imdb", split="train")
+ dp_rank = parallel_context.get_local_rank(ParallelMode.DATA)
+ sampler = DistributedSampler(dataset, num_replicas=DATA_PARALLEL_SIZE, rank=dp_rank, seed=42)
+ dataloader = DataLoader(dataset, batch_size=BATCH_SIZE // DATA_PARALLEL_SIZE, shuffle=False, sampler=sampler)

for epoch in range(100):
+    sampler.set_epoch(epoch)

    for batch in dataloader:
        inputs = tokenizer(batch["text"], padding=True, truncation=True, max_length=1024, return_tensors="pt")
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
        labels = inputs["input_ids"]

        outputs = model(**inputs, labels=labels)

        optim.zero_grad()
        outputs.loss.backward()
        optim.step()
```

**Installation and try it out**

You can install the package through the following command:

```bash
pip install pipegoose
```

And try out a hybrid tensor and data parallelism training script (You must have at least 4 GPUs in order to try hybrid 2D parallelism).

```bash
cd pipegoose/examples
torchrun --standalone --nnodes=1 --nproc-per-node=4 hybrid_parallelism.py
```

We did a small scale correctness test by comparing the training losses between a paralleized transformer and one kept by default, starting at identical checkpoints and training data. We will conduct rigorous large scale convergence and weak scaling law benchmarks against Megatron and DeepSpeed in the near future if we manage to make it.
- Data Parallelism [[link]](https://wandb.ai/xariusdrake/pipegoose/runs/smjfnm9g)
- Tensor Parallelism [[link]](https://wandb.ai/xariusdrake/pipegoose/runs/iz17f50n)
- Hybrid 2D Parallelism (TP+DP) [[link]](https://wandb.ai/xariusdrake/pipegoose/runs/us31p3q1)

**Features**
- Megatron-style 3D parallelism
- Sequence parallelism and Mixture of Experts that work in 3D parallelism
- ZeRO-1: Distributed Optimizer
- Highly optimized CUDA kernels port from Megatron-LM, DeepSpeed
- ...

**Appreciation**

- Big thanks to 🤗 [Hugging Face](https://huggingface.co/) for sponsoring this project with 8x A100 GPUs for testing! And [Zach Schrier](https://twitter.com/zach_schrier) for monthly twitch donations

- The library's APIs are inspired by [OSLO](https://github.com/EleutherAI/oslo)'s and [ColossalAI](https://github.com/hpcaitech/ColossalAI)'s APIs.
