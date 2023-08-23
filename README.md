# 🚧 PipeGoose: Train 🤗 `transformers` in 3D parallelism - WIP

![pipeline](3d-parallelism.png)

Honk honk honk! This project is actively under development. Check out my learning progress [here](https://twitter.com/xariusrke/status/1667999818554413057).


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

- Supports training `transformers` model.
- Supports ZeRO-1 and ZeRO-Offload.
- Implements parallel compute and data transfer using separate CUDA streams.
- Gradient checkpointing will be implemented by enforcing virtual dependency in the backpropagation graph, ensuring that the activation for gradient checkpoint will be recomputed just in time for each (micro-batch, partition).
- Custom algorithms for model partitioning with two default partitioning models based on elapsed time and GPU memory consumption per layer.
- Potential support includes:
    - Callbacks within the pipeline: `Callback(function, microbatch_idx, partition_idx)` for before and after the forward, backward, and recompute steps (for gradient checkpointing).
    - Mixed precision training.
    - Elastic training
    - Fault-tolerance

**Appreciation**

Big thanks to 🤗 [Hugging Face](https://huggingface.co/) for sponsoring this project with 8x A100 GPUs for testing! And [Zach Schrier](https://twitter.com/zach_schrier) for monthly twitch donations
