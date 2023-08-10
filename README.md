# 🚧 PipeGoose: Pipeline Parallelism for transformers model - WIP

![pipeline](parallelism-deepspeed-3d.png)

Honk honk honk! This project is actively under development. Check out my learning progress [here](https://twitter.com/xariusrke/status/1667999818554413057).


``` python
from transformer import AutoModel, AutoTokenizer
from pipegoose import Pipeline

model = AutoModel.from_pretrained("bloom")
tokenizer = AutoTokenizer.from_pretrained("bloom")

pipeline = Pipeline(model, tokenizer, partrition=partrition_func)

pipeline.fit(dataloader, n_microbatches=16)
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