from typing import Dict

from pipegoose.nn.pipeline_parallel.pipeline_context import PipelineContext


def get_progresses_from_pipeline_context(pipeline_context: PipelineContext) -> Dict:
    schedules = pipeline_context.schedules
    progresses = {
        i: {(item.microbatch_idx, item.partition_idx): False for item in sublist} for i, sublist in enumerate(schedules)
    }
    return progresses
