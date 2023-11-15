import torch
import inspect
import copy
from abc import ABC, abstractclassmethod
from enum import Enum, auto
from typing import List
from torch import nn
from transformers.utils.fx import (
    HFTracer,
    _generate_random_int,
    transform_to_dynamic_input_,
    _generate_supported_model_classes,
    _SUPPORTED_MODELS,
    _SUPPORTED_MODELS_FOR_DYNAMIC_AXES,
    _wrap_method_for_model_tracing,
    _reset_tensor_methods,
)
from transformers.models.auto import get_values
from packaging import version
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
    MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING,
    MODEL_FOR_PRETRAINING_MAPPING,
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
    MODEL_MAPPING,
    GPT2DoubleHeadsModel,
    PretrainedConfig,
    PreTrainedModel,
    logging,
)

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.distributed.parallel_mode import ParallelMode

_GLOBAL_ARGS = None


class PartitionPolicy(Enum):
    UNIFORM = auto()


class BasePartitioner(ABC):
    """Base class for partitioning a model into multiple partitions."""

    @abstractclassmethod
    def split(self) -> List[nn.Module]:
        raise NotImplementedError

    """_summary_
    hf_fx_compatibility(model)

    _description_
    Check if the model is compatible with the HFTracer

    """


def hf_fx_compatibility(model):
    added_model = tuple(_generate_supported_model_classes("vit"))
    transformers_fx_models = tuple(
        _SUPPORTED_MODELS + _SUPPORTED_MODELS_FOR_DYNAMIC_AXES + added_model
    )
    if isinstance(model, PreTrainedModel) and isinstance(model, transformers_fx_models):
        return True
    else:
        return False


def get_args():
    """Return arguments."""
    assert _GLOBAL_ARGS is not None, "{} is not initialized.".format("args")
    return _GLOBAL_ARGS


class MpTracer(HFTracer):
    def __init__(
        self,
        leaf_modules=(),
        manual_input_shape=None,
        trace_batch=None,
        batch_size=1,
        sequence_length=[128, 128],
        num_choices=-1,
    ):
        super().__init__(batch_size, sequence_length, num_choices)
        self.leaf_modules = leaf_modules
        if manual_input_shape is not None:
            self.encoder_shape = manual_input_shape

        self.trace_batch = trace_batch

    def is_manual_leaf_module(self, m):
        for i in self.leaf_modules:
            if isinstance(m, i):
                return True
        return False

    def is_leaf_module(self, m: torch.nn.Module, model_qualified_name: str) -> bool:
        return super().is_leaf_module(
            m, model_qualified_name
        ) or self.is_manual_leaf_module(m)

    def _generate_dummy_input(self, model, input_name):
        """Generates dummy input for model inference recording."""
        args = get_args()
        model_class = model.__class__
        device = model.device
        # device = 'cpu'
        inputs_dict = dict()
        if self.trace_batch is not None:
            return self.trace_batch

        if input_name in ["labels", "start_positions", "end_positions"]:
            batch_size = self.encoder_shape[0]
            if model_class in get_values(MODEL_FOR_MULTIPLE_CHOICE_MAPPING):
                inputs_dict["labels"] = torch.ones(
                    batch_size, dtype=torch.long, device=device
                )
            elif model_class in get_values(MODEL_FOR_QUESTION_ANSWERING_MAPPING):
                inputs_dict["start_positions"] = torch.zeros(
                    batch_size, dtype=torch.long, device=device
                )
                inputs_dict["end_positions"] = torch.zeros(
                    batch_size, dtype=torch.long, device=device
                )
            elif model_class in [
                *get_values(MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING),
                *get_values(MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING),
                *get_values(MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING),
            ]:
                inputs_dict["labels"] = torch.zeros(
                    batch_size, dtype=torch.long, device=device
                )
            elif model_class in [
                *get_values(MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING),
                *get_values(MODEL_FOR_CAUSAL_LM_MAPPING),
                *get_values(MODEL_FOR_MASKED_LM_MAPPING),
                *get_values(MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING),
                GPT2DoubleHeadsModel,
            ]:
                inputs_dict["labels"] = torch.zeros(
                    self.decoder_shape, dtype=torch.long, device=device
                )
            elif model_class in get_values(MODEL_FOR_PRETRAINING_MAPPING):
                inputs_dict["labels"] = torch.zeros(
                    self.encoder_shape, dtype=torch.long, device=device
                )
            else:
                raise NotImplementedError(f"{model_class} not supported yet.")

        elif "mask" in input_name or "ids" in input_name:
            shape = (
                self.encoder_shape
                if "decoder" not in input_name
                else self.decoder_shape
            )
            inputs_dict[input_name] = torch.ones(shape, dtype=torch.long, device=device)
        elif "pixel_values" in input_name:
            shape = [
                self.encoder_shape[0],
                model.config.num_channels,
                model.config.image_size,
                model.config.image_size,
            ]
            inputs_dict[input_name] = torch.ones(
                shape, dtype=torch.float, device=device
            )
        else:
            shape = (
                self.encoder_shape
                if "decoder" not in input_name
                else self.decoder_shape
            )
            shape += [model.config.hidden_size]
            inputs_dict[input_name] = torch.ones(
                shape, dtype=torch.float, device=device
            )

        if args.fp16 or args.half_precision_backend == "apex":
            half_inputs_dict = {}
            for k, v in inputs_dict.items():
                half_inputs_dict[k] = v.half()
            inputs_dict = half_inputs_dict

        return inputs_dict

    def trace(self, root: PreTrainedModel, concrete_args=None, method_names=None):
        if concrete_args is None:
            concrete_args = {}

        sig = inspect.signature(root.forward)
        input_names = sig.parameters.keys() - concrete_args.keys()

        self.record(root, input_names, method_names=method_names)

        for method_name, cache_name in self.recorded_methods.items():
            _wrap_method_for_model_tracing(root, method_name, cache_name)

        graph = torch.fx.Tracer.trace(self, root, concrete_args=concrete_args)

        _reset_tensor_methods(self.original_methods)

        torch_version = version.parse(torch.__version__)
        if torch_version.minor <= 11:
            # torch version compatibility
            # https://github.com/huggingface/transformers/pull/17129
            # https://github.com/pytorch/pytorch/pull/59569
            for node in graph.nodes:
                if node.op == "placeholder":
                    # Removing default values for inputs as the forward pass will fail with them.
                    if node.target in input_names:
                        node.args = ()
                    # It is a concrete arg so it is not used and should be removed.
                    else:
                        graph.erase_node(node)
        return graph


def symbolic_trace(
    model,
    input_names=None,
    batch_size=1,
    sequence_length=(128, 128),
    num_choices=-1,
    extra_leaf_modules=(),
    trace_batch=None,
):
    """
    Performs symbolic tracing on the model.

    Args:
        model (:obj:`PretrainedModel`):
            The model to trace.
        input_names (:obj:`List[str]`, `optional`):
            The names of the inputs of the traced model. If unset, model.dummy_inputs().keys() are used instead.
        batch_size (:obj:`int`, `optional`, defaults to 1):
            The batch size of the traced model inputs.
        sequence_length (:obj:`int` or :obj:`List[int]]`):
            The sequence length of the traced model inputs. For sequence-to-sequence models with different sequence
            lengths between the encoder and the decoder inputs, this must be :obj:`[encoder_sequence_length,
            decoder_sequence_length]`.
        num_choices (:obj:`int`, `optional`, defaults to -1):
            The number of possible choices for a multiple choice task.

    Returns:
        :obj:`torch.fx.GraphModule`: A GraphModule constructed by recording operations seen while tracing the model.

    Example::

        from transformers.utils.fx import symbolic_trace
        traced_model = symbolic_trace(
            model,
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            batch_size=1,
            sequence_length=128,
        )
    """
    if input_names is None or input_names == []:
        input_names = model.dummy_inputs.keys()

    sig = inspect.signature(model.forward)
    concrete_args = {
        p.name: p.default for p in sig.parameters.values() if p.name not in input_names
    }
    # print(concrete_args)
    # Preparing HFTracer batch_size and sequence_lenght values for potential dynamic axes.
    use_dynamic_batch_size = batch_size <= 0
    if isinstance(sequence_length, (list, tuple)):
        use_dynamic_sequence_length = sequence_length[0] <= 0 or sequence_length[1] <= 0
    elif isinstance(sequence_length, int):
        use_dynamic_sequence_length = sequence_length <= 0
    else:
        use_dynamic_sequence_length = False

    if use_dynamic_batch_size or use_dynamic_sequence_length:
        forbidden_values = [
            model.config.num_attention_heads,
            model.config.hidden_size,
            model.config.hidden_size // model.config.num_attention_heads,
        ]
        if use_dynamic_batch_size:
            batch_size = _generate_random_int(forbidden_values=forbidden_values)
        forbidden_values.append(batch_size)
        if use_dynamic_sequence_length:
            encoder_sequence_length = _generate_random_int(
                forbidden_values=forbidden_values
            )
            forbidden_values.append(encoder_sequence_length)
            decoder_sequence_length = _generate_random_int(
                forbidden_values=forbidden_values
            )
            sequence_length = [encoder_sequence_length, decoder_sequence_length]

    if isinstance(extra_leaf_modules, list):
        extra_leaf_modules = tuple(extra_leaf_modules)
    elif isinstance(extra_leaf_modules, nn.Module):
        extra_leaf_modules = tuple([extra_leaf_modules])
    else:
        assert isinstance(extra_leaf_modules, tuple), "leaf_modules should be tuple"
    # Tracing.
    tracer = MpTracer(
        leaf_modules=default_leaf_modules + extra_leaf_modules,
        trace_batch=trace_batch,
        batch_size=batch_size,
        sequence_length=sequence_length,
        num_choices=num_choices,
    )
    with torch.no_grad():
        traced_graph = tracer.trace(model, concrete_args=concrete_args)
    traced = torch.fx.GraphModule(model, traced_graph)
    dummy_inputs = {}

    for name in input_names:
        dummy_inputs.update(tracer._generate_dummy_input(model, name))

    del traced_graph, tracer

    traced.config = copy.deepcopy(model.config)
    traced.num_choices = num_choices

    traced.use_dynamic_batch_size = use_dynamic_batch_size
    traced.use_dynamic_sequence_length = use_dynamic_sequence_length
    traced.static_batch_size = batch_size
    traced.static_sequence_length = sequence_length

    transform_to_dynamic_input_(traced)

    return traced, dummy_inputs


class UniformPartitioner(BasePartitioner):
    def __init__(self, module: nn.Module, parallel_context: ParallelContext):
        self.module = module
        self.parallel_context = parallel_context

    def split(self) -> List[nn.Module]:
        n_partitions = self.parallel_context.pipeline_parallel_size
        module = self.module
        partitions = []
        start = 0
        end = 0
        print("module")
        print(hf_fx_compatibility(module))

        def _flatten_model(model, parent_name=""):
            model_list = []
            for name, child_module in model.named_children():
                # Form the full name of the module
                full_name = f"{parent_name}.{name}" if parent_name else name
                if (
                    full_name == "transformer.h"
                ):  # Check if the module is the 'h' attribute
                    # If it's the 'h' ModuleList, append each of its blocks as a whole
                    for block in child_module:
                        model_list.append(block)
                elif len(list(child_module.children())) == 0:
                    # If it's a leaf node, append the module itself
                    model_list.append(child_module)
                else:
                    # Otherwise, continue flattening its children
                    model_list.extend(_flatten_model(child_module, full_name))
            return model_list

        prepared_model = _flatten_model(module)
        for p in prepared_model:
            print(type(p))
            print(p)

        return partitions


def _get_partitioner(policy: PartitionPolicy) -> BasePartitioner:
    """Return the corresponding partitioner based on the policy."""
    policy_to_partitioner = {
        PartitionPolicy.UNIFORM: UniformPartitioner,
    }

    return policy_to_partitioner[policy]


def get_model_partition(
    module: nn.Module, policy: PartitionPolicy, parallel_context: ParallelContext
) -> nn.Module:
    """Get the corresponding partition of the current process."""
    partitioner = _get_partitioner(policy)
    partitions = partitioner(module, parallel_context).split()

    # TODO: remove this, use pipeline_context instead
    def _get_partition_idx():
        rank = parallel_context.get_local_rank(ParallelMode.PIPELINE)
        rank_per_group = len(parallel_context.get_ranks_in_group(ParallelMode.PIPELINE))
        partition_idx = rank // rank_per_group
        return partition_idx

    partition_idx = _get_partition_idx()
    return partitions[partition_idx]
