from abc import ABC, abstractclassmethod
from enum import Enum, auto
from typing import List
from torch import nn
import torch
from typing import Dict
from collections import defaultdict

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.distributed.parallel_mode import ParallelMode
from transformers.utils.fx import symbolic_trace


class PartitionPolicy(Enum):
    UNIFORM = auto()


class BasePartitioner(ABC):
    """Base class for partitioning a model into multiple partitions."""

    @abstractclassmethod
    def split(self) -> List[nn.Module]:
        raise NotImplementedError


class UniformPartitioner(BasePartitioner):
    def __init__(self, model: nn.Module, parallel_context: ParallelContext):
        self.model = model
        self.parallel_context = parallel_context

    def _split_nodes(
        self, traced_graph_module: torch.fx.GraphModule, shard_count: int = 3
    ) -> Dict:
        """Utility used to trace a graph and identify shard cutpoints."""

        param_count: Dict[str, int] = {}

        # Find the total number of params in the model and
        # the number of params per shard we are aiming for.

        # Note: we need to iterate over named_parameters AND named_modules because
        # sometimes the parameters of a module is split into weight and bia in the
        # traced graph and sometimes not.
        # Example:
        # The embedding in traced graph is called transformer_wte. The naming as parameter
        # is transformer.wte.weight while as a module it is transformer.wte
        #
        # The projection inside the attention layer is split into weight and bias
        # in the traced graph while in the module we only see the projection as a whole module.

        for name, param in traced_graph_module.named_parameters():
            name = name.replace(".", "_")
            param_count[name] = param.numel()

        total_param_count = 0
        for name, module in traced_graph_module.named_modules():
            if len(name) > 0 and name.count(".") == 0:
                # also note that the parameters of the lm_head for some models (e.g. GPT2) are not
                # considered in named_parameters(). therefore, we must count the parameters using
                # named_modules.
                # we recursively go deeper into the modules, we cannot naively count the parameters of each module,
                # because we then would count the same parameter multiple times. hence, we only count the
                # parameters of the top-level modules.
                total_param_count += sum([x.numel() for x in module.parameters()])

            name = name.replace(".", "_")
            param_count[name] = sum([x.numel() for x in module.parameters()])

        per_shard_param = total_param_count // shard_count

        node_name_to_shard_id: Dict[str, int] = {}
        shard_id = 0
        shard_id_to_param_count = [0 for _ in range(shard_count)]

        output_from_shard = {}

        for node in traced_graph_module.graph.nodes:
            if node.op == "output":
                break

            if node.op in ("call_module", "get_attr"):
                # call_module and get_attr are the two operations which involve accessing parameters
                current_param_count = param_count.get(node.name, 0)
                if (
                    shard_id_to_param_count[shard_id] + current_param_count
                ) >= per_shard_param and (shard_id + 1) < shard_count:
                    shard_id += 1

                shard_id_to_param_count[shard_id] += current_param_count

            # we need to collect the nodes from the previous shards which are needed in the
            # current shard because we need to propagate them until the current shard
            if hasattr(node, "args"):
                for arg in node.args:
                    if not hasattr(arg, "name"):
                        continue

                    arg_shard_id = node_name_to_shard_id.get(arg.name, shard_id)
                    if arg_shard_id < shard_id:
                        # propagate the input from arg_shard_id until shard_id
                        for idx in range(arg_shard_id, shard_id):
                            # note that we use the dict as an ordered set data structure
                            output_from_shard.setdefault(idx, dict())[arg.name] = None

            node_name_to_shard_id[node.name] = shard_id

        return node_name_to_shard_id, output_from_shard

    def split(self, input_names: List[str]) -> List[nn.Module]:
        n_partitions = self.parallel_context.pipeline_parallel_size
        model = self.model
        module_list: List[torch.fx.GraphModule] = []
        num_graphs = 0
        new_graph = torch.fx.Graph()  # type: ignore

        symbolic_traced_module = symbolic_trace(model, input_names=input_names)

        node_name_to_shard_id, output_from_shard = self._split_nodes(
            symbolic_traced_module, n_partitions
        )

        nodes_per_shard = defaultdict(dict)

        prev_shard_id = 1000
        prev_node = None
        for node in symbolic_traced_module.graph.nodes:
            # If the current node is in the next shard, we insert an output node.
            # A new graph is created and a placeholder is added for the next shard.
            if (
                node.name in node_name_to_shard_id
                and prev_shard_id < node_name_to_shard_id[node.name]
            ):
                assert prev_node, "prev_node cannot be None"

                # generate output node for the past graph/shard
                with new_graph.inserting_after(prev_node):
                    outputs = output_from_shard[prev_shard_id]
                    graph_output_names = [prev_node.name] + [
                        i for i in outputs if i != prev_node.name
                    ]

                    if isinstance(
                        nodes_per_shard[prev_shard_id][prev_node.name], tuple
                    ):
                        graph_outputs = nodes_per_shard[prev_shard_id][
                            prev_node.name
                        ] + tuple(
                            [
                                nodes_per_shard[prev_shard_id][i]
                                for i in outputs
                                if i != prev_node.name
                            ]
                        )
                    else:
                        graph_outputs = tuple(
                            [nodes_per_shard[prev_shard_id][prev_node.name]]
                            + [
                                nodes_per_shard[prev_shard_id][i]
                                for i in outputs
                                if i != prev_node.name
                            ]
                        )
                    new_graph.create_node(
                        op="output", target="output", args=(graph_outputs,)
                    )

                # generate new graph/shard and its input nodes (i.e., the output from the previous graph/shard)
                num_graphs += 1
                module_list.append(torch.fx.GraphModule(model, new_graph))
                new_graph = torch.fx.Graph()
                # generate placeholder nodes in the new graph/shard which matches the output nodes of the previous graph/shard
                for new_graph_input_name in graph_output_names:
                    graph_input_node = new_graph.create_node(
                        "placeholder", new_graph_input_name
                    )
                    nodes_per_shard[node_name_to_shard_id[node.name]][
                        new_graph_input_name
                    ] = graph_input_node

            if node.op in [
                "placeholder",
                "get_attr",
                "call_function",
                "call_method",
                "call_module",
            ]:
                # Copy the nodes from the existing graph to the new graph.
                current_shard_id = node_name_to_shard_id[node.name]
                new_node = new_graph.node_copy(
                    node, lambda x: nodes_per_shard[current_shard_id][x.name]
                )
                nodes_per_shard[current_shard_id][node.name] = new_node
            elif node.op == "output":
                # If this is the last node, we should add an output
                # node and add the last graph to the list.
                assert prev_node, "prev_node cannot be None"

                with new_graph.inserting_after(prev_node):
                    new_graph.output(nodes_per_shard[prev_shard_id][prev_node.name])
                module_list.append(torch.fx.GraphModule(model, new_graph))
                break
            prev_node = new_node
            prev_shard_id = node_name_to_shard_id[node.name]

        return module_list


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
