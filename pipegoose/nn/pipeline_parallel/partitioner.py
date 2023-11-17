from abc import ABC, abstractclassmethod
from enum import Enum, auto
from typing import List
from torch import nn
import torch
from typing import Dict
from torch.fx.node import Node
from typing import Set

from transformers.models.auto import get_values
from packaging import version

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


def _get_count(param_count: Dict, node_name: str) -> int:
    """Identify different mutations of a given node name."""
    # TODO(anj): This is not very stable since it is possible that the name
    # may not be in the same format. Is there another way to identify nodes
    # in a graph?
    if node_name in param_count:
        return param_count[node_name]
    elif node_name.split("_")[0] in param_count:
        return param_count[node_name.split("_")[0]]
    else:
        raise RuntimeError(
            f"Unable to find match between param {param_count} and node {node_name}"
        )


def _create_shard_to_param_count(
    param_count: Dict, node_name_to_shard_id: Dict
) -> Dict:
    """Utility to create a map from shard id to param count using existing state."""

    shard_to_param_count: Dict[int, int] = {}
    for node_name in node_name_to_shard_id.keys():
        try:
            count = _get_count(param_count, node_name)
        except RuntimeError:
            continue
        if node_name_to_shard_id[node_name] in shard_to_param_count:
            shard_to_param_count[node_name_to_shard_id[node_name]] += count
        else:
            shard_to_param_count[node_name_to_shard_id[node_name]] = count
    return shard_to_param_count


class UniformPartitioner(BasePartitioner):
    def __init__(self, module: nn.Module, parallel_context: ParallelContext):
        self.module = module
        self.parallel_context = parallel_context

    def _split_nodes(
        self, traced_graph_module: torch.fx.GraphModule, shard_count: int = 3
    ) -> Dict:
        """Utility used to trace a graph and identify shard cutpoints."""

        node_name_to_shard_id: Dict[str, int] = {}
        shard_id = 0
        nodes_so_far = []
        param_count: Dict[str, int] = {}
        shard_to_param_count = {}

        # Find the total number of params in the model and
        # the number of params per shard we are aiming for.
        for name, module in traced_graph_module.named_modules():
            name = name.replace(".", "_")
            param_count[name] = sum([x.numel() for x in module.parameters()])
        print(f"Total number of params are {param_count['']}")
        per_shard_param = param_count[""] // shard_count
        print(f"Per shard param count {per_shard_param}")

        for node in traced_graph_module.graph.nodes:
            if node.op == "placeholder":
                node_name_to_shard_id[node.name] = shard_id
                nodes_so_far.append(node.name)
            elif node.op in ["get_attr", "call_function", "call_method", "call_module"]:
                min_shard_id = shard_id
                min_node_name = ""
                # For each of the args of a given node, find the arg that is not the
                # last node we traversed. This is to help us find skip connections
                # across shards.
                for arg in node.args:
                    # If the node has args that are inputs to the forward function, they
                    # may not have explicit names.
                    if not hasattr(arg, "name"):
                        continue

                    if (
                        arg.name in node_name_to_shard_id
                        and arg.name != nodes_so_far[-1]
                    ):
                        if node_name_to_shard_id[arg.name] < min_shard_id:
                            min_shard_id = node_name_to_shard_id[arg.name]
                            min_node_name = arg.name

                # If there is an input that is not from the previous shard,
                # we collapse all the shards in between to be part of 1 shard.
                # and update the param count per shard accordingly.
                if min_shard_id < shard_id:
                    for node_name in reversed(nodes_so_far):
                        node_name_to_shard_id[node_name] = min_shard_id
                        if node_name == min_node_name:
                            break
                    shard_id = min_shard_id
                    # TODO(anj-s): Find a way to raise an error early if this can cause OOM errors.
                    shard_to_param_count = _create_shard_to_param_count(
                        param_count, node_name_to_shard_id
                    )

                # Update state that is tracking node -> shard id and shard id -> param count.
                node_name_to_shard_id[node.name] = shard_id
                nodes_so_far.append(node.name)
                # TODO(anj): This could just be an update, we don't need to recreate the map.
                shard_to_param_count = _create_shard_to_param_count(
                    param_count, node_name_to_shard_id
                )
                # If we have gone over the number of params per shard count that we want to
                # achieve, we should add a new shard.
                # The shard_id may not have been updated in the map if we are at a node that does not
                # have params.
                if (
                    shard_id in shard_to_param_count
                    and shard_to_param_count[shard_id] > per_shard_param
                ):
                    shard_id += 1
            elif node.op == "output":
                break
        return node_name_to_shard_id

    def split(self, input_names) -> List[nn.Module]:
        n_partitions = self.parallel_context.pipeline_parallel_size
        model = self.module
        leaf_modules = set()
        module_list: List[torch.fx.GraphModule] = []
        num_graphs = 0
        new_graph = torch.fx.Graph()  # type: ignore
        env: Dict[str, Node] = {}
        new_input_node = None

        symbolic_traced_module = symbolic_trace(self.module)

        prev_shard_id = 1000
        prev_shard_node = None

        node_name_to_shard_id = self._split_nodes(symbolic_traced_module, n_partitions)

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

                with new_graph.inserting_after(prev_node):
                    new_graph.output(env[prev_node.name])
                num_graphs += 1
                module_list.append(torch.fx.GraphModule(model, new_graph))
                new_graph = torch.fx.Graph()
                node_name = "placeholder" + str(num_graphs)
                pl_node = new_graph.create_node("placeholder", node_name)
                env[node_name] = pl_node
                new_input_node = pl_node

            if new_input_node is not None:
                # Account for a placeholder in the new graph.
                node.args = (new_input_node,)
                new_input_node = None
            if node.op in [
                "placeholder",
                "get_attr",
                "call_function",
                "call_method",
                "call_module",
            ]:
                # Copy the nodes from the existing graph to the new graph.
                new_node = new_graph.node_copy(node, lambda x: env[x.name])
                env[node.name] = new_node
            elif node.op == "output":
                # If this is the last node, we should add an output
                # node and add the last graph to the list.
                assert prev_node, "prev_node cannot be None"

                with new_graph.inserting_after(prev_node):
                    new_graph.output(env[prev_node.name])
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
