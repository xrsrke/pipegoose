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


class UniformPartitioner(BasePartitioner):
    # def __init__(self, module: nn.Module, parallel_context: ParallelContext):
    def __init__(self, module: nn.Module, n_partitions: int):
        self.module = module
        self.n_partitions = n_partitions

    def _split_nodes(
        self, traced_graph_module: torch.fx.GraphModule, shard_count: int = 3
    ) -> Dict:
        """Utility used to trace a graph and identify shard cutpoints."""

        nodes_so_far = []
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
            print(f"{name} => {param.numel()}")
            name = name.replace(".", "_")
            param_count[name] = param.numel()
        
        total_param_count = 0
        for name, module in traced_graph_module.named_modules():
            print(name)
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

        print(f"Total number of params are {total_param_count}")
        per_shard_param = total_param_count // shard_count
        print(f"Per shard param count {per_shard_param}")


        node_name_to_shard_id: Dict[str, int] = {}
        shard_id = 0
        shard_id_to_param_count = [0 for _ in range(shard_count)]
        for node in traced_graph_module.graph.nodes:
            if node.op == "output":
                break

            if node.op in ("call_module", "get_attr"):
                # call_module and get_attr are the two operations which involve accessing parameters
                print(f"\n{node.name} = {node.op} target={node.target} args={node.args} ===> {shard_id}")
                print(f"Args and their shards: {[(arg.name, node_name_to_shard_id[arg.name]) for arg in node.args if hasattr(arg, 'name')]}")

                current_param_count = param_count.get(node.name, 0)

                # if shard_id_to_param_count[shard_id] >= per_shard_param and (shard_id + 1) < shard_count:
                print(shard_id_to_param_count[shard_id] >= per_shard_param, shard_id_to_param_count[shard_id], per_shard_param)
                if (shard_id_to_param_count[shard_id] + current_param_count) >= per_shard_param and (shard_id + 1) < shard_count:
                    shard_id += 1

                shard_id_to_param_count[shard_id] += current_param_count
                print(f"shard_id_to_param_count = {shard_id_to_param_count}")
            
            node_name_to_shard_id[node.name] = shard_id

        return node_name_to_shard_id

    def split(self, input_names) -> List[nn.Module]:
        # n_partitions = self.parallel_context.pipeline_parallel_size
        n_partitions = self.n_partitions # FIXME:
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
