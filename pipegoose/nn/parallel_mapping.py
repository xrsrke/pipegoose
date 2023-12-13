from typing import Optional, Tuple, cast


class ParallelInfo:
    def __init__(self, module_name: Tuple[str], **kwargs):
        self.module_name = module_name
        self.kwargs = kwargs


class ParallelMapping:

    def __init__(self, model):
        traced = symbolic_trace(model, input_names=["input_ids", "attention_mask"])
        self.node_list = list(traced.graph.nodes)
        self.model = model

    def extract_module_from_node(self, node):
        # Split the target string into components
        target_path = node.target.split('.') if isinstance(node.target, str) else None

        # Traverse the model hierarchy based on the target path
        current_module = self.model
        try:
            for attr in target_path:
                current_module = getattr(current_module, attr)
        except AttributeError:
            return None

        return current_module

    def extract_node_target_from_module(self, submodule, prefix=''):
        for name, module in model.named_children():
            if module is submodule:
                return f'{prefix}{name}' if prefix else name
            else:
                submodule_path = find_submodule_target(module, submodule, prefix=f'{prefix}{name}.')
                if submodule_path:
                    return submodule_path
        return None

    @staticmethod
    def _extract_module_name(module_name: str) -> str:
        if "." in module_name:
            # NOTE: transformer.h.0.self_attention.dense -> self_attention.dense
            SEPARATOR = "."
            sections = module_name.split(SEPARATOR)
            return SEPARATOR.join(sections[-2:])
        return module_name

    @staticmethod
    def _search(module_name: str) -> Optional[ParallelInfo]:
        """
        Search for module_name in mappings.
        """
        module_name = self.module_name
        for child_class in ParallelMapping.__subclasses__():
            if hasattr(child_class, "__MAPPING__"):
                for items in child_class.__MAPPING__.values():
                    for item in items:
                        item = cast(ParallelInfo, item)
                        if any(module_name in mapping_name for mapping_name in item.module_name):
                            return item
                # NOTE: only search the first subclass of the current instance
                break

        return None
    def is_column_parallel(self, node_target) -> bool:
        """Returns True if the module is the first linear layer in an MLP layer,
        or if the module is a query, key, value linear,
        or a fused qkv linear of an attention layer, or an input embedding"""
        """Returns True iff the module is the first linear layer in an MLP layer, 
        or if the module is a query, key, value linear, 
        or a fused qkv linear of an attention layer, or an input embedding."""

        if not isinstance(node_target, str):
            return False

        # Check if the node is the first linear layer in an MLP layer
        if node_target.endswith('mlp.dense_h_to_4h'):
            return True

        # Check if the node is a fused QKV linear layer or the output projection of an attention layer
        if 'self_attention.query_key_value' in node_target:
            return True


        # Check if the node is part of the embedding layer
        if 'word_embeddings' in node_target:
            return True

        return False
    def is_row_parallel(self, node_target) -> bool:
        """Check if the module is the second linear layer in an MLP layer,
        or the output projection of an attention layer."""
        if not isinstance(node_target, str):
            return False

        # Check if the node is the second linear layer in an MLP layer
        if node_target.endswith('mlp.dense_4h_to_h'):
            return True

        # Check if the node is the output projection of an attention layer
        if node_target.endswith('self_attention.dense'):
            return True

        return False

    def is_lm_head(self, node_target) -> bool:
        """Returns True iff the module is language model head."""
        return isinstance(node_target, str) and 'lm_head' in node_target

    def is_text_embedding(self, node_target) -> bool:
        """Returns True iff the module is a text embedding module."""
        return isinstance(node_target, str) and 'embeddings' in node_target


if __name__ == "__main__":
    # test
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils.fx import symbolic_trace

    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
    model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m")
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

    pm = ParallelMapping(model)
    row_parallels = []
    column_parallels = []
    lm_heads = []
    text_embeddings = []
    for module_name, module in pm.model.named_modules():
        node_target = pm.extract_node_target_from_module(module)
        if node_target is None:
            continue
        if pm.is_row_parallel(node_target):
            row_parallels.append(module_name)
        if pm.is_column_parallel(node_target):
            column_parallels.append(module)
        if pm.is_lm_head(node_target):
            lm_heads.append(module)
        if pm.is_text_embedding(node_target):
            text_embeddings.append(module)

    assert len(row_parallels) == 48
    assert len(column_parallels) == 50
    assert len(lm_heads) == 1
    assert len(text_embeddings) == 2












