import math
from abc import ABC, abstractmethod
from typing import Tuple, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchtyping import TensorType


class RouterExplorationNoisePolicy(ABC):
    @abstractmethod
    def sample_like(self, input: TensorType) -> TensorType:
        pass


class SwitchNoisePolicy(RouterExplorationNoisePolicy):
    """
    Noise sampling as described in
    Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity
    by Fedus et al.
    """
    def __init__(self, eps: float=0.1):
        super().__init__()
        self.eps = eps

    def sample_like(self, input: TensorType) -> TensorType:
        # exploration with noise sampled from [1-eps, 1+eps] during training
        noise = torch.rand_like(input)  # between [0, 1)
        noise = noise * self.eps * 2  # between [0, 2*eps)
        noise += 1 - self.eps  # between [1-eps, 1+eps)
        return noise


class Router(ABC, nn.Module):
    pass


class _TopKRouter(Router):
    def __init__(
        self,
        noise_policy: RouterExplorationNoisePolicy,
        top_k: int,
        num_experts: int,
        d_model: int,
        expert_capacity: Optional[Tuple[float, float]] = None,
        alpha: float = 0.01,
        eps: float = 0.1,
        aux_loss_weight: float = 1.0,
        z_loss_weight: float = 1.0,
    ):
        super().__init__()
        self.noise_policy = noise_policy
        self.top_k = top_k
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity
        self.alpha = alpha
        self.eps = eps
        self.aux_loss_weight = aux_loss_weight
        self.z_loss_weight = z_loss_weight
        self.gate = nn.Linear(d_model, num_experts)

    def _aux_loss(
        self,
        router_prob: TensorType["batch_size*seq_len", "num_experts"],
        expert_mask: TensorType["batch_size*seq_len", "num_experts"],
    ) -> TensorType["1"]:
        """
        Auxiliary Load Balancing Loss as described in
        Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity
        by Fedus et al.
        """
        fraction_of_tokens_per_expert = expert_mask.mean(dim=0)
        fraction_of_prob_mass_per_expert = router_prob.mean(dim=0)

        inner_prod = fraction_of_tokens_per_expert @ fraction_of_prob_mass_per_expert
        aux_loss = self.alpha * self.num_experts * inner_prod

        return aux_loss

    def _z_loss(self, router_logits: TensorType["batch_size*seq_len", "num_experts"]) -> TensorType["1"]:
        """
        z-loss as described in
        ST-MoE: Designing Stable and Transferable Sparse Expert Models
        by Zoph et al.
        """
        return (torch.log(router_logits.exp().sum(dim=-1)) ** 2).mean()

    def _expert_capacity(self, total_tokens: int) -> int:
        c = self.expert_capacity[0 if self.training else 1]
        expert_capacity = math.ceil((total_tokens / self.num_experts) * c)
        return expert_capacity

    def forward(
        self, inputs: TensorType["batch_size", "seq_len", "d_model"]
    ) -> Tuple[
        TensorType["batch_size*seq_len", "num_experts"], TensorType["batch_size*seq_len", "num_experts"], TensorType["1"]
    ]:
        orig_dtype = inputs.dtype
        total_tokens = inputs.shape[0] * inputs.shape[1]

        # calculate router probability mass for each token
        router_logits = self.gate(inputs.float()).reshape(-1, self.num_experts)
        if self.training:
            # router exploration during training
            router_logits += self.noise_policy.sample_like(router_logits)

        router_prob = F.softmax(router_logits, dim=-1)

        # for each token, select the top-k experts to which we send the token
        topk_idxs = torch.topk(router_prob, self.top_k, dim=-1).indices

        # compute expert mask, set True at position (i,j) if token i is routed to expert j
        topk_expert_mask = torch.zeros_like(router_prob)
        topk_expert_mask = topk_expert_mask.scatter_(1, topk_idxs, True)

        # calculate router loss
        loss = self.aux_loss_weight * self._aux_loss(router_prob, topk_expert_mask) + self.z_loss_weight * self._z_loss(
            router_logits
        )

        if not self.expert_capacity:
            # we don't limit the capacity of the experts
            topk_weight = router_prob * topk_expert_mask
            topk_weight = topk_weight.to(orig_dtype)
            return topk_expert_mask, topk_weight, loss

        # limit the number of tokens per expert
        position_in_expert = torch.cumsum(topk_expert_mask, dim=0) * topk_expert_mask

        # filter out tokens which exceed the capacity
        expert_capacity = self._expert_capacity(total_tokens)
        capacity_limited_topk_expert_mask = topk_expert_mask * (position_in_expert < expert_capacity)

        # prune the router probabilities, only keep probabilities at position (i,j)
        # if token i is routed to expert j
        topk_weight = router_prob * capacity_limited_topk_expert_mask
        topk_weight = topk_weight.to(orig_dtype)

        return capacity_limited_topk_expert_mask, topk_weight, loss


class Top1Router(_TopKRouter):
    def __init__(
        self,
        noise_policy: RouterExplorationNoisePolicy,
        num_experts: int,
        d_model: int,
        expert_capacity: Optional[Tuple[float, float]] = None,
        alpha: float = 0.01,
        eps: float = 0.1,
        aux_loss_weight: float = 1.0,
        z_loss_weight: float = 1.0,
    ):
        super().__init__(
            noise_policy=noise_policy,
            top_k=1,
            num_experts=num_experts,
            d_model=d_model,
            expert_capacity=expert_capacity,
            alpha=alpha,
            eps=eps,
            aux_loss_weight=aux_loss_weight,
            z_loss_weight=z_loss_weight,
        )


class Top2Router(_TopKRouter):
    def __init__(
        self,
        noise_policy: RouterExplorationNoisePolicy,
        num_experts: int,
        d_model: int,
        expert_capacity: Optional[Tuple[float, float]] = None,
        alpha: float = 0.01,
        eps: float = 0.1,
        aux_loss_weight: float = 1.0,
        z_loss_weight: float = 1.0,
    ):
        super().__init__(
            noise_policy=noise_policy,
            top_k=2,
            num_experts=num_experts,
            d_model=d_model,
            expert_capacity=expert_capacity,
            alpha=alpha,
            eps=eps,
            aux_loss_weight=aux_loss_weight,
            z_loss_weight=z_loss_weight,
        )
