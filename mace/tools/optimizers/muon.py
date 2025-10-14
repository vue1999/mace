"""Muon optimizer implementation adapted for MACE models."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import torch


def _zeropower_via_newtonschulz5(matrix: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Compute the zero power of a matrix using the Newton–Schulz method."""

    if matrix.ndim < 2:
        raise ValueError("Muon optimizer expects tensors with at least 2 dimensions")

    a, b, c = (3.4445, -4.7750, 2.0315)
    result = matrix
    transpose = False
    if matrix.size(-2) > matrix.size(-1):
        result = result.mT
        transpose = True

    result = result / (result.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        gram = result @ result.mT
        update = b * gram + c * gram @ gram
        result = a * result + update @ result

    if transpose:
        result = result.mT
    return result


def _muon_update(
    grad: torch.Tensor,
    momentum: torch.Tensor,
    *,
    beta: float,
    ns_steps: int = 5,
) -> torch.Tensor:
    """Apply the Muon update to a gradient tensor."""

    if grad.ndim < 2:
        raise ValueError("Muon optimizer expects tensors with at least 2 dimensions")

    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta)

    original_shape = update.shape
    if update.ndim == 2:
        matrix = update
    else:
        matrix = update.flatten(0, -2)

    matrix = _zeropower_via_newtonschulz5(matrix, steps=ns_steps)
    aspect_ratio = float(matrix.size(-2)) / float(matrix.size(-1))
    matrix = matrix * max(1.0, aspect_ratio) ** 0.5

    return matrix.view(original_shape)


class MuonWithAdam(torch.optim.Optimizer):
    """Combined Muon + Adam optimizer treated as separate parameter groups."""

    def __init__(
        self,
        param_groups: Sequence[dict],
        *,
        ns_steps: int = 5,
    ) -> None:
        validated_groups = []
        for group in param_groups:
            if "use_muon" not in group:
                raise ValueError("Param group must define 'use_muon'")
            validated_groups.append(group)

        defaults = dict(ns_steps=ns_steps)
        super().__init__(validated_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group.get("use_muon", False):
                beta = group.get("momentum", 0.95)
                lr = group["lr"]
                wd = group.get("weight_decay", 0.0)
                for param in group["params"]:
                    if param.grad is None:
                        continue
                    grad = param.grad
                    state = self.state[param]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(param)
                    update = _muon_update(
                        grad.view_as(param),
                        state["momentum_buffer"],
                        beta=beta,
                        ns_steps=self.defaults["ns_steps"],
                    ).reshape_as(param)
                    if wd != 0:
                        param.mul_(1 - lr * wd)
                    param.add_(update, alpha=-lr)
            else:
                lr = group["lr"]
                wd = group.get("weight_decay", 0.0)
                beta1, beta2 = group.get("betas", (0.9, 0.999))
                eps = group.get("eps", 1e-8)
                for param in group["params"]:
                    if param.grad is None:
                        continue
                    grad = param.grad
                    state = self.state[param]
                    if len(state) == 0:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(param)
                        state["exp_avg_sq"] = torch.zeros_like(param)

                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    state["step"] += 1
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    denom = exp_avg_sq.sqrt().add_(eps)
                    bias_correction1 = 1 - beta1 ** state["step"]
                    bias_correction2 = 1 - beta2 ** state["step"]
                    step_size = lr * (bias_correction2 ** 0.5) / bias_correction1
                    if wd != 0:
                        param.mul_(1 - lr * wd)
                    param.addcdiv_(exp_avg, denom, value=-step_size)
        return loss


def build_muon_param_groups(
    param_groups: Sequence[dict],
    *,
    momentum: float,
    betas: Tuple[float, float],
    eps: float,
) -> List[dict]:
    """Split existing parameter groups into Muon and Adam sub-groups."""

    muon_ready_groups: List[dict] = []
    for group in param_groups:
        params = list(group["params"])
        group["params"] = params
        lr = group.get("lr")
        weight_decay = group.get("weight_decay", 0.0)

        muon_params = [param for param in params if param.ndim >= 2]
        adam_params = [param for param in params if param.ndim < 2]

        if muon_params:
            muon_ready_groups.append(
                {
                    "params": muon_params,
                    "use_muon": True,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "momentum": momentum,
                }
            )
        if adam_params:
            muon_ready_groups.append(
                {
                    "params": adam_params,
                    "use_muon": False,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "betas": betas,
                    "eps": eps,
                }
            )

    return muon_ready_groups

