# Based on gsplat; modified from the original.
# Licensed under the Apache License, Version 2.0.

import torch

from ..cuda._wrapper import adam


class SelectiveAdam(torch.optim.Adam):
    """
    A custom optimizer that extends the standard Adam optimizer by
    incorporating selective updates.

    This class is useful for situations where only a subset of parameters
    should be updated at each step, such as in sparse models or in cases where
    parameter visibility is controlled by an external mask.

    Additionally, the operations are fused into a single kernel. This optimizer
    leverages the `adam` function from a CUDA backend for
    optimized sparse updates.

    This is one of the two optimizers mentioned in the Taming3DGS paper.

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        eps (float): Term added to the denominator to improve numerical stability (default: 1e-8).
        betas (Tuple[float, float]): Coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999)).

    Examples:

        >>> N = 100
        >>> param = torch.randn(N, requires_grad=True)
        >>> optimizer = SelectiveAdam([param], eps=1e-8, betas=(0.9, 0.999))
        >>> visibility_mask = torch.cat([torch.ones(50), torch.zeros(50)])  # Visible first half, hidden second half

        >>> # Forward pass
        >>> loss = torch.sum(param ** 2)

        >>> # Backward pass
        >>> loss.backward()

        >>> # Optimization step with selective updates
        >>> optimizer.step(visibility=visibility_mask)

    """

    def __init__(self, params, eps, betas, lr=1e-3, weight_decay=0, force_enable=False):
        super().__init__(params=params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.force_enable = force_enable

    def set_visibility(self, visibility):
        self.visibility = visibility

    def set_index(self, index, length, device="cuda"):
        visibility = torch.zeros([length], dtype=torch.bool, device=device)
        visibility[index] = 1
        self.visibility = visibility

    @torch.no_grad()
    def step(self, visibility=None):
        if visibility is None and hasattr(self, "visibility"):
            visibility = self.visibility
            delattr(self, "visibility")
        if self.force_enable:
            assert visibility is not None

        N = visibility.numel()
        for group in self.param_groups:
            lr = group["lr"]
            eps = group["eps"]
            beta1, beta2 = group["betas"]

            assert len(group["params"]) == 1, "more than one tensor in group"
            param = group["params"][0]
            if param.grad is None:
                continue

            # Lazy state initialization
            state = self.state[param]
            if len(state) == 0:
                state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
                state["exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)
                state["step_per_param"] = torch.zeros(param.shape[0], device=param.device, dtype=torch.float32)

            stored_state = self.state.get(param, None)
            exp_avg = stored_state["exp_avg"]
            exp_avg_sq = stored_state["exp_avg_sq"]
            step_per_param = stored_state["step_per_param"]

            adam(
                param,
                param.grad,
                exp_avg,
                exp_avg_sq,
                visibility,
                step_per_param,
                lr,
                beta1,
                beta2,
                eps,
            )
