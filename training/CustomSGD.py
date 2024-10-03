import torch
from torch.optim.optimizer import Optimizer, _use_grad_for_differentiable

from training.training_utils import sgd_with_lr_modulation


class CustomSGD(Optimizer):
    def __init__(
        self,
        params,
        is_backbone,
        w=None,
        f_hat=None,
        f=None,
        sum_l_k=None,
        t_k=None,
        lr=None,
        momentum=0,
        weight_decay=0,
        max_f=0.001,
        xi=1e-7,
    ):
        # Verify passed values
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        # Store backbone flags
        self.is_backbone = is_backbone

        # Prepare lr modulation parameters
        self.w = w
        self.max_f = max_f
        self.xi = xi

        self.f_hat = f_hat
        self.f = f
        self.sum_l_k = sum_l_k
        self.t_k = t_k

        self.steps_so_far = 0

        # Default optimizer updates
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            differentiable=False,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("differentiable", False)

    def _init_group(self, group, params_with_grad, d_p_list, momentum_buffer_list):
        for p in group["params"]:
            if p.grad is not None:
                params_with_grad.append(p)
                d_p_list.append(p.grad)

                state = self.state[p]
                momentum_buffer_list.append(state.get("momentum_buffer"))

        return None

    @_use_grad_for_differentiable
    def step(self, closure=None):
        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []

            self._init_group(group, params_with_grad, d_p_list, momentum_buffer_list)

            sgd_with_lr_modulation(
                params=params_with_grad,
                d_p_list=d_p_list,
                is_backbone=self.is_backbone,
                f_hat=self.f_hat,
                sum_l_k=self.sum_l_k,
                t_k=self.t_k,
                momentum_buffer_list=momentum_buffer_list,
                weight_decay=group["weight_decay"],
                momentum=group["momentum"],
                backbone_lr=group["lr"][self.steps_so_far][0],
                head_lr=group["lr"][self.steps_so_far][1],
                max_f=self.max_f,
            )

            self.steps_so_far += 1

            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state["momentum_buffer"] = momentum_buffer

        return None

    def update_a1_star_params(self, current_batch):
        for i in range(len(self.f_hat)):
            if self.f_hat[i] is not None:
                self.f_hat[i] = self.sum_l_k[i] / (self.t_k[i] ** 2 + self.xi)
                self.f[i] = self.f[i] + self.w[current_batch] * self.f_hat[i]
                self.f_hat[i] = torch.clip(
                    input=self.f[i], max=self.max_f, out=self.f_hat[i]
                )

        return None
