import torch
from torch.optim import Optimizer

from training.training_utils import sgd_with_lr_modulation


class WeightConstrainingByLRModulation(Optimizer):
    def __init__(
        self,
        params,
        device,
        w,
        lr=0.01,
        momentum=0,
        weight_decay=0,
        max_f=0.001,
        xi=1e-7,
    ):
        # Verify passed values
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        # Prepare lr modulation parameters
        self.w = w
        self.max_f = max_f
        self.xi = xi

        self.f_hat = list()
        self.f = list()
        self.sum_l_k = list()
        self.t_k = list()
        all_params = list()

        backbone_params = dict(list(params)[0])
        for el in backbone_params.keys():
            all_params.append(backbone_params[el])
            self.f_hat.append(
                torch.zeros_like(backbone_params[el]).to(device).requires_grad_(False)
            )
            self.f.append(
                torch.zeros_like(backbone_params[el]).to(device).requires_grad_(False)
            )
            self.sum_l_k.append(
                torch.zeros_like(backbone_params[el]).to(device).requires_grad_(False)
            )
            self.t_k.append(
                torch.zeros_like(backbone_params[el]).to(device).requires_grad_(False)
            )

        tw_params = dict(list(params)[1])
        for el in tw_params.keys():
            all_params.append(tw_params[el])
            self.f_hat.append(None)
            self.f.append(None)
            self.sum_l_k.append(None)
            self.t_k.append(None)

        # Default optimizer updates
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        super().__init__(all_params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    def _init_group(self, group, params_with_grad, d_p_list, momentum_buffer_list):
        for p in group["params"]:
            if p.grad is not None:
                params_with_grad.append(p)
                d_p_list.append(p.grad)

                state = self.state[p]
                momentum_buffer_list.append(state.get("momentum_buffer"))

        return None

    def step(self, closure=None):
        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []

            self._init_group(group, params_with_grad, d_p_list, momentum_buffer_list)

            sgd_with_lr_modulation(
                params=params_with_grad,
                sum_l_k=self.sum_l_k,
                t_k=self.t_k,
                d_p_list=d_p_list,
                f_hat=self.f_hat,
                momentum_buffer_list=momentum_buffer_list,
                weight_decay=group["weight_decay"],
                momentum=group["momentum"],
                lr=group["lr"],
                max_f=self.max_f,
            )

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
