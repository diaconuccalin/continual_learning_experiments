from torch.optim import Optimizer

from training.training_utils import sgd_with_lr_modulation


class WeightConstrainingByLRModulation(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0, weight_decay=0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

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
        loss = None

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []

            self._init_group(group, params_with_grad, d_p_list, momentum_buffer_list)

            sgd_with_lr_modulation(
                params_with_grad,
                d_p_list,
                momentum_buffer_list,
                weight_decay=group["weight_decay"],
                momentum=group["momentum"],
                lr=group["lr"],
            )

            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state["momentum_buffer"] = momentum_buffer

        return loss
