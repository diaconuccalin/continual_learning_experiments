import torch
from torch.optim.optimizer import Optimizer, _use_grad_for_differentiable

from evaluation.evaluation_utils import plot_ar1_star_f_hat
from training.training_utils import sgd_with_lr_modulation


class CustomSGD(Optimizer):
    def __init__(
        self,
        params,
        is_backbone,
        w=None,
        f_hat=None,
        sum_l_k=None,
        previous_weights=None,
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
        self.sum_l_k = sum_l_k
        self.previous_weights = previous_weights

        self.current_epoch = 0

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
                previous_weights=self.previous_weights,
                momentum_buffer_list=momentum_buffer_list,
                weight_decay=group["weight_decay"],
                momentum=group["momentum"],
                backbone_lr=group["lr"][self.current_epoch][0],
                head_lr=group["lr"][self.current_epoch][1],
                max_f=self.max_f,
            )

            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state["momentum_buffer"] = momentum_buffer

        return None

    def update_a1_star_params(
        self, current_batch, debug_mode=False, session_name=None, device=None
    ):
        # In debug mode, store f_hat values for plotting
        if debug_mode:
            assert (device is not None) and (
                session_name is not None
            ), "In debug mode, device needs to be passed to the update_a1_star_params method."
            all_f_hat = torch.empty(0).to(device)
        else:
            all_f_hat = None

        # Save current weights to use them in updating f_hat and storing them for the next AR1* iteration
        current_weights = list()
        for el in self.param_groups[0]["params"]:
            if el.grad is not None:
                current_weights.append(el.clone().detach().requires_grad_(False))
        assert len(current_weights) == len(self.previous_weights), (
            "Expected t_k to be the same length as the number of parameters, actually got: "
            + str(len(current_weights))
            + " vs "
            + str(len(self.previous_weights))
        )

        # Update AR1* values
        for i in range(len(self.f_hat)):
            if self.f_hat[i] is not None:
                # Compute new f_hat
                self.f_hat[i] = (self.w[current_batch] * self.sum_l_k[i]) / (
                    (self.previous_weights[i] - current_weights[i]) ** 2 + self.xi
                )

                # Clip f_hat
                self.f_hat[i] = torch.clip(
                    input=self.f_hat[i], min=0.0, max=self.max_f, out=self.f_hat[i]
                )

                # Reset sum_l_k and store current weights in previous weights list
                self.sum_l_k[i].zero_()
                self.previous_weights[i] = current_weights[i]

                # Store f_hat values for plotting
                if debug_mode:
                    all_f_hat = torch.cat((all_f_hat, self.f_hat[i].flatten()))

        # Plot f_hat values
        if debug_mode:
            plot_ar1_star_f_hat(
                all_f_hat=all_f_hat,
                current_batch=current_batch,
                session_name=session_name,
            )

        return None
