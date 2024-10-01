from typing import List, Optional

import torch
from torch import Tensor

CONSTANT_TRAINING_PARAMETERS = {
    "epochs_per_batch": 1,
    "initial_lr": 0.01,
    "momentum": 0.9,
    "l2": 0.0005,
    "input_image_size": (384, 384),
    "current_run": 0,
    "num_layers": 12,
    "pretrained_weights_path": "weights/pretrained_imagenet/B_16_imagenet1k.pth",
}


def sgd_with_lr_modulation(
    params: List[Tensor],
    d_p_list: List[Tensor],
    f_hat: List[Tensor],
    sum_l_k: List[Tensor],
    t_k: List[Tensor],
    momentum_buffer_list: List[Optional[Tensor]],
    weight_decay: float,
    momentum: float,
    lr: float,
    max_f: float,
):
    assert (
        len(params) == len(d_p_list) == len(f_hat) == len(sum_l_k) == len(t_k)
    ), "Issue with the AR1* provided parameters! Check their creation in the corresponding training loop."

    for i, param in enumerate(params):
        # Prepare AR1* parameters
        initial_param = param.detach().clone().requires_grad_(False)
        initial_d_p = d_p_list[i].detach().clone().requires_grad_(False)

        # Perform SGD
        d_p = d_p_list[i]

        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1)

            d_p = buf

        if f_hat[i] is None:
            param.add_(d_p, alpha=-lr)
        else:
            param.add_((1 - (f_hat[i] / max_f)) * d_p, alpha=-lr)

            # Do necessary updates for AR1*
            sum_l_k[i] += (initial_param - params[i]) * initial_d_p
            t_k[i] = params[i] - initial_param

        # Release memory
        del initial_param
        del initial_d_p
