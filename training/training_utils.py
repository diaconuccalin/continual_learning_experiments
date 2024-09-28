from typing import List, Optional

import torch
from torch import Tensor


def sgd_with_lr_modulation(
    params: List[Tensor],
    d_p_list: List[Tensor],
    momentum_buffer_list: List[Optional[Tensor]],
    weight_decay: float,
    momentum: float,
    lr: float,
):
    for i, param in enumerate(params):
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

        # TODO: Add weight constraining component
        params[i] = param - lr * d_p
