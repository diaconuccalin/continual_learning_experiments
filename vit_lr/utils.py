import numpy as np
import torch


def q_rsqrt(x):
    with torch.no_grad():
        y = np.asarray((x,), dtype=np.float32)
        x2 = y * 0.5
        i = y.view(np.int32)
        i = np.right_shift(i, 1)
        i = 0x5F3759DF - i
        y = i.view(np.float32)
        y = y * (1.5 - (x2 * y * y))

        result = torch.from_numpy(y)

    return result


def fastexp_gist(x):
    x_copy = x.type(torch.float32)
    x_copy = x_copy * 12102203.17133801 + 1064986823.010288
    x_copy = torch.where(x_copy < 8388608, 0, x_copy).type(torch.float32)
    x_copy = torch.where(x_copy > 2139095040, 2139095040, x_copy).type(torch.float32)

    return x_copy.type(torch.uint32).view(torch.float32)
