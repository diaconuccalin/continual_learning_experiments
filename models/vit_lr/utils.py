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


def vit_lr_image_preprocessing(x):
    x = x.astype(np.float32)

    x /= 255

    x[..., 0] -= 0.485
    x[..., 1] -= 0.456
    x[..., 2] -= 0.406

    x[..., 0] /= 0.229
    x[..., 1] /= 0.224
    x[..., 2] /= 0.225

    return torch.from_numpy(x).permute((0, 3, 1, 2))


def bordering_resize(x, original_image_size, input_image_size):
    # Prepare new image
    new_x = np.full(
        (
            x.shape[0],
            input_image_size[0],
            input_image_size[1],
            x.shape[-1],
        ),
        fill_value=128,
    )

    # Compute horizontal and vertical border dimension
    border_size_0 = int((input_image_size[0] - original_image_size[0]) / 2)
    border_size_1 = int((input_image_size[1] - original_image_size[1]) / 2)

    border_size_0_remaining = (
        input_image_size[0] - original_image_size[0] - border_size_0
    )
    border_size_1_remaining = (
        input_image_size[1] - original_image_size[1] - border_size_1
    )

    # Perform bordering
    new_x[
        :,
        border_size_0:-border_size_0_remaining,
        border_size_1:-border_size_1_remaining,
        :,
    ] = x
    return new_x
