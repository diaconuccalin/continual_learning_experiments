import numpy as np
import torch

from core50.CORE50DataLoader import CORE50DataLoader
from vit_lr.ViTLR_model import ViTLR


def vit_lr_demo():
    input_image_size = (384, 384)

    # Generate dataset object
    dataset = CORE50DataLoader(
        root="core50/data",
        original_image_size=(350, 350),
        input_image_size=input_image_size,
        resize_procedure="border",
        channels=3,
        scenario="ni",
        load_entire_batch=False,
        start_batch=1,
        start_run=4,
        start_idx=400,
    )

    x = torch.from_numpy(dataset.__next__()[0].astype(np.float32)).permute((0, 3, 1, 2))

    model = ViTLR(input_size=input_image_size)

    print(model)
    out = model(x)
    print(out)
    print(out.shape)

    return None


if __name__ == "__main__":
    vit_lr_demo()
