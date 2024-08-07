import matplotlib.pyplot as plt
import numpy as np
import torch

from datasets.core50.CORE50DataLoader import CORE50DataLoader
from datasets.imagenet.imagenet_1k_class_names import imagenet_classes
from vit_lr.ResizeProcedure import ResizeProcedure
from vit_lr.ViTLR_model import ViTLR


def vit_lr_demo(start_batch=0, start_run=0, start_idx=0):
    weight_path = "weigths/B_16_imagenet1k.pth"
    input_image_size = (384, 384)
    num_layers = 12
    k = 10

    # Generate dataset object
    dataset = CORE50DataLoader(
        root="datasets/core50/data",
        original_image_size=(350, 350),
        input_image_size=input_image_size,
        resize_procedure=ResizeProcedure.BORDER,
        channels=3,
        scenario="ni",
        load_entire_batch=False,
        start_batch=start_batch,
        start_run=start_run,
        start_idx=start_idx,
    )

    # Get test image
    x, _ = dataset.__next__()

    # Load model
    model = ViTLR(input_size=input_image_size, num_layers=num_layers)

    # Load weights
    weights = torch.load(weight_path)

    for i in range(num_layers):
        # torch.eye is an identity matrix
        # Required because the proj_out layer is not present in the default ViT
        weights["transformer.blocks." + str(i) + ".attn.proj_out.weight"] = torch.eye(
            n=model.state_dict()[
                "transformer.blocks." + str(i) + ".attn.proj_out.weight"
            ].shape[0]
        )

    model.load_state_dict(weights)
    print(model, "\n")

    # Display image
    plt.imshow(x[0, ...], interpolation="nearest")
    plt.axis("off")
    plt.show()

    # Perform inference
    y_pred = model(torch.from_numpy(x.astype(np.float32) / 256).permute((0, 3, 1, 2)))[
        0
    ]

    # Extract top k results
    sm = torch.nn.Softmax(dim=0)
    y_pred = sm(y_pred)
    top_k_pred = torch.topk(y_pred, k)

    for i in range(k):
        print(
            imagenet_classes[top_k_pred.indices[i].item()],
            "-",
            str(top_k_pred.values[i].item() * 100)[:5] + "%",
        )

    return None


if __name__ == "__main__":
    vit_lr_demo(start_batch=1, start_run=4, start_idx=400)
