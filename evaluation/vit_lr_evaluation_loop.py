import numpy as np
import torch
from PIL import Image

from datasets.core50.constants import CORE50_CLASS_NAMES
from models.vit_lr.ViTLR_model import ViTLR
from models.vit_lr.utils import bordering_resize, vit_lr_image_preprocessing


def vit_lr_single_evaluation(
    num_layers,
    input_size,
    weights_path,
    image_path,
    original_image_size,
    input_image_size,
    device,
    num_classes=len(CORE50_CLASS_NAMES),
):
    # Create model object
    model = ViTLR(
        device=device,
        num_layers=num_layers,
        input_size=input_size,
        num_classes=num_classes,
    )

    # Load and adjust weights
    weights = torch.load(
        weights_path,
        weights_only=False,
        map_location=device,
    )

    if "model_state_dict" in weights.keys():
        weights = weights["model_state_dict"]

    weights["fc.weight"] = model.fc.weight.data
    weights["fc.bias"] = model.fc.bias.data

    for i in range(num_layers):
        # torch.eye is an identity matrix
        weights["transformer.blocks." + str(i) + ".attn.proj_out.weight"] = torch.eye(
            n=model.state_dict()[
                "transformer.blocks." + str(i) + ".attn.proj_out.weight"
            ].shape[0]
        )

        # Mark proj_out as frozen
        model.transformer.blocks[i].attn.proj_out.weight.requires_grad = False
        if model.transformer.blocks[i].attn.proj_out.bias is not None:
            model.transformer.blocks[i].attn.proj_out.bias.requires_grad = False

    # Prepare model
    model.load_state_dict(weights)
    model.to(device)
    model.eval()

    # Prepare image
    img = Image.open(image_path)
    img = np.array(img)
    img = img[np.newaxis, :]

    img = bordering_resize(
        img, original_image_size=original_image_size, input_image_size=input_image_size
    )
    img = vit_lr_image_preprocessing(img)
    img = img.to(device)

    pred = model(img)
    sm = torch.nn.Softmax(dim=1)

    return sm(pred)
