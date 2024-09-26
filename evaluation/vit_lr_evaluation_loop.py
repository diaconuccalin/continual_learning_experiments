import numpy as np
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from datasets.core50.CORe50DataLoader import CORe50DataLoader
from datasets.core50.constants import (
    CORE50_CLASS_NAMES,
    CORE50_ROOT_PATH,
    CORE50_CATEGORY_NAMES,
    NI_TESTING_BATCH,
    NIC_CUMULATIVE_TESTING_BATCH,
    NC_TESTING_BATCH,
)
from models.vit_lr.ResizeProcedure import ResizeProcedure
from models.vit_lr.ViTLR_model import ViTLR
from models.vit_lr.vit_lr_utils import bordering_resize, vit_lr_image_preprocessing


def vit_lr_single_evaluation(
    num_layers,
    mini_batch_size,
    input_size,
    weights_path,
    image_path,
    original_image_size,
    input_image_size,
    device,
    category_based_split,
):
    # Compute number of classes
    if category_based_split:
        num_classes = len(CORE50_CATEGORY_NAMES)
    else:
        num_classes = len(CORE50_CLASS_NAMES)

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


def vit_lr_evaluation_pipeline(
    input_image_size,
    current_task,
    current_run,
    num_layers,
    weights_path,
    category_based_split,
    device,
):
    # Compute number of classes
    if category_based_split:
        num_classes = len(CORE50_CATEGORY_NAMES)
    else:
        num_classes = len(CORE50_CLASS_NAMES)

    # Choose batch
    if current_task == "ni":
        batch = NI_TESTING_BATCH
    elif current_task in ["nc", "multi-task-nc"]:
        batch = NC_TESTING_BATCH
    elif current_task in ["nic", "nicv2_391"]:
        batch = NIC_CUMULATIVE_TESTING_BATCH
    else:
        raise ValueError("Invalid task name!")

    # Generate data loader
    data_loader = CORe50DataLoader(
        root=CORE50_ROOT_PATH,
        original_image_size=(350, 350),
        input_image_size=input_image_size,
        resize_procedure=ResizeProcedure.BORDER,
        image_channels=3,
        scenario=current_task,
        mini_batch_size=1,
        start_run=current_run,
        batch=batch,
        start_idx=0,
        category_based_split=category_based_split,
        randomize_data_order=False,
    )

    # Prepare model
    model = ViTLR(
        device=device,
        num_layers=num_layers,
        input_size=input_image_size,
        num_classes=num_classes,
    )

    # Load stored information
    print("Loading pretrained model...")
    weights = torch.load(weights_path, weights_only=False, map_location=device)

    losses = None
    if "loss" in weights.keys():
        losses = weights["loss"]

    if "model_state_dict" in weights.keys():
        weights = weights["model_state_dict"]

    model.load_state_dict(weights)

    # Move model to GPU
    model.to(device)

    # Prepare model for evaluation
    model.eval()

    # Compute accuracy
    n_correct_preds = 0
    preds_so_far = 0
    total_samples = len(data_loader.idx_order)

    # Store for conf matrix
    all_y_trains = list()
    all_y_preds = list()

    # Prepare progress bar
    progress_bar = tqdm(range(total_samples), colour="yellow", desc="Evaluation")

    # Iterate through samples
    for _ in progress_bar:
        # Prepare sample
        x_train, y_train = data_loader.__next__()
        x_train = vit_lr_image_preprocessing(x_train).to(device)

        y_train = y_train.item()
        y_pred = torch.argmax(model(x_train)).item()

        all_y_trains.append(y_train)
        all_y_preds.append(y_pred)

        preds_so_far += 1
        if y_train == y_pred:
            n_correct_preds += 1

        progress_bar.set_postfix_str(
            f"Accuracy: %0.3f" % (100 * n_correct_preds / preds_so_far) + "%"
        )

    return (
        losses,
        n_correct_preds / preds_so_far,
        confusion_matrix(all_y_trains, all_y_preds, labels=range(num_classes)),
    )
