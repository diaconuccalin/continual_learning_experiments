import torch
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from datasets.core50.CORe50DataLoader import CORe50DataLoader
from datasets.core50.constants import (
    CORE50_CLASS_NAMES,
    CORE50_ROOT_PATH,
    CORE50_CATEGORY_NAMES,
)
from models.vit_lr.ResizeProcedure import ResizeProcedure
from models.vit_lr.ViTLR_model import ViTLR
from models.vit_lr.vit_lr_utils import vit_lr_image_preprocessing


def vit_lr_evaluation_pipeline(
    batch,
    input_image_size,
    current_task,
    current_run,
    num_blocks,
    category_based_split,
    device,
    weights_path=None,
    model=None,
):
    # Assert that only one of weights path or model is provided
    assert (weights_path is not None) ^ (
        model is not None
    ), "Exactly one of weights path or model must be provided!"

    # Compute number of classes
    if category_based_split:
        num_classes = len(CORE50_CATEGORY_NAMES)
    else:
        num_classes = len(CORE50_CLASS_NAMES)

    # Generate data loader
    data_loader = CORe50DataLoader(
        root=CORE50_ROOT_PATH,
        original_image_size=(350, 350),
        input_image_size=input_image_size,
        resize_procedure=ResizeProcedure.BORDER,
        image_channels=3,
        scenario=current_task,
        start_run=current_run,
        batch=batch,
        start_idx=0,
        category_based_split=category_based_split,
        randomize_data_order=False,
    )

    losses = None
    if weights_path is not None:
        # Prepare model
        model = ViTLR(
            device=device,
            num_blocks=num_blocks,
            input_size=input_image_size,
            num_classes=num_classes,
            dropout_rate=0.0,
        )

        # Load stored information
        print("Loading pretrained model...")
        weights = torch.load(weights_path, weights_only=False, map_location=device)

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
        x_train = vit_lr_image_preprocessing(x=x_train, device=device)

        y_train = y_train.item()
        y_pred = torch.argmax(model(x=x_train, get_activation=False)).item()

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
