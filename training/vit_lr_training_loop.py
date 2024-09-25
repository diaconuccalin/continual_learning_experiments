import os

import numpy as np
import torch
from tqdm import tqdm

from datasets.core50.CORe50DataLoader import CORe50DataLoader
from datasets.core50.constants import (
    CORE50_ROOT_PATH,
    CORE50_CATEGORY_NAMES,
    CORE50_CLASS_NAMES,
    NEW_TO_OLD_NAMES,
)
from models.vit_lr.ResizeProcedure import ResizeProcedure
from models.vit_lr.ViTLR_model import ViTLR
from models.vit_lr.utils import vit_lr_image_preprocessing


def vit_lr_epoch(
    model,
    data_loader,
    optimizer,
    criterion,
    current_epoch,
    total_epochs,
    current_task,
    current_run,
    current_batch,
    mini_batch_size,
    save_dir_path,
    device,
    profiling_activated,
):
    losses_list = list()
    batch_len = len(data_loader.lup[current_task][current_run][current_batch]) + sum(
        [len(data_loader.exemplar_set[key]) for key in data_loader.exemplar_set.keys()]
    )
    if mini_batch_size < 1 or mini_batch_size > batch_len:
        mini_batch_size = batch_len

    # Setup progress bar
    progress_bar = tqdm(
        range((batch_len) // mini_batch_size),
        colour="green",
        desc="Epoch " + str(current_epoch),
        postfix={"loss": 0.0},
    )

    # Implement profiling
    if profiling_activated:
        prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=10, warmup=10, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./logs/testing"),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )

        prof.start()

    # Iterate through batch
    for _ in progress_bar:
        # Perform profiler step
        if profiling_activated:
            prof.step()

        # Reset gradients
        optimizer.zero_grad()

        # Prepare gradient accumulator
        grads = list()

        for step_in_mini_batch in range(mini_batch_size):
            # Get sample
            x_train, y_train = data_loader.__next__()

            if data_loader.debug_mode:
                continue

            # Load on GPU
            x_train = vit_lr_image_preprocessing(x_train).to(device)
            y_train = y_train.to(device)

            # Forward
            y_pred = model(x_train)

            # Backward step
            loss = criterion(y_pred, y_train)
            loss.backward()

            # Store loss
            losses_list.append(loss.item())

            # Accumulate model gradients
            if len(grads) == 0:
                for el in model.parameters():
                    grads.append(el.grad)
            else:
                for i, el in enumerate(model.parameters()):
                    if grads[i] is not None:
                        grads[i] += el.grad

            # Reset model gradients
            model.zero_grad()

        if data_loader.debug_mode:
            continue

        # Set accumulated model gradients to prepare for update
        for i, el in enumerate(model.parameters()):
            if grads[i] is not None:
                el.grad = grads[i] / mini_batch_size

        # Update model
        optimizer.step()

        # Update progress bar
        progress_bar.set_postfix(
            loss=np.array(losses_list[-mini_batch_size:]).sum() / mini_batch_size
        )

    # Save model only when debug mode is not active
    if not data_loader.debug_mode:
        print("\nSaving model...\n")
        torch.save(
            {
                "epoch": current_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": losses_list,
            },
            os.path.join(save_dir_path, "checkpoint_e" + str(current_epoch) + ".pth"),
        )


def vit_naive_rehearsal_training_pipeline(
    batches,
    epochs_per_batch,
    initial_lr,
    momentum,
    l2,
    input_image_size,
    current_task,
    current_run,
    num_layers,
    exemplar_set_ratio,
    device,
    pretrained_weights_path,
    session_name,
    trainable_backbone,
    randomize_data_order,
    category_based_split,
    profiling_activated,
    data_loader_debug_mode=False,
):
    # Obtain old scenario name
    current_task = NEW_TO_OLD_NAMES[current_task]

    # Compute number of classes
    if category_based_split:
        num_classes = len(CORE50_CATEGORY_NAMES)
    else:
        num_classes = len(CORE50_CLASS_NAMES)

    # Generate data loader
    print("Creating data loader...")
    data_loader = CORe50DataLoader(
        root=CORE50_ROOT_PATH,
        original_image_size=(350, 350),
        input_image_size=input_image_size,
        resize_procedure=ResizeProcedure.BORDER,
        image_channels=3,
        scenario=current_task,
        mini_batch_size=1,
        exemplar_set_ratio=exemplar_set_ratio,
        start_run=current_run,
        randomize_data_order=randomize_data_order,
        category_based_split=category_based_split,
        debug_mode=data_loader_debug_mode,
    )

    # Prepare model save path
    print("Preparing model save path...")
    root_path = os.path.join("weights", "trained_models")
    save_path = os.path.join(root_path, session_name)

    if os.path.exists(save_path):
        print(
            "\n\nA session with the same name already exists in the directory containing the saved models. "
            "\nANY EXISTING RESULT WILL BE OVERWRITTEN AT THE END OF EACH EPOCH!\n\n"
        )
    else:
        if not os.path.exists(root_path):
            os.mkdir(root_path)
        os.mkdir(save_path)

    # ~~~~ Prepare model
    print("Preparing model...")
    # Generate model object
    model = ViTLR(
        device=device,
        num_layers=num_layers,
        input_size=input_image_size,
        num_classes=num_classes,
    )

    # Load weights
    print("Loading pretrained weights...")
    weights = torch.load(
        pretrained_weights_path, weights_only=False, map_location=device
    )

    if "model_state_dict" in weights.keys():
        weights = weights["model_state_dict"]

    # Required for fully connected layer
    # Overwrite original weights for ImageNet 1k (1000 classes => incompatible fc layer dimension)
    weights["fc.weight"] = model.fc.weight.data
    weights["fc.bias"] = model.fc.bias.data

    # Required because the proj_out layer is not present in the default ViT
    for i in range(num_layers):
        # torch.eye is an identity matrix
        weights["transformer.blocks." + str(i) + ".attn.proj_out.weight"] = torch.eye(
            n=model.state_dict()[
                "transformer.blocks." + str(i) + ".attn.proj_out.weight"
            ].shape[0]
        )
    model.load_state_dict(weights)

    # Set whether backbone is trainable
    print("Marking backbone as trainable or not...")
    model.set_backbone_trainable(trainable_backbone)

    # Mark proj_out as frozen
    for i in range(num_layers):
        model.transformer.blocks[i].attn.proj_out.weight.requires_grad = False
        if model.transformer.blocks[i].attn.proj_out.bias is not None:
            model.transformer.blocks[i].attn.proj_out.bias.requires_grad = False

    # Move model to GPU
    model.to(device)

    # Prepare for training
    model.train()

    # ~~~~ Prepare optimizer
    print("Preparing optimizer and loss function...")
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=initial_lr,
        momentum=momentum,
        weight_decay=l2,
    )
    criterion = torch.nn.CrossEntropyLoss()

    # Iterate through batches and epochs
    print("Starting the training loop...\n")
    current_epoch = 0
    for current_batch in batches:
        for e in range(epochs_per_batch):
            current_epoch += 1

            # Set data loader parameters
            data_loader.update_batch(current_batch)
            data_loader.idx = 0

            # Run epoch
            vit_lr_epoch(
                model=model,
                data_loader=data_loader,
                optimizer=optimizer,
                criterion=criterion,
                current_epoch=current_epoch,
                total_epochs=epochs_per_batch * len(batches),
                current_task=current_task,
                current_run=current_run,
                current_batch=current_batch,
                mini_batch_size=32,
                save_dir_path=save_path,
                device=device,
                profiling_activated=profiling_activated,
            )

            # Update for rehearsal
            data_loader.update_exemplar_set()
