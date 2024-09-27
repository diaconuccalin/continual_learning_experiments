import os

import torch
from tqdm import tqdm

from datasets.core50.CORe50DataLoader import CORe50DataLoader
from datasets.core50.constants import (
    CORE50_ROOT_PATH,
    CORE50_CATEGORY_NAMES,
    CORE50_CLASS_NAMES,
)
from models.vit_lr.ResizeProcedure import ResizeProcedure
from models.vit_lr.ViTLR_model import ViTLR
from models.vit_lr.vit_lr_utils import vit_lr_image_preprocessing


def vit_lr_epoch(
    model,
    data_loader,
    optimizer,
    criterion,
    current_epoch,
    total_epochs,
    mini_batch_size,
    save_dir_path,
    model_saving_frequency,
    device,
    profiling_activated,
):
    losses_list = list()

    # Determine batch sizes
    batch_len = len(data_loader.idx_order)
    if mini_batch_size < 1 or mini_batch_size > batch_len:
        mini_batch_size = batch_len

    # Setup progress bar for mini-batches smaller than entire batch
    steps_number = batch_len // mini_batch_size
    if steps_number > 1:
        progress_bar = tqdm(
            range(steps_number),
            colour="green",
            desc="Epoch " + str(current_epoch) + "/" + str(total_epochs),
            postfix={"loss": 0.0},
        )
    else:
        progress_bar = range(steps_number)

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

        # Setup progress bar for mini-batches of the same size as the entire batch
        if steps_number <= 1:
            sub_progress_bar = tqdm(
                range(mini_batch_size),
                colour="green",
                desc="Epoch " + str(current_epoch) + "/" + str(total_epochs),
                postfix={"loss": 0.0},
            )
        else:
            sub_progress_bar = range(mini_batch_size)

        # Iterate through mini-batch
        for step_in_mini_batch in sub_progress_bar:
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

            if steps_number <= 1:
                sub_progress_bar.set_postfix(
                    loss=sum(losses_list[-step_in_mini_batch:]) / step_in_mini_batch
                )

        if data_loader.debug_mode:
            continue

        # Set accumulated model gradients to prepare for update
        for i, el in enumerate(model.parameters()):
            if grads[i] is not None:
                el.grad = grads[i] / mini_batch_size

        # Update model
        optimizer.step()

        # Update progress bar
        if steps_number > 1:
            progress_bar.set_postfix(
                loss=sum(losses_list[-mini_batch_size:]) / mini_batch_size
            )

    # Populate rehearsal memory
    data_loader.populate_rm()

    # Save model only when debug mode is not active
    if not data_loader.debug_mode and (
        (model_saving_frequency > 0 and (current_epoch % model_saving_frequency == 0))
        or current_epoch == total_epochs
    ):
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


def vit_native_rehearsal_training_pipeline(
    batches,
    # If value set to anything lower than 1, the training loop will run for 1 epoch
    # over all the batches considered as a single batch
    epochs_per_batch,
    initial_lr,
    momentum,
    l2,
    input_image_size,
    current_task,
    current_run,
    num_layers,
    mini_batch_size,
    rehearsal_memory_size,
    device,
    pretrained_weights_path,
    session_name,
    model_saving_frequency,
    trainable_backbone,
    randomize_data_order,
    category_based_split,
    profiling_activated,
    data_loader_debug_mode=False,
):
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
        rehearsal_memory_size=rehearsal_memory_size,
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
            "\nANY EXISTING RESULT MAY BE OVERWRITTEN AT THE END OF EACH EPOCH!\n\n"
        )
    else:
        if not os.path.exists(root_path):
            os.mkdir(root_path)
        os.mkdir(save_path)

    # Prepare model
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

    # Prepare optimizer
    print("Preparing optimizer and loss function...")
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=initial_lr,
        momentum=momentum,
        weight_decay=l2,
    )
    criterion = torch.nn.CrossEntropyLoss()

    # Iterate through batches and epochs
    print("\nStarting the training loop...\n")

    if epochs_per_batch < 1:
        # Set data loader parameters
        data_loader.update_batch(0)
        data_loader.idx = 0

        # Run epoch
        vit_lr_epoch(
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            criterion=criterion,
            current_epoch=1,
            total_epochs=1,
            mini_batch_size=mini_batch_size,
            save_dir_path=save_path,
            model_saving_frequency=model_saving_frequency,
            device=device,
            profiling_activated=profiling_activated,
        )
    else:
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
                    mini_batch_size=mini_batch_size,
                    save_dir_path=save_path,
                    model_saving_frequency=model_saving_frequency,
                    device=device,
                    profiling_activated=profiling_activated,
                )

    return None


def vit_cwr_star_training_pipeline(
    batches,
    # If value set to anything lower than 1, the training loop will run for 1 epoch
    # over all the batches considered as a single batch
    epochs_per_batch,
    initial_lr,
    momentum,
    l2,
    input_image_size,
    current_task,
    current_run,
    num_layers,
    mini_batch_size,
    rehearsal_memory_size,
    device,
    pretrained_weights_path,
    session_name,
    model_saving_frequency,
    randomize_data_order,
    category_based_split,
    profiling_activated,
    data_loader_debug_mode=False,
):
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
        rehearsal_memory_size=rehearsal_memory_size,
        batch=batches[0],
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
            "\nANY EXISTING RESULT MAY BE OVERWRITTEN AT THE END OF EACH EPOCH!\n\n"
        )
    else:
        if not os.path.exists(root_path):
            os.mkdir(root_path)
        os.mkdir(save_path)

    # Prepare model
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

    # Prepare consolidated weights (and biases) tensors and other required variables
    cw = torch.zeros(model.fc.weight.shape).to(device)
    cb = torch.zeros(model.fc.bias.shape).to(device)
    past = torch.zeros(num_classes).to(device)
    w_past = torch.zeros(num_classes).to(device)

    # Required because the proj_out layer is not present in the default ViT
    for i in range(num_layers):
        # torch.eye is an identity matrix
        weights["transformer.blocks." + str(i) + ".attn.proj_out.weight"] = torch.eye(
            n=model.state_dict()[
                "transformer.blocks." + str(i) + ".attn.proj_out.weight"
            ].shape[0]
        )
    model.load_state_dict(weights)

    # Mark proj_out as frozen
    for i in range(num_layers):
        model.transformer.blocks[i].attn.proj_out.weight.requires_grad = False
        if model.transformer.blocks[i].attn.proj_out.bias is not None:
            model.transformer.blocks[i].attn.proj_out.bias.requires_grad = False

    # Move model to GPU
    model.to(device)

    # Prepare for training
    model.train()

    # Prepare optimizer
    print("Preparing optimizer and loss function...")
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=initial_lr,
        momentum=momentum,
        weight_decay=l2,
    )
    criterion = torch.nn.CrossEntropyLoss()

    # Iterate through batches and epochs
    print("\nStarting the training loop...\n")
    current_epoch = 0

    for current_batch in batches:
        print("\nStarting epoch...\n")
        current_epoch += 1

        # Set data loader parameters
        data_loader.update_batch(current_batch)
        data_loader.idx = 0

        # Find classes occurring in the current batch (and their occurrence count)
        cur = data_loader.get_classes_in_current_batch()

        # Update temporary weights (tw)
        model.fc.weight.data.zero_()
        model.fc.bias.data.zero_()

        for j in cur.keys():
            model.fc.weight.data[j] = cw[j]
            model.fc.bias.data[j] = cb[j]

        # Freeze backbone if required
        if current_epoch == 1:
            model.set_backbone_trainable(True)
        else:
            model.set_backbone_trainable(False)

        # Run epoch
        vit_lr_epoch(
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            criterion=criterion,
            current_epoch=current_epoch,
            total_epochs=epochs_per_batch * len(batches),
            mini_batch_size=mini_batch_size,
            save_dir_path=save_path,
            model_saving_frequency=model_saving_frequency,
            device=device,
            profiling_activated=profiling_activated,
        )

        print("\nUpdating consolidated weights...\n")
        # Update consolidated weights
        for j in cur.keys():
            w_past[j] = torch.sqrt(past[j] / cur[j])

            cw[j] = (
                cw[j] * w_past[j]
                + (model.fc.weight.data[j] - torch.mean(model.fc.weight.data))
            ) / (w_past[j] + 1)
            cb[j] = (
                cb[j] * w_past[j]
                + (model.fc.bias.data[j] - torch.mean(model.fc.bias.data))
            ) / (w_past[j] + 1)

            past[j] += cur[j]

        print(cur.keys())
        print(cw)

    return None
