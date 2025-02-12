import os
import random

import torch
from tqdm import tqdm

from datasets.core50.CORe50DataLoader import CORe50DataLoader
from datasets.core50.constants import (
    CORE50_ROOT_PATH,
    CORE50_CATEGORY_NAMES,
    CORE50_CLASS_NAMES,
)
from evaluation.evaluation_utils import plot_confusion_matrix
from evaluation.vit_lr_evaluation_loop import vit_lr_evaluation_pipeline
from models.vit_lr.ResizeProcedure import ResizeProcedure
from models.vit_lr.ViTLR_model import ViTLR
from models.vit_lr.vit_lr_utils import vit_lr_image_preprocessing
from training.CustomSGD import CustomSGD
from training.PipelineScenario import (
    PIPELINES_WITH_LEARNING_RATE_MODULATION,
    LR_PIPELINES,
    PIPELINES_WITH_RM,
    AR1_STAR_PURE_PIPELINES,
    CWR_STAR_PIPELINES,
    PipelineScenario,
)


def vit_lr_epoch(
    model,
    data_loader,
    optimizer,
    criterion,
    current_epoch,
    total_epochs,
    mini_batch_size,
    session_name,
    save_dir_path,
    model_saving_frequency,
    current_scenario,
    current_run,
    num_blocks,
    populate_rm_epochs,
    category_based_split,
    device,
    profiling_activated,
    should_validate=False,
    validation_batch=None,
):
    losses_list = list()

    # Check that validation batch is provided if needed
    if should_validate:
        assert (
            validation_batch is not None
        ), "Validation batch must be provided when validation step is required."

    # Determine batch sizes
    batch_len = len(data_loader.idx_order)
    if mini_batch_size < 1 or mini_batch_size > batch_len:
        mini_batch_size = batch_len

    # Choose activations to store
    if data_loader.use_latent_replay and current_epoch in populate_rm_epochs:
        potential_new_activations_indexes = random.sample(
            data_loader.idx_order, data_loader.h
        )
        actual_new_activations_indexes = list()
        new_activations = list()
    else:
        potential_new_activations_indexes = None
        actual_new_activations_indexes = None
        new_activations = None

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
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                os.path.join(".", "logs", session_name)
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )

        prof.start()
    else:
        prof = None

    # Iterate through batch
    for _ in progress_bar:
        # Perform profiler step
        if profiling_activated:
            prof.step()

        # Reset gradients
        optimizer.zero_grad()

        # Prepare gradient accumulator
        grads = dict()
        number_of_encounters = dict()
        for name, _ in model.named_parameters():
            grads[name] = None
            number_of_encounters[name] = 0

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
            x_train = vit_lr_image_preprocessing(x=x_train, device=device)
            y_train = y_train.to(device)

            # Forward
            if (potential_new_activations_indexes is not None) and (
                (data_loader.idx_order[data_loader.idx - 1])
                in potential_new_activations_indexes
            ):
                y_pred, activation = model(x=x_train, get_activation=True)
                new_activations.append((activation, y_train))
                actual_new_activations_indexes.append(
                    data_loader.idx_order[data_loader.idx - 1]
                )
            else:
                y_pred = model(x=x_train, get_activation=False)

            # Backward step
            loss = criterion(y_pred, y_train)
            loss.backward()

            # Store loss
            losses_list.append(loss.item())

            # Accumulate model gradients
            for name, el in model.named_parameters():
                if el.grad is not None:
                    number_of_encounters[name] += 1

                if grads[name] is None:
                    grads[name] = el.grad
                elif el.grad is not None:
                    grads[name] += el.grad

            # Reset model gradients
            model.zero_grad()

            if steps_number <= 1:
                sub_progress_bar.set_postfix(
                    loss=sum(losses_list[-step_in_mini_batch:]) / step_in_mini_batch
                )

        if data_loader.debug_mode:
            if potential_new_activations_indexes is not None:
                new_activations = [(None, -1)] * len(potential_new_activations_indexes)
            continue

        # Set accumulated model gradients to prepare for update
        for name, el in model.named_parameters():
            if grads[name] is not None:
                el.grad = grads[name] / number_of_encounters[name]

        # Update model
        optimizer.current_epoch = current_epoch - 1
        optimizer.step()

        # Update progress bar
        if steps_number > 1:
            progress_bar.set_postfix(
                loss=sum(losses_list[-mini_batch_size:]) / mini_batch_size
            )

    if current_scenario in PIPELINES_WITH_RM:
        # Replace activations with new ones
        if (current_scenario in LR_PIPELINES) and (
            potential_new_activations_indexes is not None
        ):
            data_loader.stored_activations_indexes = actual_new_activations_indexes
            data_loader.stored_activations = new_activations
            data_loader.h = len(actual_new_activations_indexes)

        # Populate rehearsal memory
        if current_epoch in populate_rm_epochs:
            print("\nPopulating rehearsal memory...\n")
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

    # Validate if required
    if should_validate:
        _, acc, conf_mat = vit_lr_evaluation_pipeline(
            batch=validation_batch,
            input_image_size=data_loader.input_image_size,
            current_task=data_loader.scenario,
            current_run=current_run,
            category_based_split=category_based_split,
            device=device,
            model=model,
        )

        plot_confusion_matrix(
            conf_mat=conf_mat,
            labels=CORE50_CATEGORY_NAMES,
            category_based_split=category_based_split,
            save_location=os.path.join(
                save_dir_path,
                "validation_confusion_matrix_e" + str(current_epoch) + ".png",
            ),
        )


def vit_training_pipeline(
    batches,
    initial_batches,
    # If value set to anything lower than 1, the training loop will run for 1 epoch
    # over all the batches considered as a single batch
    epochs_per_batch,
    learning_rates,
    momentum,
    l2,
    input_image_size,
    current_task,
    current_run,
    num_blocks,
    mini_batch_size,
    rehearsal_memory_size,
    populate_rm_epochs,
    device,
    pretrained_weights_path,
    current_scenario,
    session_name,
    lr_modulation_batch_specific_weights=None,
    xi=1e-7,
    max_f=0.001,
    latent_replay_layer=-1,
    model_saving_frequency=1,
    randomize_data_order=True,
    category_based_split=False,
    profiling_activated=False,
    data_loader_debug_mode=False,
    should_validate=False,
    validation_batch=None,
):
    # Check that batch-specific weights are provided when dealing with AR1*
    if current_scenario in PIPELINES_WITH_LEARNING_RATE_MODULATION:
        assert (
            lr_modulation_batch_specific_weights is not None
        ), "Batch-specific weights required for AR1* pipeline!"

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
        rehearsal_memory_size=rehearsal_memory_size,
        batch=(
            batches
            if current_scenario is PipelineScenario.NATIVE_CUMULATIVE
            else batches[0]
        ),
        start_run=current_run,
        initial_batches=initial_batches,
        use_latent_replay=current_scenario in LR_PIPELINES,
        randomize_data_order=randomize_data_order,
        category_based_split=category_based_split,
        debug_mode=data_loader_debug_mode,
        mini_batch_size=mini_batch_size,
        keep_rehearsal_proportion=current_scenario in LR_PIPELINES,
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
        num_blocks=num_blocks,
        input_size=input_image_size,
        num_classes=num_classes,
        latent_replay_layer=latent_replay_layer,
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

    if current_scenario in PIPELINES_WITH_RM:
        # Prepare consolidated weights (and biases) tensors and other required variables
        cw = torch.zeros(model.fc.weight.shape).to(device)
        cb = torch.zeros(model.fc.bias.shape).to(device)
        past = torch.zeros(num_classes).to(device)
        w_past = torch.zeros(num_classes).to(device)
    else:
        cw = cb = past = w_past = None

    # Required because the proj_out layer is not present in the default ViT
    for i in range(num_blocks):
        # torch.eye is an identity matrix
        weights["transformer.blocks." + str(i) + ".attn.proj_out.weight"] = torch.eye(
            n=model.state_dict()[
                "transformer.blocks." + str(i) + ".attn.proj_out.weight"
            ].shape[0]
        )

    # Remove unused transformer blocks
    keys_to_delete = list()
    for i in range(num_blocks, 12):
        for el in weights.keys():
            if "transformer.blocks." + str(i) + "." in el:
                keys_to_delete.append(el)

    for key in keys_to_delete:
        weights.pop(key)

    # Actually load weights into model
    model.load_state_dict(weights)

    # Set backbone as trainable (it will be changed according to the scenario and current epoch)
    model.set_backbone_requires_grad(True)

    # Move model to GPU
    model.to(device)

    # Prepare for training
    model.train()

    # Prepare optimizer
    print("Preparing optimizer and loss function...")

    # Mark backbone trainable parameters
    is_backbone = dict()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "fc." not in name:
            is_backbone[name] = True
        else:
            is_backbone[name] = False

    if current_scenario in AR1_STAR_PURE_PIPELINES:
        # Initialize a_star parameters
        f_hat = dict()
        sum_l_k = dict()
        previous_weights = dict()

        for name, param in model.named_parameters():
            # Required, since in the optimizer, only params with grads will be considered
            if not param.requires_grad:
                continue

            if "fc." not in name:
                # Treat backbone parameters
                f_hat[name] = torch.zeros_like(param).to(device).requires_grad_(False)
                sum_l_k[name] = torch.zeros_like(param).to(device).requires_grad_(False)
                previous_weights[name] = (
                    param.clone().detach().to(device).requires_grad_(False)
                )
            else:
                # Treat head (tw/temporary weights) parameters
                f_hat[name] = None
                sum_l_k[name] = None
                previous_weights[name] = None

        optimizer = CustomSGD(
            model.parameters(),
            is_backbone=list(is_backbone.values()),
            w=lr_modulation_batch_specific_weights,
            f_hat=list(f_hat.values()),
            sum_l_k=list(sum_l_k.values()),
            previous_weights=list(previous_weights.values()),
            lr=learning_rates,
            momentum=momentum,
            weight_decay=l2,
            max_f=max_f,
            xi=xi,
        )
    else:
        f_hat = None
        sum_l_k = None
        previous_weights = None

        optimizer = CustomSGD(
            model.parameters(),
            is_backbone=list(is_backbone.values()),
            lr=learning_rates,
            momentum=momentum,
            weight_decay=l2,
        )

    criterion = torch.nn.CrossEntropyLoss()

    # Iterate through batches and epochs
    print("\nStarting the training loop...\n")
    current_epoch = 0

    if epochs_per_batch < 1:
        # Set data loader parameters
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
            session_name=session_name,
            save_dir_path=save_path,
            model_saving_frequency=model_saving_frequency,
            current_scenario=current_scenario,
            current_run=current_run,
            num_blocks=num_blocks,
            populate_rm_epochs=populate_rm_epochs,
            category_based_split=category_based_split,
            device=device,
            profiling_activated=profiling_activated,
            should_validate=should_validate,
            validation_batch=validation_batch,
        )
    else:
        for batch_counter, current_batch in enumerate(batches):
            for e in range(epochs_per_batch):
                current_epoch += 1

                # Set data loader parameters
                data_loader.update_batch(current_batch)
                data_loader.idx = 0

                if current_scenario in PIPELINES_WITH_RM:
                    # Find classes occurring in the current batch (and their occurrence count)
                    cur = data_loader.get_classes_in_current_batch()

                    # Update temporary weights (tw)
                    model.fc.weight.data.zero_()
                    model.fc.bias.data.zero_()

                    for j in cur.keys():
                        model.fc.weight.data[j] = cw[j]
                        model.fc.bias.data[j] = cb[j]

                    # Freeze backbone if required
                    if data_loader.batches_so_far in initial_batches:
                        model.set_backbone_requires_grad(True)
                    elif current_scenario in CWR_STAR_PIPELINES:
                        model.set_backbone_requires_grad(False)

                        optimizer.is_backbone = [
                            is_backbone[name]
                            for name, el in model.named_parameters()
                            if el.requires_grad
                        ]
                    else:
                        if current_scenario in LR_PIPELINES:
                            model.set_backbone_requires_grad(
                                False, only_before_lr_layer=True
                            )

                            optimizer.is_backbone = [
                                is_backbone[name]
                                for name, el in model.named_parameters()
                                if el.requires_grad
                            ]

                            if current_scenario in AR1_STAR_PURE_PIPELINES:
                                optimizer.f_hat = [
                                    f_hat[name]
                                    for name, el in model.named_parameters()
                                    if el.requires_grad
                                ]

                                optimizer.sum_l_k = [
                                    sum_l_k[name]
                                    for name, el in model.named_parameters()
                                    if el.requires_grad
                                ]

                                optimizer.previous_weights = [
                                    previous_weights[name]
                                    for name, el in model.named_parameters()
                                    if el.requires_grad
                                ]
                else:
                    cur = None

                # Run epoch
                vit_lr_epoch(
                    model=model,
                    data_loader=data_loader,
                    optimizer=optimizer,
                    criterion=criterion,
                    current_epoch=current_epoch,
                    total_epochs=epochs_per_batch * len(batches),
                    mini_batch_size=mini_batch_size,
                    session_name=session_name,
                    save_dir_path=save_path,
                    model_saving_frequency=model_saving_frequency,
                    current_scenario=current_scenario,
                    current_run=current_run,
                    num_blocks=num_blocks,
                    populate_rm_epochs=populate_rm_epochs,
                    category_based_split=category_based_split,
                    device=device,
                    profiling_activated=profiling_activated,
                    should_validate=should_validate,
                    validation_batch=validation_batch,
                )

                if current_scenario in PIPELINES_WITH_RM:
                    print("\nUpdating consolidated weights...\n")
                    # Update consolidated weights
                    for j in cur.keys():
                        w_past[j] = torch.sqrt(past[j] / cur[j])

                        cw[j] = (
                            cw[j] * w_past[j]
                            + (
                                model.fc.weight.data[j]
                                - torch.mean(model.fc.weight.data)
                            )
                        ) / (w_past[j] + 1)
                        cb[j] = (
                            cb[j] * w_past[j]
                            + (model.fc.bias.data[j] - torch.mean(model.fc.bias.data))
                        ) / (w_past[j] + 1)

                        past[j] += cur[j]

                    # Update AR1* parameters
                    if current_scenario in AR1_STAR_PURE_PIPELINES and (
                        (batch_counter < (len(batches) - 1))
                        and current_batch != batches[batch_counter + 1]
                    ):
                        print("\nUpdating AR1* parameters...\n")
                        optimizer.update_a1_star_params(
                            batch_counter,
                            debug_mode=data_loader_debug_mode,
                            session_name=session_name,
                            device=device,
                        )

                    # Save cwr parameters
                    if (
                        not data_loader.debug_mode
                        and model_saving_frequency > 0
                        and (current_epoch % model_saving_frequency == 0)
                    ):
                        print("\nSaving CWR* parameters...\n")
                        torch.save(
                            {
                                "cw": cw,
                                "cb": cb,
                                "cur": cur,
                                "past": past,
                                "w_past": w_past,
                            },
                            os.path.join(
                                save_path, "cwr_e" + str(current_epoch) + ".pth"
                            ),
                        )

    if data_loader_debug_mode:
        return None

    # Update and save final model
    print("\nSaving final model...\n")

    if current_scenario in PIPELINES_WITH_RM:
        model.fc.weight.data = cw
        model.fc.bias.data = cb

    torch.save(
        {
            "epoch": current_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        os.path.join(save_path, "final.pth"),
    )

    return None
