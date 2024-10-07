import argparse
import os

import torch

from datasets.core50.constants import (
    NI_TRAINING_BATCHES,
    NC_TRAINING_BATCHES,
    NIC_CUMULATIVE_TRAINING_BATCHES,
    CORE50_CATEGORY_NAMES,
    NI_TESTING_BATCH,
    NC_TESTING_BATCH,
    NIC_CUMULATIVE_TESTING_BATCH,
    NI_BATCH_SPECIFIC_WEIGHTS,
    NC_BATCH_SPECIFIC_WEIGHTS,
    NIC_BATCH_SPECIFIC_WEIGHTS,
    AR1_STAR_LEARNING_RATES,
    CWR_STAR_LEARNING_RATES,
    AR1_STAR_FREE_LEARNING_RATES,
    NI_POPULATE_RM_EPOCHS,
    NC_POPULATE_RM_EPOCHS,
    NIC_POPULATE_RM_EPOCHS,
)
from evaluation.evaluation_utils import plot_confusion_matrix, plot_losses
from evaluation.vit_lr_evaluation_loop import vit_lr_evaluation_pipeline
from training.PipelineScenario import PipelineScenario
from training.training_utils import CONSTANT_TRAINING_PARAMETERS
from training.vit_lr_training_loops import vit_training_pipeline


def create_arg_parser():
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("session_name", help="Name to be used when saving the weights.")
    parser.add_argument(
        "pipeline",
        help="The pipeline to be run. One of: vit_lr_naive_finetune, vit_lr_core50_evaluation.",
    )
    parser.add_argument(
        "--current_task",
        help="The task to be used.",
        required=True,
    )
    parser.add_argument(
        "--weights_path", help="Path to the trained model weights.", required=False
    )
    parser.add_argument(
        "--profile",
        help="Activates profiling, defaults to False.",
        action="store_true",
        required=False,
        default=False,
    )
    parser.add_argument(
        "--rehearsal_memory_size",
        "-rm_size",
        help="Defines the number of patterns to be stored in the rehearsal memory. "
        + "More values can be passed using the format val1,val2,val3...",
        type=lambda arg: list(map(int, arg.split(","))),
        required=False,
        default=[0],
    )
    parser.add_argument(
        "--runs",
        help="Defines the runs (different shuffles of the batches) to be used. "
        + "More values can be passed using the format val1,val2,val3...",
        type=lambda arg: list(map(int, arg.split(","))),
        required=False,
        default=[0],
    )
    parser.add_argument(
        "--do_validation",
        "-val",
        help="Activates validation step after each epoch, defaults to False.",
        action="store_true",
        required=False,
        default=False,
    )
    parser.add_argument(
        "--data_loader_debug_mode",
        help="Activates the data loader debug mode, defaults to False.",
        action="store_true",
        required=False,
        default=False,
    )

    return parser


def vit_demo_naive_finetune(
    device,
    session_name,
    category_based_split,
    profiling_activated,
    current_task,
    should_validate,
):
    vit_training_pipeline(
        current_scenario=PipelineScenario.NATIVE_REHEARSAL,
        batches=NI_TRAINING_BATCHES,
        populate_rm_epochs=NI_POPULATE_RM_EPOCHS,
        initial_batches=[
            0,
        ],
        current_task=current_task,
        mini_batch_size=32,
        rehearsal_memory_size=0,
        device=device,
        session_name=session_name,
        trainable_backbone=True,
        randomize_data_order=True,
        category_based_split=category_based_split,
        profiling_activated=profiling_activated,
        should_validate=should_validate,
        validation_batch=NI_TESTING_BATCH,
        **CONSTANT_TRAINING_PARAMETERS,
    )


def vit_lr_core50_evaluation(device, weights_path, category_based_split, current_task):
    # Choose batch
    if current_task == "ni":
        batch = NI_TESTING_BATCH
    elif current_task in ["nc", "multi-task-nc"]:
        batch = NC_TESTING_BATCH
    elif current_task in ["nic", "nicv2_391"]:
        batch = NIC_CUMULATIVE_TESTING_BATCH
    else:
        raise ValueError("Invalid task name!")

    losses, accuracy, conf_mat = vit_lr_evaluation_pipeline(
        batch=batch,
        input_image_size=(384, 384),
        current_task=current_task,
        current_run=0,
        num_layers=12,
        weights_path=weights_path,
        device=device,
        category_based_split=category_based_split,
    )

    # Prepare accuracy save path
    saving_path_accuracy = weights_path.replace(".pth", "_evaluation_results.txt")
    saving_path_accuracy = saving_path_accuracy.replace(
        "weights/", "evaluation_results/"
    )

    # Prepare confusion matrix save path
    saving_path_conf_mat = weights_path.replace(".pth", "_confusion_matrix.png")
    saving_path_conf_mat = saving_path_conf_mat.replace(
        "weights/", "evaluation_results/"
    )

    # Prepare losses plot save path
    saving_path_losses_plot = weights_path.replace(".pth", "_losses.png")
    saving_path_losses_plot = saving_path_losses_plot.replace(
        "weights/", "evaluation_results/"
    )

    # Create directories if they do not exist
    path_split = saving_path_accuracy.split(os.sep)
    for idx in range(len(path_split) - 1):
        if not os.path.exists(os.sep.join(path_split[: idx + 1])):
            os.mkdir(os.sep.join(path_split[: idx + 1]))

    # Write results to files
    with open(saving_path_accuracy, "w") as f:
        f.write("OBTAINED ACCURACY: %0.3f" % (accuracy * 100) + "%\n")
    plot_confusion_matrix(
        conf_mat=conf_mat,
        labels=CORE50_CATEGORY_NAMES,
        category_based_split=category_based_split,
        save_location=saving_path_conf_mat,
    )
    plot_losses(losses, save_location=saving_path_losses_plot)

    print("Evaluation successfully completed!")


def native_cumulative_train(
    device,
    session_name,
    profiling_activated,
    data_loader_debug_mode,
    current_task,
    runs,
    should_validate=False,
):
    # Choose batches
    if current_task == "ni":
        batches = NI_TRAINING_BATCHES
        validation_batch = NI_TESTING_BATCH
        populate_rm_batches = NI_POPULATE_RM_EPOCHS
    elif current_task in ["nc", "multi-task-nc"]:
        batches = NC_TRAINING_BATCHES
        validation_batch = NC_TESTING_BATCH
        populate_rm_batches = NC_POPULATE_RM_EPOCHS
    elif current_task in ["nic", "nicv2_391"]:
        batches = NIC_CUMULATIVE_TRAINING_BATCHES
        validation_batch = NIC_CUMULATIVE_TESTING_BATCH
        populate_rm_batches = NIC_POPULATE_RM_EPOCHS
    else:
        raise ValueError("Invalid task name!")

    for current_run in runs:
        print(
            "--------------- Starting run",
            current_run,
            "---------------",
        )

        vit_training_pipeline(
            current_scenario=PipelineScenario.NATIVE_REHEARSAL,
            batches=batches,
            initial_batches=[
                0,
            ],
            current_task=current_task,
            current_run=current_run,
            mini_batch_size=128,
            rehearsal_memory_size=0,
            populate_rm_epochs=populate_rm_batches,
            device=device,
            session_name=session_name + "_run_" + str(current_run),
            model_saving_frequency=1,
            trainable_backbone=True,
            randomize_data_order=True,
            category_based_split=False,
            profiling_activated=profiling_activated,
            data_loader_debug_mode=data_loader_debug_mode,
            should_validate=should_validate,
            validation_batch=validation_batch,
            **CONSTANT_TRAINING_PARAMETERS,
        )


def native_cwr_star_or_ar1_star_free_train(
    device,
    session_name,
    profiling_activated,
    data_loader_debug_mode,
    current_task,
    runs,
    rehearsal_memory_size,
    is_cwr_star=True,
    should_validate=False,
):
    # Choose batches
    if current_task == "ni":
        batches = NI_TRAINING_BATCHES
        validation_batch = NI_TESTING_BATCH
        populate_rm_batches = NI_POPULATE_RM_EPOCHS
    elif current_task in ["nc", "multi-task-nc"]:
        batches = NC_TRAINING_BATCHES
        validation_batch = NC_TESTING_BATCH
        populate_rm_batches = NC_POPULATE_RM_EPOCHS
    elif current_task in ["nic", "nicv2_391"]:
        batches = NIC_CUMULATIVE_TRAINING_BATCHES
        validation_batch = NIC_CUMULATIVE_TESTING_BATCH
        populate_rm_batches = NIC_POPULATE_RM_EPOCHS
    else:
        raise ValueError("Invalid task name!")

    for current_run in runs:
        print(
            "--------------- Starting run",
            current_run,
            "---------------",
        )

        vit_training_pipeline(
            current_scenario=(
                PipelineScenario.CWR_STAR
                if is_cwr_star
                else PipelineScenario.AR1_STAR_FREE
            ),
            batches=batches,
            initial_batches=[
                0,
            ],
            current_task=current_task,
            current_run=current_run,
            mini_batch_size=128,
            learning_rates=(
                CWR_STAR_LEARNING_RATES if is_cwr_star else AR1_STAR_FREE_LEARNING_RATES
            ),
            rehearsal_memory_size=rehearsal_memory_size,
            populate_rm_epochs=populate_rm_batches,
            device=device,
            session_name=session_name + "_run_" + str(current_run),
            model_saving_frequency=40,
            randomize_data_order=True,
            category_based_split=False,
            profiling_activated=profiling_activated,
            data_loader_debug_mode=data_loader_debug_mode,
            should_validate=should_validate,
            validation_batch=validation_batch,
            **CONSTANT_TRAINING_PARAMETERS,
        )


def native_ar1_star_train(
    device,
    session_name,
    profiling_activated,
    data_loader_debug_mode,
    current_task,
    runs,
    rehearsal_memory_size,
    should_validate=False,
):
    # Choose batches
    if current_task == "ni":
        batches = NI_TRAINING_BATCHES
        validation_batch = NI_TESTING_BATCH
        lr_modulation_batch_specific_weights = NI_BATCH_SPECIFIC_WEIGHTS
        populate_rm_batches = NI_POPULATE_RM_EPOCHS
    elif current_task in ["nc", "multi-task-nc"]:
        batches = NC_TRAINING_BATCHES
        validation_batch = NC_TESTING_BATCH
        lr_modulation_batch_specific_weights = NC_BATCH_SPECIFIC_WEIGHTS
        populate_rm_batches = NC_POPULATE_RM_EPOCHS
    elif current_task in ["nic", "nicv2_391"]:
        batches = NIC_CUMULATIVE_TRAINING_BATCHES
        validation_batch = NIC_CUMULATIVE_TESTING_BATCH
        lr_modulation_batch_specific_weights = NIC_BATCH_SPECIFIC_WEIGHTS
        populate_rm_batches = NIC_POPULATE_RM_EPOCHS
    else:
        raise ValueError("Invalid task name!")

    for current_run in runs:
        print(
            "--------------- Starting run",
            current_run,
            "---------------",
        )

        vit_training_pipeline(
            current_scenario=PipelineScenario.AR1_STAR,
            batches=batches,
            initial_batches=[
                0,
            ],
            current_task=current_task,
            current_run=current_run,
            mini_batch_size=128,
            learning_rates=AR1_STAR_LEARNING_RATES,
            rehearsal_memory_size=rehearsal_memory_size,
            populate_rm_epochs=populate_rm_batches,
            device=device,
            session_name=session_name + "_run_" + str(current_run),
            lr_modulation_batch_specific_weights=lr_modulation_batch_specific_weights,
            model_saving_frequency=40,
            randomize_data_order=True,
            category_based_split=False,
            profiling_activated=profiling_activated,
            data_loader_debug_mode=data_loader_debug_mode,
            should_validate=should_validate,
            validation_batch=validation_batch,
            **CONSTANT_TRAINING_PARAMETERS,
        )


def main():
    # Training constants (temporary implementation)
    category_based_split = False

    # Parse arguments
    parser = create_arg_parser()
    args = parser.parse_args()

    session_name = args.session_name
    pipeline = args.pipeline
    weights_path = args.weights_path
    profiling_activated = args.profile
    rehearsal_memory_sizes = args.rehearsal_memory_size
    runs = args.runs
    do_validation = args.do_validation
    data_loader_debug_mode = args.data_loader_debug_mode
    current_task = args.current_task

    # Check if pipeline is supported
    available_pipelines = [
        "vit_demo_naive_finetune",
        "vit_lr_core50_evaluation",
        "native_cumulative_train",
        "native_cwr_star_train",
        "native_ar1_star_train",
        "native_ar1_star_free_train",
    ]
    assert (
        pipeline in available_pipelines
    ), "Pipeline currently not supported. Choose one from: " + str(available_pipelines)

    # Set seed
    torch.manual_seed(42)

    # Set cuda device
    cuda_device = None
    if torch.cuda.is_available():
        if cuda_device is None:
            cuda_device = torch.cuda.device_count() - 1
        device = torch.device("cuda:" + str(cuda_device))
        print("DEVICE SET TO GPU " + str(cuda_device) + "!\n")
    else:
        print("DEVICE SET TO CPU!\n")
        device = torch.device("cpu")

    # Run chosen pipeline
    if pipeline == "vit_demo_naive_finetune":
        vit_demo_naive_finetune(
            device=device,
            session_name=session_name,
            category_based_split=category_based_split,
            profiling_activated=profiling_activated,
            current_task=current_task,
            should_validate=do_validation,
        )
    elif pipeline == "vit_lr_core50_evaluation":
        vit_lr_core50_evaluation(
            device=device,
            weights_path=weights_path,
            category_based_split=category_based_split,
            current_task=current_task,
        )
    elif pipeline == "native_cumulative_train":
        native_cumulative_train(
            device=device,
            session_name=session_name,
            profiling_activated=profiling_activated,
            data_loader_debug_mode=data_loader_debug_mode,
            current_task=current_task,
            runs=runs,
            should_validate=do_validation,
        )
    elif pipeline == "native_cwr_star_train":
        for rehearsal_memory_size in rehearsal_memory_sizes:
            print(
                "--------------- Starting training with rm size of",
                rehearsal_memory_size,
                "---------------",
            )

            native_cwr_star_or_ar1_star_free_train(
                device=device,
                session_name=session_name
                + "_rms_"
                + str(rehearsal_memory_size)
                + "_run_",
                profiling_activated=profiling_activated,
                data_loader_debug_mode=data_loader_debug_mode,
                current_task=current_task,
                runs=runs,
                rehearsal_memory_size=rehearsal_memory_size,
                is_cwr_star=True,
                should_validate=do_validation,
            )
    elif pipeline == "native_ar1_star_train":
        for rehearsal_memory_size in rehearsal_memory_sizes:
            print(
                "--------------- Starting training with rm size of",
                rehearsal_memory_size,
                "---------------",
            )

            native_ar1_star_train(
                device=device,
                session_name=session_name + "_rms_" + str(rehearsal_memory_size),
                profiling_activated=profiling_activated,
                data_loader_debug_mode=data_loader_debug_mode,
                current_task=current_task,
                runs=runs,
                rehearsal_memory_size=rehearsal_memory_size,
                should_validate=do_validation,
            )
    elif pipeline == "native_ar1_star_free_train":
        for rehearsal_memory_size in rehearsal_memory_sizes:
            print(
                "--------------- Starting training with rm size of",
                rehearsal_memory_size,
                "---------------",
            )

            native_cwr_star_or_ar1_star_free_train(
                device=device,
                session_name=session_name + "_rms_" + str(rehearsal_memory_size),
                profiling_activated=profiling_activated,
                data_loader_debug_mode=data_loader_debug_mode,
                current_task=current_task,
                runs=runs,
                rehearsal_memory_size=rehearsal_memory_size,
                is_cwr_star=False,
                should_validate=do_validation,
            )

    return None


if __name__ == "__main__":
    main()
