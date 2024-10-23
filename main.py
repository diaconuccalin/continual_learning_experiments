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
    NIC_SINGLE_CUMULATIVE_TRAINING_BATCHES,
)
from evaluation.evaluation_utils import plot_confusion_matrix, plot_losses
from evaluation.vit_lr_evaluation_loop import vit_lr_evaluation_pipeline
from training.PipelineScenario import (
    PipelineScenario,
    PIPELINES_WITH_RM,
    PIPELINES_WITH_LR_MODULATION,
    CWR_STAR_PIPELINES,
    AR1_STAR_PURE_PIPELINES,
    AR1_STAR_FREE_PIPELINES,
    LR_PIPELINES,
)
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
        "--n_blocks",
        help="Defines the number of transformer blocks to be included in the ViT model. "
        + "Defaults to 12 and accepts integer values between 1 and 12. "
        + "More values can be passed using the format val1,val2,val3...",
        type=lambda arg: list(map(int, arg.split(","))),
        required=False,
        default=[12],
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
    parser.add_argument(
        "--latent_replay_layer",
        "-lr_layer",
        help="Defines the index of the transformer block to be used as latent replay layer. "
        + "Defaults to -1, which is allowed in all non-AR1* [free] pipelines, that have a predetermined LR layer.",
        type=int,
        required=False,
        default=-1,
    )

    return parser


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


def vit_lr_train(
    device,
    session_name,
    profiling_activated,
    data_loader_debug_mode,
    current_task,
    current_scenario,
    n_blocks,
    runs,
    rehearsal_memory_size=0,
    should_validate=False,
    latent_replay_layer=None,
):
    # Choose batches
    populate_rm_batches = None
    lr_modulation_batch_specific_weights = None
    if current_task == "ni":
        batches = NI_TRAINING_BATCHES
        validation_batch = NI_TESTING_BATCH

        if current_scenario in PIPELINES_WITH_RM:
            populate_rm_batches = NI_POPULATE_RM_EPOCHS
        if current_scenario in PIPELINES_WITH_LR_MODULATION:
            lr_modulation_batch_specific_weights = NI_BATCH_SPECIFIC_WEIGHTS
    elif current_task in ["nc", "multi-task-nc"]:
        batches = NC_TRAINING_BATCHES
        validation_batch = NC_TESTING_BATCH

        if current_scenario in PIPELINES_WITH_RM:
            populate_rm_batches = NC_POPULATE_RM_EPOCHS
        if current_scenario in PIPELINES_WITH_LR_MODULATION:
            lr_modulation_batch_specific_weights = NC_BATCH_SPECIFIC_WEIGHTS
    elif current_task in ["nic", "nicv2_391"]:
        if current_scenario == PipelineScenario.NATIVE_CUMULATIVE:
            batches = NIC_SINGLE_CUMULATIVE_TRAINING_BATCHES
        else:
            batches = NIC_CUMULATIVE_TRAINING_BATCHES
        validation_batch = NIC_CUMULATIVE_TESTING_BATCH

        if current_scenario in PIPELINES_WITH_RM:
            populate_rm_batches = NIC_POPULATE_RM_EPOCHS
        if current_scenario in PIPELINES_WITH_LR_MODULATION:
            lr_modulation_batch_specific_weights = NIC_BATCH_SPECIFIC_WEIGHTS
    else:
        raise ValueError("Invalid task name!")

    if current_scenario != PipelineScenario.NATIVE_CUMULATIVE:
        model_saving_frequency = 40
    else:
        model_saving_frequency = 1

    if current_scenario in CWR_STAR_PIPELINES:
        learning_rates = CWR_STAR_LEARNING_RATES
    elif current_scenario in AR1_STAR_PURE_PIPELINES:
        learning_rates = AR1_STAR_LEARNING_RATES
    elif current_scenario in AR1_STAR_FREE_PIPELINES:
        learning_rates = AR1_STAR_FREE_LEARNING_RATES
    else:
        learning_rates = [
            (0.01, 0.01),
        ] * len(batches)

    if current_scenario == PipelineScenario.LR_CWR_STAR:
        latent_replay_layer = 11
    elif current_scenario in LR_PIPELINES:
        assert (
            latent_replay_layer is not None
        ), "Latent replay layer must be set for current scenario."
    else:
        latent_replay_layer = -1

    for current_run in runs:
        print(
            "--------------- Starting run",
            current_run,
            "---------------",
        )

        for num_blocks in n_blocks:
            print(
                "--------------- Starting training with",
                num_blocks,
                "blocks ---------------",
            )

            vit_training_pipeline(
                current_scenario=current_scenario,
                batches=batches,
                initial_batches=[0],
                current_task=current_task,
                current_run=current_run,
                num_blocks=num_blocks,
                mini_batch_size=128,
                epochs_per_batch=(
                    -1 if current_scenario is PipelineScenario.NATIVE_CUMULATIVE else 1
                ),
                rehearsal_memory_size=rehearsal_memory_size,
                learning_rates=learning_rates,
                populate_rm_epochs=populate_rm_batches,
                device=device,
                session_name=session_name
                + "_run_"
                + str(current_run)
                + "_blocks_"
                + str(num_blocks),
                lr_modulation_batch_specific_weights=lr_modulation_batch_specific_weights,
                model_saving_frequency=model_saving_frequency,
                randomize_data_order=True,
                category_based_split=False,
                profiling_activated=profiling_activated,
                data_loader_debug_mode=data_loader_debug_mode,
                should_validate=should_validate,
                validation_batch=validation_batch,
                latent_replay_layer=latent_replay_layer,
                **CONSTANT_TRAINING_PARAMETERS,
            )


def main():
    # Parse arguments
    parser = create_arg_parser()
    args = parser.parse_args()

    session_name = args.session_name
    pipeline = args.pipeline
    weights_path = args.weights_path
    profiling_activated = args.profile
    rehearsal_memory_sizes = args.rehearsal_memory_size
    runs = args.runs
    n_blocks = args.n_blocks
    do_validation = args.do_validation
    data_loader_debug_mode = args.data_loader_debug_mode
    current_task = args.current_task
    latent_replay_layer = args.latent_replay_layer

    # Check if pipeline is supported
    available_pipelines = [
        "core50_evaluation",
        "native_cumulative_train",  # PipelineScenario.NATIVE_CUMULATIVE
        "native_cwr_star_train",  # PipelineScenario.NATIVE_CWR_STAR
        "native_ar1_star_train",  # PipelineScenario.NATIVE_AR1_STAR
        "native_ar1_star_free_train",  # PipelineScenario.NATIVE_AR1_STAR_FREE
        "lr_cwr_star_train",  # PipelineScenario.LR_CWR_STAR
        "lr_ar1_star_train",  # PipelineScenario.LR_AR1_STAR
        "lr_ar1_star_free_train",  # PipelineScenario.LR_AR1_STAR_FREE
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
    if pipeline == "core50_evaluation":
        vit_lr_core50_evaluation(
            device=device,
            weights_path=weights_path,
            category_based_split=False,
            current_task=current_task,
        )
    elif pipeline == "native_cumulative_train":
        vit_lr_train(
            current_scenario=PipelineScenario.NATIVE_CUMULATIVE,
            device=device,
            session_name=session_name,
            profiling_activated=profiling_activated,
            data_loader_debug_mode=data_loader_debug_mode,
            current_task=current_task,
            n_blocks=n_blocks,
            runs=runs,
            should_validate=do_validation,
        )
    else:
        current_scenario = None
        if pipeline == "native_cwr_star_train":
            current_scenario = PipelineScenario.NATIVE_CWR_STAR
        elif pipeline == "native_ar1_star_train":
            current_scenario = PipelineScenario.NATIVE_AR1_STAR
        elif pipeline == "native_ar1_star_free_train":
            current_scenario = PipelineScenario.NATIVE_AR1_STAR_FREE
        elif pipeline == "lr_cwr_star_train":
            current_scenario = PipelineScenario.LR_CWR_STAR
        elif pipeline == "lr_ar1_star_train":
            current_scenario = PipelineScenario.LR_AR1_STAR
        elif pipeline == "lr_ar1_star_free_train":
            current_scenario = PipelineScenario.LR_AR1_STAR_FREE

        for rehearsal_memory_size in rehearsal_memory_sizes:
            print(
                "--------------- Starting training with rm size of",
                rehearsal_memory_size,
                "---------------",
            )

            vit_lr_train(
                current_scenario=current_scenario,
                device=device,
                session_name=session_name + "_rms_" + str(rehearsal_memory_size),
                profiling_activated=profiling_activated,
                data_loader_debug_mode=data_loader_debug_mode,
                current_task=current_task,
                rehearsal_memory_size=rehearsal_memory_size,
                runs=runs,
                n_blocks=n_blocks,
                should_validate=do_validation,
                latent_replay_layer=latent_replay_layer,
            )

    return None


if __name__ == "__main__":
    main()
