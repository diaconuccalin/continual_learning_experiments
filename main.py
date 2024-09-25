import argparse
import os

import numpy as np
import torch

from datasets.core50.constants import (
    NI_TRAINING_BATCHES,
    NC_TRAINING_BATCHES,
    NIC_CUMULATIVE_TRAINING_BATCHES,
)
from evaluation.vit_lr_evaluation_loop import vit_lr_evaluation_pipeline
from training.vit_lr_training_loop import (
    vit_native_rehearsal_training_pipeline,
)


def create_arg_parser():
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("session_name", help="Name to be used when saving the weights.")
    parser.add_argument(
        "pipeline",
        help="The pipeline to be run. One of: vit_lr_naive_finetune, vit_lr_core50_evaluation.",
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
        "--exemplar_set_ratio",
        help="Defines the ratio of the entire training set to be stored as exemplars.",
        required=False,
        default=0.0,
    )
    parser.add_argument(
        "--data_loader_debug_mode",
        help="Activates the data loader debug mode, defaults to False.",
        action="store_true",
        required=False,
        default=False,
    )

    return parser


def vit_demo_naive_finetune():
    vit_native_rehearsal_training_pipeline(
        batches=NI_TRAINING_BATCHES,
        epochs_per_batch=10,
        initial_lr=0.01,
        momentum=0.9,
        l2=0.0005,
        input_image_size=(384, 384),
        current_task="ni",
        current_run=0,
        num_layers=12,
        mini_batch_size=32,
        exemplar_set_ratio=0.0,
        device=device,
        pretrained_weights_path="weights/pretrained_imagenet/B_16_imagenet1k.pth",
        session_name=session_name,
        trainable_backbone=True,
        randomize_data_order=True,
        category_based_split=category_based_split,
        profiling_activated=profiling_activated,
    )


def vit_lr_core50_evaluation():
    accuracy, conf_mat = vit_lr_evaluation_pipeline(
        input_image_size=(384, 384),
        current_task="ni",
        current_run=0,
        num_layers=12,
        weights_path=weights_path,
        device=device,
        category_based_split=category_based_split,
    )

    # Extract the string up to the file name from weights_path
    saving_path_accuracy = weights_path.replace(".pth", "_evaluation_results.txt")
    saving_path_accuracy = saving_path_accuracy.replace(
        "weights/", "evaluation_results/"
    )

    saving_path_conf_mat = weights_path.replace(".pth", "_confusion_matrix.txt")
    saving_path_conf_mat = saving_path_conf_mat.replace(
        "weights/", "evaluation_results/"
    )

    path_split = saving_path_accuracy.split(os.sep)
    for idx in range(len(path_split) - 1):
        if not os.path.exists(os.sep.join(path_split[: idx + 1])):
            os.mkdir(os.sep.join(path_split[: idx + 1]))

    with open(saving_path_accuracy, "w") as f:
        f.write("OBTAINED ACCURACY: %0.3f" % (accuracy * 100) + "%\n")
    np.savetxt(saving_path_conf_mat, conf_mat)

    print("Evaluation successfully completed!")


def vit_rehearsal_train():
    vit_native_rehearsal_training_pipeline(
        batches=NC_TRAINING_BATCHES,
        epochs_per_batch=1,
        initial_lr=0.01,
        momentum=0.9,
        l2=0.0005,
        input_image_size=(384, 384),
        current_task="multi-task-nc",
        current_run=0,
        num_layers=12,
        mini_batch_size=32,
        exemplar_set_ratio=exemplar_set_ratio,
        device=device,
        pretrained_weights_path="weights/pretrained_imagenet/B_16_imagenet1k.pth",
        session_name=session_name,
        trainable_backbone=True,
        randomize_data_order=True,
        category_based_split=category_based_split,
        profiling_activated=profiling_activated,
        data_loader_debug_mode=data_loader_debug_mode,
    )


def latent_replay_native_cumulative():
    vit_native_rehearsal_training_pipeline(
        batches=NIC_CUMULATIVE_TRAINING_BATCHES,
        epochs_per_batch=1,
        initial_lr=0.01,
        momentum=0.9,
        l2=0.0005,
        input_image_size=(384, 384),
        current_task="nic",
        current_run=0,
        num_layers=12,
        mini_batch_size=0,
        exemplar_set_ratio=0.0,
        device=device,
        pretrained_weights_path="weights/pretrained_imagenet/B_16_imagenet1k.pth",
        session_name=session_name,
        trainable_backbone=True,
        randomize_data_order=True,
        category_based_split=False,
        profiling_activated=profiling_activated,
        data_loader_debug_mode=data_loader_debug_mode,
    )


if __name__ == "__main__":
    # Training constants (temporary implementation)
    category_based_split = False

    # Parse arguments
    parser = create_arg_parser()
    args = parser.parse_args()

    session_name = args.session_name
    pipeline = args.pipeline
    weights_path = args.weights_path
    profiling_activated = args.profile
    exemplar_set_ratio = float(args.exemplar_set_ratio)
    data_loader_debug_mode = args.data_loader_debug_mode

    # Check if pipeline is supported
    assert pipeline in [
        "vit_demo_naive_finetune",
        "vit_rehearsal_train",
        "vit_lr_core50_evaluation",
        "latent_replay_native_cumulative",
    ], "Pipeline currently not supported."

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
        vit_demo_naive_finetune()
    elif pipeline == "vit_lr_core50_evaluation":
        vit_lr_core50_evaluation()
    elif pipeline == "vit_rehearsal_train":
        vit_rehearsal_train()
    elif pipeline == "latent_replay_native_cumulative":
        latent_replay_native_cumulative()
