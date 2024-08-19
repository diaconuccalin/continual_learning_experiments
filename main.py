import argparse

import torch

from datasets.core50.constants import NI_TRAINING_BATCHES
from evaluation.vit_lr_evaluation_loop import vit_lr_evaluation_pipeline
from training.vit_lr_training_loop import vit_lr_training_pipeline

if __name__ == "__main__":
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
    args = parser.parse_args()

    session_name = args.session_name
    pipeline = args.pipeline
    weights_path = args.weights_path

    assert pipeline in [
        "vit_lr_naive_finetune",
        "vit_lr_core50_evaluation",
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
    if pipeline == "vit_lr_naive_finetune":
        vit_lr_training_pipeline(
            batches=NI_TRAINING_BATCHES,
            epochs_per_batch=1,
            initial_lr=0.1,
            momentum=0.9,
            l2=0.0005,
            input_image_size=(384, 384),
            current_task="ni",
            current_run=0,
            num_layers=12,
            device=device,
            pretrained_weights_path="weights/pretrained_imagenet/B_16_imagenet1k.pth",
            session_name=session_name,
        )
    elif pipeline == "vit_lr_core50_evaluation":
        accuracy = vit_lr_evaluation_pipeline(
            input_image_size=(384, 384),
            current_task="ni",
            current_run=0,
            num_layers=12,
            weights_path=weights_path,
            device=device,
        )

        print("OBTAINED ACCURACY: %0.3f" % (accuracy * 100) + "%")
