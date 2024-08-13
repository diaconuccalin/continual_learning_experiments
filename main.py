import argparse

import torch

from datasets.core50.constants import NI_TRAINING_BATCHES
from training.vit_lr_training_loop import vit_lr_training_pipeline

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("session_name", help="Name to be used when saving the weights.")
    parser.add_argument(
        "pipeline", help="The pipeline to be followed. One of: vit_lr_naive_finetune"
    )
    args = parser.parse_args()

    session_name = args.session_name
    pipeline = args.pipeline

    assert pipeline in ["vit_lr_naive_finetune"], "Pipeline currently not supported."

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
            initial_lr=0.001,
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
