import matplotlib.pyplot as plt
import torch

from datasets.core50.CORE50DataLoader import CORE50DataLoader
from datasets.imagenet.imagenet_1k_class_names import imagenet_classes
from vit_lr.ResizeProcedure import ResizeProcedure
from vit_lr.ViTLR_model import ViTLR
from vit_lr.utils import vit_lr_image_preprocessing


def vit_lr_demo(device, start_batch=0, start_run=0, start_idx=0):
    weight_path = "weigths/B_16_imagenet1k.pth"
    input_image_size = (384, 384)
    num_layers = 12
    k = 10

    # Generate dataset object
    dataset = CORE50DataLoader(
        root="datasets/core50/data",
        original_image_size=(350, 350),
        input_image_size=input_image_size,
        resize_procedure=ResizeProcedure.BORDER,
        channels=3,
        scenario="ni",
        load_entire_batch=False,
        start_batch=start_batch,
        start_run=start_run,
        start_idx=start_idx,
    )

    # Get test image
    x, _ = dataset.__next__()

    # Load model
    model = ViTLR(device=device, input_size=input_image_size, num_layers=num_layers)

    # Load weights
    weights = torch.load(weight_path)

    for i in range(num_layers):
        # torch.eye is an identity matrix
        # Required because the proj_out layer is not present in the default ViT
        weights["transformer.blocks." + str(i) + ".attn.proj_out.weight"] = torch.eye(
            n=model.state_dict()[
                "transformer.blocks." + str(i) + ".attn.proj_out.weight"
            ].shape[0]
        )

    model.load_state_dict(weights)
    print(model, "\n")

    # Display image
    plt.imshow(x[0, ...], interpolation="nearest")
    plt.axis("off")
    plt.show()

    # Preprocess image
    x = vit_lr_image_preprocessing(x)

    # Set device to GPU
    model.to(device)
    x = x.to(device)

    # Perform inference
    y_pred = model(x)[0]

    # Extract top k results
    sm = torch.nn.Softmax(dim=0)
    y_pred = sm(y_pred)
    top_k_pred = torch.topk(y_pred, k)

    for i in range(k):
        print(
            imagenet_classes[top_k_pred.indices[i].item()],
            "-",
            str(top_k_pred.values[i].item() * 100)[:5] + "%",
        )

    return None


def vit_lr_naive_finetune(
    device,
    initial_lr=0.001,
    momentum=0.9,
    l2=0.0005,
    start_batch=0,
    start_run=0,
    start_idx=0,
):
    weight_path = "weigths/B_16_imagenet1k.pth"
    input_image_size = (384, 384)
    num_layers = 12

    # Generate dataset object
    dataset = CORE50DataLoader(
        root="datasets/core50/data",
        original_image_size=(350, 350),
        input_image_size=input_image_size,
        resize_procedure=ResizeProcedure.BORDER,
        channels=3,
        scenario="ni",
        load_entire_batch=True,
        start_batch=start_batch,
        start_run=start_run,
        start_idx=start_idx,
    )

    # Get test image
    x, _ = dataset.__next__()

    # Load model
    model = ViTLR(device=device, input_size=input_image_size, num_layers=num_layers)

    # Load weights
    weights = torch.load(weight_path)

    for i in range(num_layers):
        # torch.eye is an identity matrix
        # Required because the proj_out layer is not present in the default ViT
        weights["transformer.blocks." + str(i) + ".attn.proj_out.weight"] = torch.eye(
            n=model.state_dict()[
                "transformer.blocks." + str(i) + ".attn.proj_out.weight"
            ].shape[0]
        )

        # Mark proj_out as frozen
        model.transformer.blocks[i].attn.proj_out.weight.requires_grad = False

        if model.transformer.blocks[i].attn.proj_out.bias is not None:
            model.transformer.blocks[i].attn.proj_out.bias.requires_grad = False

    model.load_state_dict(weights)

    # Set device to GPU
    model.to(device)

    # Optimizer setup
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=initial_lr,
        momentum=momentum,
        weight_decay=l2,
    )
    criterion = torch.nn.CrossEntropyLoss()

    # Iterate through batches
    for i, train_batch in enumerate(dataset):
        train_x, train_y = train_batch

        print(train_x.shape)
        print(train_y.shape)
        print(i)
        print()

    return None


if __name__ == "__main__":
    torch.manual_seed(42)

    if torch.cuda.is_available():
        print("DEVICE SET TO GPU!\n")
        device = torch.device("cuda")
    else:
        print("DEVICE SET TO CPU!\n")
        device = torch.device("cpu")

    # vit_lr_demo(start_batch=1, start_run=4, start_idx=400, device=device)
    vit_lr_naive_finetune(device=device)
