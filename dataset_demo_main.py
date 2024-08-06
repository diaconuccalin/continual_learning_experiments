from matplotlib import pyplot as plt

from datasets.core50.CORE50DataLoader import CORE50DataLoader


def dataset_demo():
    # Generate dataset object
    dataset = CORE50DataLoader(
        root="datasets/core50/data",
        original_image_size=(350, 350),
        input_image_size=(384, 384),
        resize_procedure="border",
        channels=3,
        scenario="ni",
        load_entire_batch=False,
        start_batch=1,
        start_run=4,
        start_idx=400,
    )

    # Extract sample
    x, y = dataset.__next__()

    for idx in range(x.shape[0]):
        # Display image
        plt.imshow(x[idx], interpolation="nearest")
        plt.show()

        # Print label
        print("Showing image of class", dataset.class_names[int(y[idx])])

    return None


if __name__ == "__main__":
    dataset_demo()
