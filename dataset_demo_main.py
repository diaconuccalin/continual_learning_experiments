import numpy as np
from matplotlib import pyplot as plt

from core50.dataset import CORE50


def dataset_demo():
    # Generate dataset object
    dataset = CORE50(
        root="core50/data/core50_128x128",
        preload=False,
        scenario="ni",
        cumul=False,
        demo=True
    )

    # Extract sample
    [x, ], y, _ = dataset.__next__()

    # Display image
    plt.imshow(x.astype(np.uint8), interpolation='nearest')
    plt.show()

    # Print label
    print("Showing image of class", dataset.classnames[int(y)])

    return None


if __name__ == "__main__":
    dataset_demo()
