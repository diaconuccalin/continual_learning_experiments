import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(conf_mat, labels, category_based_split, save_location):
    # Prepare drawing values for confusion matrix
    if category_based_split:
        num_classes = len(labels)
        step = 1
        x_x_label_offset = -0.25
        x_y_label_offset = -0.75
        y_x_label_offset = -0.75
        y_y_label_offset = 0
    else:
        num_classes = 5 * len(labels)
        step = 5
        x_x_label_offset = 1
        x_y_label_offset = -1
        y_x_label_offset = -1
        y_y_label_offset = 2.5

    # Prepare plot
    fig, ax = plt.subplots()

    # Plot confusion matrix
    ax.matshow(conf_mat)

    # Show grid between categories
    ax.set_xticks(np.arange(-0.5, num_classes + 0.5, step), minor=True)
    ax.set_yticks(np.arange(-0.5, num_classes + 0.5, step), minor=True)
    ax.grid(which="minor", color="w", linestyle="-")

    # Hide default ticks
    plt.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        labeltop=False,
        labelbottom=False,
        left=False,
        right=False,
        labelleft=False,
        labelright=False,
    )

    # Write category labels
    for i in range(0, num_classes, step):
        ax.text(
            i + x_x_label_offset,
            x_y_label_offset,
            labels[i // step],
            ha="left",
            va="bottom",
            color="black",
            rotation=45,
        )

        ax.text(
            y_x_label_offset,
            i + y_y_label_offset,
            labels[i // step],
            ha="right",
            va="center",
            color="black",
        )

    # Make sure nothing of the plot is cut off
    plt.tight_layout()

    # Save plot
    plt.savefig(save_location)


def plot_losses(losses, save_location):
    # Reset plot
    plt.clf()

    # Plot losses
    plt.plot(range(len(losses)), losses, linewidth=0.1)

    # Set axis labels
    plt.xlabel("Step")
    plt.ylabel("Loss")

    # Make sure nothing of the plot is cut off
    plt.tight_layout()

    # Save plot
    plt.savefig(save_location)
