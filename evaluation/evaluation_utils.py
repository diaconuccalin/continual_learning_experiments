import os

import matplotlib.pyplot as plt
import numpy as np
import torch


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
    res = ax.matshow(conf_mat, cmap="Greens")

    # Show colorbar
    fig.subplots_adjust(right=2.0)
    cb = fig.colorbar(res, shrink=0.7, pad=0.05)
    cb.set_ticks([])

    # Show grid between categories
    ax.set_xticks(np.arange(-0.5, num_classes + 0.5, step), minor=True)
    ax.set_yticks(np.arange(-0.5, num_classes + 0.5, step), minor=True)
    ax.grid(which="minor", color="gainsboro", linestyle="-")

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

    # Write true/predicted axis labels on right and bottom
    ax.set_xlabel("Predicted")
    ax.yaxis.set_label_position("right")
    ax.set_ylabel("True")

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

    # Add title at bottom
    plt.title("Confusion matrix", y=-0.19)

    # Make sure nothing of the plot is cut off
    plt.tight_layout()

    # Save plot
    plt.savefig(save_location)


def plot_losses(losses, save_location):
    # Check if losses are available
    if losses is None:
        return None

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


def plot_ar1_star_f_hat(all_f_hat, current_batch, session_name):
    # Generate histogram
    hist = torch.histc(all_f_hat)

    # Extract minimum and maximum values
    mn = all_f_hat.min().item()
    mx = all_f_hat.max().item()

    # Clear previous plots
    plt.clf()

    # Bar plot histogram
    plt.bar(
        np.arange(mn, mx + 0.001 * mx, (mx - mn) / 99),
        hist.cpu(),
        width=(mx - mn) / 150,
    )

    # Change y-scale to logarithmic
    plt.yscale("log")

    # Adapt ticks to existent values
    plt.xticks(
        np.round(np.arange(mn, mx + 0.1 * mx, (mx - mn) / 10), decimals=6), fontsize=5
    )

    # Save figure
    root_path = os.path.join("evaluation_results", session_name, "plots")
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    plt.savefig(os.path.join(root_path, f"f_hat_{current_batch}.png"))
