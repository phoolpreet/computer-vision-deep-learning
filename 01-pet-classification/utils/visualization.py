import os

import matplotlib.pyplot as plt
import numpy as np
from configs import config


def plot_losses(train_losses, val_losses):
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")

    # Find the epoch with the lowest validation loss
    min_val_loss_epoch = np.argmin(val_losses)

    # Highlight the minimum validation loss point
    plt.plot(
        min_val_loss_epoch,
        val_losses[min_val_loss_epoch],
        "ro",
        markersize=8,
        label="Minimum Validation Loss",
    )

    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(config.OUTPUT_PATH, "training-losses.jpg"))


def plot_confusion_matrix():
    pass
