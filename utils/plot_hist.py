"""
Name: plot_history.py
Author: aj-gordon
"""

# import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "Times"


def plot_history(
    hist, save_location: str, show: bool = True, colourmap: str = "plasma"
):
    """
    Function that plots the accuracy and loss history of model during training

    Inputs:
        hist - the history output from model fitting
        save_location (str) - the filename to save the plot to
        show [opt] (bool) - whether to show the plot or just save
        colourmap [opt] (str) - the colour scheme to use in plotting

    """

    cmap = get_cmap(colourmap)
    norm = Normalize(vmin=0, vmax=2)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    # plot the loss metrics
    ax1.plot(hist.history["loss"], label="training", color=cmap(norm(0)))
    ax1.plot(hist.history["val_loss"], label="validation", color=cmap(norm(1)))

    # make it look nice
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Value")
    ax1.legend(loc="best")

    # plot the accuracy metrics
    ax2.plot(hist.history["accuracy"], label="training", color=cmap(norm(0)))
    ax2.plot(hist.history["val_accuracy"], label="validation", color=cmap(norm(1)))

    # make it look nice
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Value")
    ax2.legend(loc="best")

    fig.tight_layout()

    plt.savefig(save_location)

    if show:
        plt.show()
    else:
        plt.close()


def plot_history_v2(
    hist, save_location: str, show: bool = True, colourmap: str = "plasma"
):
    """
    Function that plots the accuracy and loss history of model during training

    Inputs:
        hist - the history output from model fitting
        save_location (str) - the filename to save the plot to
        show [opt] (bool) - whether to show the plot or just save
        colourmap [opt] (str) - the colour scheme to use in plotting

    """

    cmap = get_cmap(colourmap)
    norm = Normalize(vmin=0, vmax=2)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    # plot the loss metrics
    ax1.plot(hist["loss"], label="training", color=cmap(norm(0)))
    ax1.plot(hist["val_loss"], label="validation", color=cmap(norm(1)))

    # make it look nice
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Value")
    ax1.legend(loc="best")

    # plot the accuracy metrics
    ax2.plot(hist["accuracy"], label="training", color=cmap(norm(0)))
    ax2.plot(hist["val_accuracy"], label="validation", color=cmap(norm(1)))

    # make it look nice
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Value")
    ax2.legend(loc="best")

    fig.tight_layout()

    plt.savefig(save_location)

    if show:
        plt.show()
    else:
        plt.close()
