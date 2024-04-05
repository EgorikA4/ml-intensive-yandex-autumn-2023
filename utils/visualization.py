"""A module that visualizes graphs and predicted values."""
from typing import Any

import numpy as np
import torchshow as ts
from matplotlib import pyplot as plt
from torch import Tensor, no_grad, stack, argmax


def plot_loss(loss: list[float], title: str, num_epochs: int) -> None:
    """Draw a loss graphic.

    Args:
        loss (list[float]): the history of the loss of the model.
        title (str): name of the graphic.
        num_epochs (int): the number of past epochs.
    """
    plt.title(title)
    plt.plot(loss)
    plt.grid()
    plt.xticks(np.arange(num_epochs))


def show_losses(train_hist: list[float], val_hist: list[float], num_epochs: int) -> None:
    """Plot test and validation loss charts.

    Args:
        train_hist (list[float]): the history of the training loss.
        val_hist (list[float]): the history of the validation loss.
        num_epochs (int): the number of past epochs.
    """
    plt.figure(figsize=(16, 4))
    plt.subplot(1, 2, 1)
    plot_loss(train_hist, f'Train Loss = {train_hist[-1]}', num_epochs)
    plt.subplot(1, 2, 2)
    plot_loss(val_hist, f'Val Loss = {val_hist[-1]}', num_epochs)
    plt.show()


def show_images(
    model: Any,
    targets: tuple[list[Tensor], list[int | Tensor]],
    device: Any,
    mode: str,
    num_of_examples: int = 4,
) -> None:
    """Output the original images and predicted by the model.

    Args:
        model (Any): the trained model.
        targets (tuple[list[Tensor], list[int | Tensor]]): validation dataset.
        device (Any): the device on which the model is trained.
        num_of_examples (int): number of output images. By default 4.
        mode (str): image output mode ('classification', 'segmentation').
    """
    __classes = {
        0: 'Normal',
        1: 'Infection',
        2: 'COVID-19',
    }
    model.eval()
    with no_grad():
        idxs = np.random.randint(0, len(targets), num_of_examples)

        if mode == 'segmentation':
            images, true_ans = [], []
            for id in idxs:
                images.append(targets[id][0])
                true_ans.append(targets[id][1])

            images = stack(images).to(device)
            true_ans = stack(true_ans).to(device)

            pred = model(images)
            ts.show(true_ans, nrows=1, figsize=(12, 2))
            ts.show(pred, nrows=1, figsize=(12, 2))
        else:
            for ind, id in enumerate(idxs):
                img, true_ans = targets[id]
                pred = argmax(model(true_ans.to(device)))
                plt.subplot(1, num_of_examples, ind + 1)
                plt.imshow(img.permute(1, 2, 0), cmap='gray')
                plt.xticks([])
                plt.yticks([])
                plt.title(f'True = {__classes[true_ans]}\nPred = {__classes[pred]}', fontsize=12)
            plt.show()
