"""Module for training models."""

import copy
import os
from typing import Any

import dotenv
import numpy as np
from IPython.display import clear_output
from sklearn.metrics import f1_score
from torch import cuda, device, nn, save, set_grad_enabled
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from exceptions import IncorrectMode
from utils.dice_loss import dice_coeff
from utils.visualization import show_images, show_losses

dotenv.load_dotenv()

PATH_TO_SAVE = os.getenv('PATH_TO_SAVE', default='')


class IncorrectTrainerMode(IncorrectMode):
    """Incorrect model training mode exception."""

    def __init__(self, mode: str, possible_modes: tuple[str]) -> None:
        """Initialize incorrect model training mode exception.

        Args:
            mode (str): wrong mode.
            possible_modes (tuple[str]): acceptable model training modes.
        """
        super().__init__(f'mode {mode} not in {possible_modes}')


class Trainer:
    """Training different models."""

    __possible_modes = 'classification', 'segmentation'
    _device = device('cuda' if cuda.is_available() else 'cpu')

    def __init__(
        self,
        dataset: Any,
        model: Any,
        optimizer: Any,
        criterion: Any,
        scheduler: Any,
        mode: str,
    ) -> None:
        """Initialize the characteristics for training the model.

        Args:
            dataset (Any): dataset for training.
            model (Any): a model for training.
            optimizer (Any): a function that adjusts the attributes of the neural network.
            criterion (Any): a loss function.
            scheduler (Any): adjusts the hyperparameter of the optimizer.
            mode (str): possible training modes ('classification', 'segmentation')
        """
        self.dataset = dataset
        self.model = model.to(self._device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.mode = mode

    @property
    def mode(self) -> str:
        """Training mode.

        Returns:
            str: the mode in which to train the model.
        """
        return self._mode

    @mode.setter
    def mode(self, new_mode: str) -> None:
        """Set a new training mode.

        Args:
            new_mode (str): new training mode ('classification', 'segmentation').

        Raises:
            IncorrectTrainerMode: incorrect model training mode ('classification', 'segmentation').
        """
        if new_mode not in self.__possible_modes:
            raise IncorrectTrainerMode(new_mode, self.__possible_modes)
        self._mode = new_mode

    def save_model(self, epoch: int, best_score: float) -> None:
        """Save the state of the model.

        Args:
            epoch (int): the current epoch of training.
            best_score (float): the best score value to save.
        """
        best_model_wts = copy.deepcopy(self.model.state_dict())

        data_to_save = {
            'epoch': epoch,
            'model_state_dict': best_model_wts,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_score': best_score,
        }

        if not os.path.exists(PATH_TO_SAVE):
            os.mkdir(PATH_TO_SAVE)

        save(data_to_save, os.path.join(PATH_TO_SAVE, f'best_checkpoint[epoch_{epoch}].pt'))

    def run_epoch(self, dataloader: Any, is_train: bool = True) -> float:
        """Start training on a single epoch.

        Args:
            dataloader (Any): training data.
            is_train (bool): model mode. By default, True.

        Returns:
            float: loss on the epoch.
        """
        self.model.train(is_train)
        total_loss = total_score = 0.
        with set_grad_enabled(is_train):
            for images, true_ans in tqdm(dataloader):
                images = images.to(self._device)
                true_ans = true_ans.to(self._device).float()

                pred = self.model(images)
                loss = self.criterion(pred, true_ans)
                if self.mode == 'segmentation':
                    loss += dice_coeff(nn.functional.sigmoid(pred), true_ans)

                if is_train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                total_loss += loss.item()

                score = f1_score(true_ans, pred, average='macro')
                total_score += score

                self.scheduler.step(score)

        return np.array([total_loss, total_score]) / len(dataloader.dataset)

    def run_train_loop(self, epochs: int, batch_size: int) -> None:
        """Start the full training loop of the model.

        Args:
            epochs (int): number of epochs in training.
            batch_size (int): sample size.
        """
        train_set, val_set = random_split(self.dataset, [0.8, 0.2])

        train_loader = DataLoader(train_set, batch_size)
        val_loader = DataLoader(val_set, batch_size)

        train_hist = []
        val_hist = []

        best_score = 0.

        for epoch in range(1, epochs + 1):
            train_loss, _ = self.run_epoch(train_loader)
            train_hist.append(train_loss)

            val_loss, val_score = self.run_epoch(val_loader, is_train=False)
            val_hist.append(val_loss)

            if PATH_TO_SAVE != '' and val_score < best_score:
                best_score = val_score
                self.save_model(epoch, best_score)

            clear_output()
            show_images(self.model, val_loader.dataset, self.mode, self._device)
            show_losses(train_hist, val_hist, epoch)
