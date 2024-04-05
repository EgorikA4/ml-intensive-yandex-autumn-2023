"""Custom Lung Dataset."""
import os
from typing import Any

import dotenv
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from exceptions import IncorrectMode

dotenv.load_dotenv()

TEST_IMAGES_FOLDER = os.getenv('TEST_IMAGES_FOLDER')
TRAIN_IMAGES_FOLDER = os.getenv('TRAIN_IMAGES_FOLDER')
TRAIN_LUNG_MASKS_FOLDER = os.getenv('TRAIN_LUNG_MASKS_FOLDER')
LABEL_FILE = os.getenv('LABEL_FILE')


class IncorrectDatasetMode(IncorrectMode):
    """Exception of incorrect dataset mode."""

    def __init__(self, possible_modes: list[str]) -> None:
        """Possible operating modes of the dataset.

        Args:
            possible_modes (list[str]): list of modes that the dataset supports.
        """
        super().__init__(f'Dataset mode should be one of the list: {possible_modes}')


class CustomLungDataset(Dataset):
    """Custom lung dataset."""

    _possible_modes = 'classification', 'segmentation', 'test'

    def __init__(self, root_dir: str, mode: str, transforms: Any | None = None) -> None:
        """Initialize the dataset.

        Root directory with dataset folders and files,
        mode of using the dataset ('classification', 'segmentation', 'test'),
        dataset transformation.

        Args:
            root_dir (str): directory with dataset folders and files.
            mode (str): classififcation (use to train a lung classifier),\
                segmentation (use for lung mask segmentation training),\
                test (use to predict the final answer).
            transforms (optional): dataset transforms. Defaults to None.
        """
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.transforms = transforms

        # path to test images
        self._path_test_images = os.path.join(self.root_dir, TEST_IMAGES_FOLDER)

        # path to train images
        self._path_train_images = os.path.join(self.root_dir, TRAIN_IMAGES_FOLDER)

        # path to train lung masks
        self._path_train_lung_masks = os.path.join(
            self.root_dir, TRAIN_LUNG_MASKS_FOLDER,
        )

        # path to labels on train images
        self._path_labels = os.path.join(self.root_dir, LABEL_FILE)

    @property
    def mode(self) -> str:
        """A mode of dataset.

        Returns:
            str: dataset mode.
        """
        return self._mode

    @mode.setter
    def mode(self, new_mode: str) -> None:
        """Set a new dataset mode.

        Args:
            new_mode (str): dataset mode (classification, segmentation, test).

        Raises:
            IncorrectDatasetMode: the dataset's operating mode should be one of the possible.
        """
        if new_mode not in self._possible_modes:
            raise IncorrectDatasetMode(self._possible_modes)
        self._mode = new_mode

    def __len__(self) -> int:
        """Get dataset length.

        Returns:
            int: length of dataset depending on mode.
        """
        if self.mode in {'classification', 'segmentation'}:
            return len(os.listdir(self._path_train_images))
        return len(os.listdir(self._path_test_images))

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int | np.ndarray] | np.ndarray:
        """Get a dataset element by index.

        Args:
            idx (int): dataset element index.

        Raises:
            IndexError: index is greater than the dataset length.

        Returns:
            tuple[np.ndarray, int | np.ndarray]: image and answer.
        """
        dataset_len = self.__len__()
        if idx > dataset_len:
            raise IndexError(f'index {idx} should be less than {dataset_len}')

        img_name = f'img_{idx}.png'

        if self.mode == 'classification':
            img = Image.open(os.path.join(self._path_train_images, img_name))
            mask = Image.open(os.path.join(self._path_train_lung_masks, img_name))
            img = Image.composite(img, mask, mask=mask)
            label = pd.read_csv(self._path_labels).iloc[idx, 1]

            if self.transforms:
                img = self.transforms(img)

            # img + mask, label
            return (img, label)

        elif self.mode == 'segmentation':
            img = Image.open(os.path.join(self._path_train_images, img_name))
            mask = Image.open(os.path.join(self._path_train_lung_masks, img_name))

            if self.transforms:
                img = self.transforms(img)
                mask = self.transforms(mask)

            # img, mask
            return (img, mask)

        img = Image.open(os.path.join(self._path_test_images, img_name))
        if self.transforms:
            img = self.transforms(img)

        # img
        return img
