"""Custom implementation of the "Unet" architecture."""
import numpy as np
from torch import nn

from .unet_parts import DoubleConv, Down, Up


class Unet(nn.Module):
    """Custom Unet architecture."""

    __scales = 64, 128, 256, 512, 1024

    def __init__(self, n_channels: np.ndarray, n_classes: np.ndarray) -> None:
        """Initialize the model with the specified input channels.

        Args:
            n_channels (int): number of image channels.
            n_classes (int): number of output classes.
        """
        super().__init__()
        self._inc = DoubleConv(n_channels, self.__scales[0])

        self._down1 = Down(self.__scales[0], self.__scales[1])
        self._down2 = Down(self.__scales[1], self.__scales[2])
        self._down3 = Down(self.__scales[2], self.__scales[3])
        self._down4 = Down(self.__scales[3], self.__scales[4] // 2)

        self._up1 = Up(self.__scales[4], self.__scales[3] // 2)
        self._up2 = Up(self.__scales[3], self.__scales[2] // 2)
        self._up3 = Up(self.__scales[2], self.__scales[1] // 2)
        self._up4 = Up(self.__scales[1], self.__scales[0])

        self._outc = nn.Conv2d(self.__scales[0], n_classes, 1)

    def forward(self, x_in: np.ndarray) -> np.ndarray:
        """Forward pass through the model.

        Args:
            x_in (np.ndarray): the input tensor.

        Returns:
            np.ndarray: the output tensor of the model.
        """
        x1 = self._inc(x_in)

        x2 = self._down1(x1)
        x3 = self._down2(x2)
        x4 = self._down3(x3)
        x5 = self._down4(x4)

        x_in = self._up1(x5, x4)
        x_in = self._up2(x_in, x3)
        x_in = self._up3(x_in, x2)
        x_in = self._up4(x_in, x1)

        return self._outc(x_in)
