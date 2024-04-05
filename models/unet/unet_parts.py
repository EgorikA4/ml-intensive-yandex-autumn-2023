"""Parts of the Unet model."""
import numpy as np
from torch import cat, nn


class DoubleConv(nn.Module):
    """(Conv2d -> BatchNorm -> ReLU) * 2."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize double counvolutional layers.

        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
        """
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_in: np.ndarray) -> np.ndarray:
        """Forward pass through the model.

        Args:
            x_in (np.ndarray): the input tensor.

        Returns:
            np.ndarray: the output tensor of the model.
        """
        return self.double_conv(x_in)


class Down(nn.Module):
    """Downscaling with maxpool then DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize Downscaling layer that is based on the MaxPool2d.

        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
        """
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, x_in: np.ndarray) -> np.ndarray:
        """Forward pass through the model.

        Args:
            x_in (np.ndarray): the input tensor.

        Returns:
            np.ndarray: the output tensor of the model.
        """
        return self.double_conv(self.maxpool(x_in))


class Up(nn.Module):
    """Upscaling then double DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize Upscaling layer that is based on the Upsample.

        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
        """
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """Forward pass through the model.

        Args:
            x1 (np.ndarray): the input tensor of the previous layer.
            x2 (np.ndarray): the input tensor of the symmetric layer.

        Returns:
            np.ndarray: the output tensor of the model.
        """
        x1 = self.upsample(x1)
        # the difference between the sizes in layers
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # add additional paddings
        x1 = nn.functional.pad(
            x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
        )

        # combining two layers
        x_out = cat([x2, x1], dim=1)
        return self.double_conv(x_out)
