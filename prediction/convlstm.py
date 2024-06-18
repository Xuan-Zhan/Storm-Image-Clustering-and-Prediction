import torch.nn as nn
import torch
from .convlstmcell import ConvLSTMCell


class ConvLSTM(nn.Module):
    """
    A Convolutional LSTM (ConvLSTM) module for processing sequences of images.

    This module encapsulates a ConvLSTM cell for processing
    single time-step images. It is designed to handle sequences
    by unrolling the ConvLSTM cell across time steps.

    Attributes:
        frame_size (tuple):
            The size (height, width) of the input images.
        out_channels (int):
            The number of output channels of the ConvLSTM cell.
        convLSTMcell (ConvLSTMCell):
            The ConvLSTM cell used for processing the input.
    Args:
        in_channels (int):
            The number of channels in the input image.
        out_channels (int):
            The number of output channels.
        kernel_size (tuple):
            The size of the filter kernel, (height, width).
        padding (int):
            The padding added to all four sides of the input.
        activation (str):
            The activation function to use.
        frame_size (tuple):
            The size (height, width) of the input images.
    """
    def __init__(self, in_channels, out_channels,
                 kernel_size, padding, activation, frame_size):
        """
        Initializes the ConvLSTM module with a ConvLSTM cell.

        Parameters
        ----------
        in_channels: int
            The number of channels in the input image.

        out_channels: int
            The number of output channels.

        kernel_size: tuple
            The size of the filter kernel, (height, width).

        padding: int
            The padding added to all four sides of the input.

        activation: string
            The activation function to use.

        frame_size: tuple
            The size (height, width) of the input images.
        """
        super(ConvLSTM, self).__init__()

        self.frame_size = frame_size
        self.out_channels = out_channels

        # We will unroll this over time steps
        self.convLSTMcell = ConvLSTMCell(in_channels, out_channels,
                                         kernel_size, padding, activation,
                                         frame_size)

    def forward(self, X):
        """
        Forward pass of the ConvLSTM module.

        Processes a single image through the ConvLSTM cell
        and returns the hidden state.

        Parameters
        ----------
        X: torch.Tensor
            The input tensor of shape
            (batch_size, num_channels, height, width).

        Return
        ------
        torch.Tensor
            The hidden state output by the ConvLSTM cell of shape
            (batch_size, out_channels, height, width).
        """
        # X is a single image (batch_size, num_channels, height, width)
        # Initialize Hidden State and Cell State
        H = torch.zeros(X.size(0), self.out_channels, *self.frame_size, device=X.device) # noqa
        C = torch.zeros(X.size(0), self.out_channels, *self.frame_size, device=X.device) # noqa

        # Forward pass through the ConvLSTMCell
        H, C = self.convLSTMcell(X, H, C)

        return H
