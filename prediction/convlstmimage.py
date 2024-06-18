import torch.nn as nn
import torch
from .convlstmcellimage import ConvLSTMCellImage

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ConvLSTMImage(nn.Module):
    """
    Implements a ConvLSTM module specifically designed for sequences of images.

    This module encapsulates the ConvLSTMCellImage
    to process sequences of images,
    making it suitable for tasks that require understanding spatial
    and temporal dynamics, such as video prediction
    or spatiotemporal sequence forecasting.
    It unrolls the ConvLSTMCellImage over time steps,
    maintaining spatial information within the LSTM's hidden
    and cell states across the sequence.

    Attributes:
        out_channels (int):
            The number of output channels of the ConvLSTM cell.
        convLSTMcell (ConvLSTMCellImage):
            The ConvLSTM cell adapted for image input.

    Args:
        in_channels (int):
            The number of channels in the input images.
        out_channels (int):
            The number of channels in the output feature maps.
        kernel_size (tuple):
            The size of the convolutional kernel.
        padding (int):
            The amount of padding added to the input on all sides.
        activation (str):
            The name of the activation function to use ('tanh' or 'relu').
        frame_size (tuple):
            The spatial dimensions (height, width) of the input images.
    """
    def __init__(self, in_channels, out_channels,
                 kernel_size, padding, activation, frame_size):
        """
        Initializes the ConvLSTMImage module
        with the specified parameters for processing image sequences.

        Parameters
        ----------
        in_channels: int
            The number of channels in the input images.

        out_channels: int
            The number of output feature map channels.

        kernel_size: tuple
            The dimensions of the convolutional kernel.

        padding: int
            Padding to be added to the input images.

        activation: string
            Activation function ('tanh' or 'relu').

        frame_size: tuple
            The size (height, width) of the input images.
        """
        super(ConvLSTMImage, self).__init__()

        self.out_channels = out_channels

        # We will unroll this over time steps
        self.convLSTMcell = ConvLSTMCellImage(in_channels, out_channels,
                                              kernel_size, padding, activation,
                                              frame_size)

    def forward(self, X):
        """
        Forward pass of the ConvLSTMImage module.

        Processes a sequence of images through the ConvLSTMCellImage,
        unrolling over the time dimension
        to produce a sequence of hidden states as output.

        Parameters
        ----------
        X (torch.Tensor):
            Input tensor of shape
            (batch_size, num_channels, seq_len, height, width),
            representing a sequence of images.

        Return
        ------
        torch.Tensor:
            Output tensor of shape
            (batch_size, out_channels, seq_len, height, width),
            containing the hidden states for each time step in the sequence.
        """
        # X is a frame sequence (batch_size, num_channels, seq_len, height, width) # noqa

        # Get the dimensions
        batch_size, _, seq_len, height, width = X.size()

        # Initialize output
        output = torch.zeros(batch_size, self.out_channels, seq_len,
                             height, width, device=device)

        # Initialize Hidden State
        H = torch.zeros(batch_size, self.out_channels,
                        height, width, device=device)

        # Initialize Cell Input
        C = torch.zeros(batch_size, self.out_channels,
                        height, width, device=device)

        # Unroll over time steps
        for time_step in range(seq_len):

            H, C = self.convLSTMcell(X[:, :, time_step], H, C)

            output[:, :, time_step] = H

        return output
