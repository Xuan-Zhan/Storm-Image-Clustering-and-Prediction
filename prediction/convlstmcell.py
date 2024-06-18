import torch
import torch.nn as nn


# Original ConvLSTM cell as proposed by Shi et al.
class ConvLSTMCell(nn.Module):
    """
    Implements a single Convolutional LSTM (ConvLSTM) cell.

    The ConvLSTM cell is capable of processing input data
    with spatial dimensions (i.e., images), and it combines the ability
    to extract spatial features (using convolutional layers)
    with the ability to remember long-term dependencies (LSTM architecture).

    The implementation is based on the original ConvLSTM cell
    proposed by Shi et al. and incorporates elements adapted
    from publicly available implementations.

    Attributes:
        activation (function):
            The activation function used in the cell (tanh or relu).
        conv (nn.Conv2d):
            The convolutional layer combines input and previous hidden state.
        W_ci, W_co, W_cf (nn.Parameter):
            Weights for Hadamard products for input, output, and forget gates.

    Args:
        in_channels (int):
            Number of channels in the input image.
        out_channels (int):
            Number of channels in the output feature maps.
        kernel_size (tuple):
            Size of the convolutional kernel.
        padding (int):
            Padding added to both sides of the input.
        activation (str):
            Activation function to use ('tanh' or 'relu').
        frame_size (tuple):
            Spatial size of the input images (height, width).
    """
    def __init__(self, in_channels, out_channels,
                 kernel_size, padding, activation, frame_size):
        """
        Initializes the ConvLSTMCell with the specified parameters.

        Parameters
        ----------
        in_channels: int
            Number of channels in the input image.

        out_channels: int
            Number of channels in the output feature maps.

        kernel_size: tuple
            Size of the convolutional kernel.

        padding: int
            Padding added to both sides of the input.

        activation: string
            Activation function to use ('tanh' or 'relu').

        frame_size: tuple
            Spatial size of the input images (height, width).
        """
        super(ConvLSTMCell, self).__init__()

        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = torch.relu

        # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
        self.conv = nn.Conv2d(
            in_channels=in_channels + out_channels,
            out_channels=4 * out_channels,
            kernel_size=kernel_size,
            padding=padding)

        # Initialize weights for Hadamard Products
        self.W_ci = nn.Parameter(torch.Tensor(out_channels, *frame_size))
        self.W_co = nn.Parameter(torch.Tensor(out_channels, *frame_size))
        self.W_cf = nn.Parameter(torch.Tensor(out_channels, *frame_size))

    def forward(self, X, H_prev, C_prev):
        """
        Performs a forward pass of the ConvLSTMCell.

        Processes the input and the previous states
        to produce the new hidden state and cell state.

        Parameters
        ----------
        X: torch.Tensor
            The input tensor of shape (batch_size, in_channels, height, width).

        H_prev: torch.Tensor
            The previous hidden state of shape
            (batch_size, out_channels, height, width).

        C_prev: torch.Tensor
            The previous cell state of shape
            (batch_size, out_channels, height, width).

        Return
        ------
        tuple:
            A tuple containing the new hidden state (H) and new cell state (C),
            each of shape (batch_size, out_channels, height, width).
        """
        # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
        conv_output = self.conv(torch.cat([X, H_prev], dim=1))

        # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
        i_conv, f_conv, C_conv, o_conv = torch.chunk(conv_output, chunks=4, dim=1) # noqa

        input_gate = torch.sigmoid(i_conv + self.W_ci * C_prev)
        forget_gate = torch.sigmoid(f_conv + self.W_cf * C_prev)

        # Current Cell output
        C = forget_gate*C_prev + input_gate * self.activation(C_conv)

        output_gate = torch.sigmoid(o_conv + self.W_co * C)

        # Current Hidden State
        H = output_gate * self.activation(C)

        return H, C
