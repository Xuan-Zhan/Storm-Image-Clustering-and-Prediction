import torch
import torch.nn as nn


# Original ConvLSTM cell as proposed by Shi et al.
class ConvLSTMCellImage(nn.Module):
    """
    Implements a Convolutional LSTM (ConvLSTM) cell
    specifically designed for image input.

    This variant of the ConvLSTM cell is tailored for handling image data,
    making it suitable for tasks that involve spatial data processing,
    such as video frame prediction or image time series analysis.
    It follows the original ConvLSTM design proposed by Shi et al.,
    with the ability to process spatial data through convolutional operations
    within the LSTM cell structure.

    Attributes:
        activation (function):
            The activation function used within the cell (either tanh or relu).
        conv (nn.Conv2d):
            A convolutional layer that combines input and previous hidden state
            to produce gate activations.
        W_ci, W_co, W_cf (nn.Parameter):
            Parameters for element-wise multiplication with the cell state,
            influencing the input, output, and forget gates respectively.

    Args:
        in_channels (int):
            The number of channels in the input image.
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
        Initializes the ConvLSTMCellImage
        with specified parameters for image processing.

        Parameters
        ----------
        in_channels: int
            The number of channels in the input image.

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

        super(ConvLSTMCellImage, self).__init__()

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
        Forward pass of the ConvLSTMCellImage.

        Processes the input image,
        along with the previous hidden and cell states,
        to produce the new hidden and cell states.
        This mechanism allows the cell
        to retain and process spatial information over time.

        Parameters
        ----------
        X: torch.Tensor
            Input tensor of shape (batch_size, in_channels, height, width).

        H_prev: torch.Tensor
            Previous hidden state of shape
            (batch_size, out_channels, height, width).

        C_prev: torch.Tensor
            Previous cell state of shape
            (batch_size, out_channels, height, width).

        Return
        -------
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
