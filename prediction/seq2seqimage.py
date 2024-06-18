import torch.nn as nn
from .convlstmimage import ConvLSTMImage


class Seq2SeqImage(nn.Module):
    """
    Implements a sequence-to-sequence (Seq2Seq) model
    tailored for image sequences, using ConvLSTMImage layers.

    This model is designed for tasks
    that involve sequences of images as input and require a single image
    as output, such as future frame prediction.
    It leverages ConvLSTMImage layers to capture spatial and
    temporal dependencies and concludes with a convolutional layer
    to predict the output image.

    Attributes:
        sequential (nn.Sequential):
            A sequence of ConvLSTMImage and BatchNorm3d layers.
        conv (nn.Conv2d):
            A convolutional layer
            that generates the final output image from the last hidden state.

    Args:
        num_channels (int):
            The number of channels in the input and output images.
        num_kernels (int):
            The number of output channels for the ConvLSTMImage layers.
        kernel_size (tuple):
            The size of the convolutional kernel.
        padding (int):
            The amount of padding added to the input on all sides.
        activation (str):
            The activation function used in
            the ConvLSTMImage layers (tanh or relu).
        frame_size (tuple):
            The spatial dimensions (height, width) of the input images.
        num_layers (int):
            The number of ConvLSTMImage layers in the model.
    """
    def __init__(self, num_channels, num_kernels, kernel_size, padding,
                 activation, frame_size, num_layers):
        """
        Initializes the Seq2SeqImage model
        with the specified architecture and parameters
        for processing image sequences.

        Parameters
        ----------
        num_channels: int
            The number of channels in the input and output images.

        num_kernels: int
            The number of output feature map channels
            in the ConvLSTMImage layers.

        kernel_size: int
            The dimensions of the convolutional kernel.

        padding: int
            Padding to be added to the input images.

        activation: string
            Activation function (tanh or relu).

        frame_size: tuple
            The size (height, width) of the input images.

        num_layers: int
            The total number of ConvLSTMImage layers in the model.
        """
        super(Seq2SeqImage, self).__init__()

        self.sequential = nn.Sequential()

        # Add First layer (Different in_channels than the rest)
        self.sequential.add_module(
            "convlstm1", ConvLSTMImage(
                in_channels=num_channels, out_channels=num_kernels,
                kernel_size=kernel_size, padding=padding,
                activation=activation, frame_size=frame_size)
        )

        self.sequential.add_module(
            "batchnorm1", nn.BatchNorm3d(num_features=num_kernels)
        )

        # Add rest of the layers
        for _ in range(2, num_layers+1):

            self.sequential.add_module(
                f"convlstm{_}", ConvLSTMImage(
                    in_channels=num_kernels, out_channels=num_kernels,
                    kernel_size=kernel_size, padding=padding,
                    activation=activation, frame_size=frame_size)
                )

            self.sequential.add_module(
                f"batchnorm{_}", nn.BatchNorm3d(num_features=num_kernels)
                )

        # Add Convolutional Layer to predict output frame
        self.conv = nn.Conv2d(
            in_channels=num_kernels, out_channels=num_channels,
            kernel_size=kernel_size, padding=padding)

    def forward(self, X):
        """
        Forward pass of the Seq2SeqImage model.

        Processes a sequence of images through ConvLSTMImage layers
        and BatchNorm3d layers, then uses a convolutional layer to
        predict the final output image from the last time step's hidden state.

        Parameters
        ----------
        X (torch.Tensor):
            Input tensor of shape
            (batch_size, num_channels, seq_len, height, width),
            representing a sequence of images.

        Return
        ------
        torch.Tensor:
            The predicted output image of shape
            (batch_size, num_channels, height, width),
            obtained after applying the sigmoid activation function
            to the convolutional layer's output.
        """
        # Forward propagation through all the layers
        output = self.sequential(X)

        # Return only the last output frame
        output = self.conv(output[:, :, -1])

        return nn.Sigmoid()(output)
