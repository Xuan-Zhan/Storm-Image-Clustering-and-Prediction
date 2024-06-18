from torch import nn
from .convlstm import ConvLSTM
import torch.nn.functional as F


class Seq2Seq(nn.Module):
    """
    Implements a sequence-to-sequence (Seq2Seq) model with ConvLSTM layers.

    This model is designed to process sequences of spatial data (e.g., images)
    and predict a single output, such as the wind speed
    in a hurricane prediction task. It utilizes ConvLSTM layers
    for capturing spatiotemporal dynamics,
    followed by fully connected layers to generate the final prediction.

    Attributes:
        sequential (nn.Sequential):
            A sequence of ConvLSTM and BatchNorm2d layers.
        fc (nn.Linear):
            A fully connected layer
            that reduces the feature map to a specified size.
        fc_out (nn.Linear):
            The final fully connected layer that outputs the prediction.

    Args:
        num_channels (int):
            The number of channels in the input images.
        num_kernels (int):
            The number of output channels for the ConvLSTM layers.
        kernel_size (tuple):
            The size of the convolutional kernel.
        padding (int):
            The amount of padding added to the input on all sides.
        activation (str):
            The activation function used in the ConvLSTM layers (tanh or relu).
        frame_size (tuple):
            The spatial dimensions (height, width) of the input images.
        num_layers (int):
            The number of ConvLSTM layers in the model.
        fc_size (int):
            The size of the output from the first fully connected layer.
    """
    def __init__(self, num_channels, num_kernels, kernel_size, padding, activation, frame_size, num_layers, fc_size): # noqa
        """
        Initializes the Seq2Seq model
        with the specified architecture and parameters.

        Parameters
        ----------
        num_channels: int
            The number of channels in the input images.

        num_kernels: int
            The number of output channels for the ConvLSTM layers.

        kernel_size: tuple
            The dimensions of the convolutional kernel.

        padding: int
            Padding to be added to the input images.

        activation: string
            Activation function ('tanh' or 'relu').

        frame_size: tuple
            The size (height, width) of the input images.

        num_layers: int
            The total number of ConvLSTM layers in the model.

        fc_size: int
            The dimensionality of the output
            from the first fully connected layer.
        """
        super(Seq2Seq, self).__init__()

        self.sequential = nn.Sequential()

        # Add the first ConvLSTM layer
        self.sequential.add_module(
            "convlstm1", ConvLSTM(
                in_channels=num_channels, out_channels=num_kernels,
                kernel_size=kernel_size, padding=padding,
                activation=activation, frame_size=frame_size)
        )

        # Use BatchNorm2d instead of BatchNorm3d
        self.sequential.add_module(
            "batchnorm1", nn.BatchNorm2d(num_features=num_kernels)
        )

        # Add the rest of the ConvLSTM layers
        for layer_num in range(2, num_layers + 1):
            self.sequential.add_module(
                f"convlstm{layer_num}", ConvLSTM(
                    in_channels=num_kernels, out_channels=num_kernels,
                    kernel_size=kernel_size, padding=padding,
                    activation=activation, frame_size=frame_size)
            )

            # Again, use BatchNorm2d
            self.sequential.add_module(
                f"batchnorm{layer_num}", nn.BatchNorm2d(num_features=num_kernels) # noqa
            )

        self.fc = nn.Linear(num_kernels * frame_size[0] * frame_size[1], fc_size) # noqa
        self.fc_out = nn.Linear(fc_size, 1)  # Output one value for wind speed

    def forward(self, X):
        """
        Forward pass of the Seq2Seq model.

        Processes a sequence of images
        through ConvLSTM layers and BatchNorm2d layers,
        flattens the output, and then passes it through fully connected layers
        to generate a prediction.

        Parameters
        ----------
        X: torch.Tensor
            Input tensor of shape
            (batch_size, num_channels, seq_len, height, width),
            representing a sequence of images.

        Return
        ------
        torch.Tensor:
            The output prediction of the model,
            typically a single value per input sequence
            after being squeezed to remove any singleton dimensions.
        """
        # Forward propagation through all ConvLSTM layers
        output = self.sequential(X)

        # Flatten the output for the fully connected layer
        output = output.flatten(start_dim=1)

        # Forward through fully connected layers
        output = F.relu(self.fc(output))
        output = self.fc_out(output)

        return output.squeeze()
