import torch
import pytest
from prediction import ConvLSTM, ConvLSTMCell, Seq2Seq
from prediction import HurricaneDataset, ConvLSTMImage, ConvLSTMCellImage
from prediction import Seq2SeqImage, HurricaneDatasetImage  # noqa


@pytest.fixture()
def image_folder():
    """
    Provides the path to the test image folder.

    Returns
    -------
        str: The path to the test image folder.
    """
    return 'tests/data/bkh'


def test_hurricanedataset_train(image_folder):
    """
    Tests the training mode of the HurricaneDataset class
    to ensure it loads data correctly.

    Parameters
    ----------
    image_folder: string
        The path to the folder containing the dataset images.

    Asserts:
    - The dataset is not empty.
    - The wind speed is either an integer or a float.
    - The image has a single channel and the correct shape.
    """
    dataset = HurricaneDataset(image_folder=image_folder, mode='train')
    assert len(dataset) > 0, "Dataset should not be empty"

    image, wind_speed = dataset[0]
    # Convert wind_speed to float if it's a string representation of a number
    if isinstance(wind_speed, str) and wind_speed.isdigit():
        wind_speed = float(wind_speed)
    assert isinstance(wind_speed, (int, float)), "Wind speed should be an integer or a float"  # noqa
    assert image.shape[0] == 1, "Image should have 1 channel"
    assert image.shape[1:] == (64, 64), "Image should have the shape (64, 64)"


def test_hurricanedatasetimage(image_folder):
    """
    Tests the HurricaneDatasetImage class
    to ensure it correctly handles image sequences.

    Parameters
    ----------
    image_folder: string
        The path to the folder containing the dataset images.

    Asserts:
    - The dataset is not empty.
    - The input sequence and target image have the correct shapes.
    """
    sequence_length = 15
    dataset = HurricaneDatasetImage(image_folder=image_folder, sequence_length=sequence_length) # noqa
    assert len(dataset) > 0, "Dataset should not be empty"

    input_sequence, target_image = dataset[0]
    assert input_sequence.shape[1] == sequence_length - 1, "Input sequence length should be one less than specified" # noqa
    assert input_sequence.shape[0] == 1, "Input sequence should have 1 channel" # noqa
    assert input_sequence.shape[2:] == (64, 64), "Each image in the sequence should have the shape (64, 64)" # noqa
    assert target_image.shape == (1, 64, 64), "Target image should have the shape (1, 64, 64)" # noqa


def test_convlstm():
    """
    Tests the ConvLSTM model
    for correct output shape with a single input example.

    Asserts:
    - The output shape matches the expected dimensions.
    """
    in_channels, out_channels, kernel_size, padding, activation, frame_size = 3, 16, (3, 3), 1, 'tanh', (64, 64) # noqa
    model = ConvLSTM(in_channels, out_channels, kernel_size, padding, activation, frame_size) # noqa 

    X = torch.rand(1, in_channels, *frame_size)  # Batch size of 1
    output = model(X)

    assert output.shape == (1, out_channels, *frame_size)


def test_convlstmimage():
    """
    Tests the ConvLSTMImage model
    for correct output shape with a sequence input.

    Asserts:
    - The output shape matches the expected dimensions for a sequence input.
    """
    in_channels, out_channels, kernel_size, padding, activation, frame_size = 3, 16, (3, 3), 1, 'tanh', (64, 64) # noqa
    model = ConvLSTMImage(in_channels, out_channels, kernel_size, padding, activation, frame_size) # noqa

    X = torch.rand(1, in_channels, 10, *frame_size)
    output = model(X)

    assert output.shape == (1, out_channels, 10, *frame_size)


def test_convlstmcell():
    """
    Tests the ConvLSTMCell model
    for correct output shape with initial hidden and cell states.

    Asserts:
    - The hidden and cell state output shapes match the expected dimensions.
    """
    in_channels, out_channels, kernel_size, padding, activation, frame_size = 3, 16, (3, 3), 1, 'tanh', (64, 64) # noqa
    model = ConvLSTMCell(in_channels, out_channels, kernel_size, padding, activation, frame_size) # noqa

    X = torch.rand(1, in_channels, *frame_size)
    H_prev = torch.rand(1, out_channels, *frame_size)
    C_prev = torch.rand(1, out_channels, *frame_size)

    H, C = model(X, H_prev, C_prev)

    assert H.shape == (1, out_channels, *frame_size)
    assert C.shape == (1, out_channels, *frame_size)


def test_convlstmcellimage():
    """
    Tests the ConvLSTMCellImage model
    for correct output shape with initial hidden and cell states.

    Asserts:
    - The hidden and cell state output shapes match the expected dimensions.
    """
    in_channels, out_channels, kernel_size, padding, activation, frame_size = 3, 16, (3, 3), 1, 'tanh', (64, 64) # noqa
    model = ConvLSTMCellImage(in_channels, out_channels, kernel_size, padding, activation, frame_size) # noqa

    X = torch.rand(1, in_channels, *frame_size)
    H_prev = torch.rand(1, out_channels, *frame_size)
    C_prev = torch.rand(1, out_channels, *frame_size)

    H, C = model(X, H_prev, C_prev)

    assert H.shape == (1, out_channels, *frame_size)
    assert C.shape == (1, out_channels, *frame_size)


def test_seq2seq():
    """
    Tests the Seq2Seq model for correct output dimensions
    with both single and batch inputs.

    Asserts:
    - The output for a single input has the correct dimension.
    - The output for a batch input is a 1D tensor with length = the batch size.
    """
    num_channels, num_kernels, kernel_size, padding, activation, frame_size, num_layers, fc_size = 3, 16, (3, 3), 1, 'tanh', (64, 64), 2, 100 # noqa
    model = Seq2Seq(num_channels, num_kernels, kernel_size, padding, activation, frame_size, num_layers, fc_size) # noqa

    X_single = torch.rand(1, num_channels, *frame_size)
    output_single = model(X_single)

    X_batch = torch.rand(10, num_channels, *frame_size)  # Batch size of 10
    output_batch = model(X_batch)

    assert output_single.dim() == 0 or (output_single.dim() == 1 and output_single.shape[0] == 1) # noqa

    # Check if output for a batch is 1D tensor with batch size
    assert output_batch.dim() == 1 and output_batch.shape[0] == 10


def test_seq2seqimage():
    """
    Tests the Seq2SeqImage model
    for correct output shape with a sequence input.

    Asserts:
    - The output shape matches the expected dimensions for a sequence input.
    """
    num_channels, num_kernels, kernel_size, padding, activation, frame_size, num_layers = 3, 16, (3, 3), 1, 'tanh', (64, 64), 2 # noqa
    model = Seq2SeqImage(num_channels, num_kernels, kernel_size, padding, activation, frame_size, num_layers) # noqa

    X = torch.rand(1, num_channels, 10, *frame_size)
    output = model(X)

    assert output.shape == (1, num_channels, *frame_size)
