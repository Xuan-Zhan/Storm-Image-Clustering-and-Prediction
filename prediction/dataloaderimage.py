import os
from typing import Optional, Tuple, Callable
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch


class HurricaneDatasetImage(Dataset):
    """Load data from files and form a dataset. """
    def __init__(self,
                 image_folder: str,
                 sequence_length: int = 15,
                 transform: Optional[Callable] = None,
                 image_size: Tuple[int, int] = (64, 64),
                 channels: int = 1
                 ): # noqa
        """
        Initialize the dataset.

        Parameters
        ----------
        image_folder: string
            Path to folder containing images.

        sequence_length: int, optional
            The length of sequence. Default to 15.

        transform: callable, optional
            Transform to be applied on an image.

        image_size: tuple
            Target size for the images.

        channels: int
            Number of channels in the images(1 for grayscale, 3 for RGB).
            Default to 1.
        """
        self.image_folder = image_folder
        self.sequence_length = sequence_length
        self.channels = channels
        self.transform = transform if transform else self.default_transform(image_size) # noqa
        self.samples = self._load_samples()

    def _load_samples(self):
        """Load images from the image folder."""
        return [os.path.join(self.image_folder, file)
                for file in sorted(os.listdir(self.image_folder)) if file.endswith('.jpg')] # noqa

    def default_transform(self, image_size: Tuple[int, int]) -> Callable:
        """
        Define default transformation:
        Resize, Convert to RGB or Grayscale, and Convert to Tensor.

        Parameters
        ----------
        image_size: tuple
            Target size for the images.

        Returns
        -------
        callable
            Transform to be applied on an image.
        """
        transform_list = [transforms.Resize(image_size)]

        if self.channels == 1:
            transform_list.append(transforms.Grayscale(num_output_channels=1))
        elif self.channels == 3:
            transform_list.append(transforms.Lambda(lambda img: img.convert('RGB'))) # noqa

        transform_list.append(transforms.ToTensor())
        return transforms.Compose(transform_list)

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.samples) - self.sequence_length + 1

    def __getitem__(self, idx: int):
        """
        Fetch an sequence of images from the dataset by index.

        Parameters
        ----------
        idx : int
            Index of the item to fetch.

        Returns
        -------
        tuple
            A tuple containing the input sequence and the target image.
        """
        sequence_indices = list(range(idx, idx + self.sequence_length - 1))

        sequence = [self.load_image(i) for i in sequence_indices]
        target_image = self.load_image(idx + self.sequence_length - 1)

        input_sequence = torch.stack(sequence).permute(1, 0, 2, 3)
        return input_sequence, target_image

    def load_image(self, index: int):
        """
        Load an image.

        Parameters
        ----------
        idx : int
            Index of the image to fetch.

        Returns
        -------
        torch.tensor
            An image tensor.
        """
        image_path = self.samples[index]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        return image
