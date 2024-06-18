import os
from typing import Optional, Tuple, Callable
import json
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class HurricaneDataset(Dataset):
    """Load data from files and form a dataset. """
    def __init__(self,
                 image_folder: str,
                 transform: Optional[Callable] = None,
                 image_size: Tuple[int, int] = (64, 64),
                 channels: int = 1,
                 mode: str = 'train'): # noqa
        """
        Initialize the dataset.

        Parameters
        ----------
        image_folder: string
            Path to folder containing images.

        transform: callable, optional
            Transform to be applied on an image.

        image_size: tuple
            Target size for the images. Default to (64, 64)

        channels: int
            Number of channels in the images(1 for grayscale, 3 for RGB).
            Default to 1.

        model: string
            A flag of train or test mode.
        """
        self.image_folder = image_folder
        self.channels = channels
        self.transform = transform if transform else self.default_transform(image_size) # noqa
        self.mode = mode
        self.samples = self._load_samples()

    def _load_samples(self):
        """
        Load all image file paths from the folder.
        """
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
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple:
        """
        Fetch an image and its relevant information.

        Parameters
        ----------
        idx : int
            Index of the item to fetch.

        Returns
        -------
        tuple
            A tuple containing the image and the wind speed or its path.
        """
        image_path = self.samples[idx]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)

        if self.mode == 'train':
            # Load the wind speed from the label file in training mode
            base_filename = os.path.splitext(os.path.basename(image_path))[0]
            label_filename = f"{base_filename}_label.json"
            label_path = os.path.join(self.image_folder, label_filename)
            with open(label_path, 'r') as label_file:
                label_data = json.load(label_file)
                wind_speed = label_data['wind_speed']
            return image, wind_speed
        else:
            # No labels in test mode
            return image, image_path
