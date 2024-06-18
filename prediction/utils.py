"""
Helper functions for using the prediction module. Using
the functions below, one would be able to download weights
"""
import os
from torchvision import transforms
from torchvision import transforms as TF
from PIL import Image


def get_last_images(directory, num_images=15):
    """
    Get the last 'num_images' JPG files from the specified directory.
    """
    all_files = [f for f in sorted(os.listdir(directory)) if f.endswith('.jpg')] # noqa
    return all_files[-num_images:]


def load_and_preprocess_image(image_path, image_size=(128, 128)):
    image = Image.open(image_path).convert('L')
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])
    return transform(image)


def save_image(tensor, filename):
    """Save a PyTorch tensor as an image."""
    image = TF.to_pil_image(tensor)
    image.save(filename)
