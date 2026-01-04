import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from typing import Union, Tuple
import logging
import config

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Image preprocessing pipeline for feature extraction.
    """

    def __init__(self, image_size: Tuple[int, int] = config.IMAGE_SIZE):
        """
        Initialize preprocessor with image size.

        Args:
            image_size: Target image size (height, width)
        """
        self.image_size = image_size
        self.transform = self._build_transform()

    def _build_transform(self):
        """
        Build transformation pipeline.

        Returns:
            Composed transforms
        """
        return transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet means
                std=[0.229, 0.224, 0.225]  # ImageNet stds
            )
        ])

    def preprocess(self, image: Union[str, Image.Image, np.ndarray]) -> torch.Tensor:
        """
        Preprocess image for model input.

        Args:
            image: Image path, PIL Image, or numpy array

        Returns:
            Preprocessed image tensor
        """
        try:
            # Load image if path is provided
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image).convert('RGB')
            elif not isinstance(image, Image.Image):
                raise ValueError(f"Unsupported image type: {type(image)}")

            # Apply transformations
            tensor = self.transform(image)

            # Add batch dimension
            tensor = tensor.unsqueeze(0)

            return tensor

        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise

    def preprocess_batch(self, images: list) -> torch.Tensor:
        """
        Preprocess batch of images.

        Args:
            images: List of images (paths, PIL Images, or numpy arrays)

        Returns:
            Batch tensor
        """
        tensors = []
        for img in images:
            tensor = self.preprocess(img)
            tensors.append(tensor)

        # Concatenate into batch
        batch = torch.cat(tensors, dim=0)
        return batch


def preprocess_image(image_path: str, image_size: Tuple[int, int] = config.IMAGE_SIZE) -> torch.Tensor:
    """
    Convenience function to preprocess a single image.

    Args:
        image_path: Path to image file
        image_size: Target image size

    Returns:
        Preprocessed image tensor
    """
    preprocessor = ImagePreprocessor(image_size)
    return preprocessor.preprocess(image_path)


def load_and_validate_image(image_path: str) -> Image.Image:
    """
    Load and validate image file.

    Args:
        image_path: Path to image file

    Returns:
        PIL Image object

    Raises:
        ValueError: If image cannot be loaded or is invalid
    """
    try:
        image = Image.open(image_path)
        image.verify()  # Verify it's an actual image
        image = Image.open(image_path)  # Reload after verify
        image = image.convert('RGB')  # Ensure RGB format
        return image
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {str(e)}")
        raise ValueError(f"Invalid image file: {image_path}")


def resize_image(image: Image.Image, size: Tuple[int, int], keep_aspect_ratio: bool = True) -> Image.Image:
    """
    Resize image with optional aspect ratio preservation.

    Args:
        image: PIL Image object
        size: Target size (width, height)
        keep_aspect_ratio: Whether to keep aspect ratio

    Returns:
        Resized image
    """
    if keep_aspect_ratio:
        image.thumbnail(size, Image.Resampling.LANCZOS)
        # Create new image with padding if needed
        new_image = Image.new('RGB', size, (255, 255, 255))
        paste_x = (size[0] - image.width) // 2
        paste_y = (size[1] - image.height) // 2
        new_image.paste(image, (paste_x, paste_y))
        return new_image
    else:
        return image.resize(size, Image.Resampling.LANCZOS)


def augment_image(image: Image.Image) -> list:
    """
    Apply data augmentation to image.

    Args:
        image: PIL Image object

    Returns:
        List of augmented images
    """
    augmentations = transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    ])

    augmented = [image]  # Original
    augmented.append(augmentations(image))  # Augmented version

    return augmented