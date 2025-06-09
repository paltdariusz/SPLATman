import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_images(
    image_sources: Union[Path, List[Path]],
    resize_to: Optional[Tuple[int, int]] = None,
) -> Tuple[List[torch.Tensor], List[np.ndarray]]:
    """
    Loads images from a directory or a list of file paths.

    This function can be extended to handle rendering from 3D model files.
    Currently, it loads images from disk, with optional resizing.

    Args:
        image_sources: A path to a directory containing images, or a list of paths to image files.
        resize_to: An optional tuple (width, height) to resize the images to.

    Returns:
        A tuple containing:
        - A list of images as torch.Tensor objects (C, H, W), normalized to [0, 1].
        - A list of images as NumPy arrays (H, W, C) with integer values [0, 255].
    """
    if isinstance(image_sources, Path) and image_sources.is_dir():
        image_paths = sorted(
            [
                p
                for p in image_sources.iterdir()
                if p.suffix.lower() in [".png", ".jpg", ".jpeg"]
            ]
        )
        logging.info(f"Found {len(image_paths)} images in {image_sources}")
    elif isinstance(image_sources, list):
        image_paths = image_sources
    else:
        raise ValueError(
            "image_sources must be a directory Path or a list of image file Paths."
        )

    torch_images: List[torch.Tensor] = []
    numpy_images: List[np.ndarray] = []

    for img_path in image_paths:
        try:
            img = Image.open(img_path).convert("RGB")
            if resize_to:
                img = img.resize(resize_to, Image.LANCZOS)

            # Convert to NumPy array
            np_img = np.array(img)
            numpy_images.append(np_img)

            # Convert to torch tensor
            torch_img = torch.from_numpy(np_img).permute(2, 0, 1) / 255.0
            torch_images.append(torch_img)

        except IOError:
            logging.warning(f"Could not load or process image: {img_path}")

    logging.info(f"Successfully loaded {len(torch_images)} images.")
    return torch_images, numpy_images
