from __future__ import annotations

import os
from typing import List, Tuple, Optional

from PIL import Image
import numpy as np
import torch


def load_images(image_dir: str, device: torch.device, image_size: Optional[Tuple[int, int]] = None) -> Tuple[List[str], List[np.ndarray], List[torch.Tensor]]:
    """Load images from a directory.

    Parameters
    ----------
    image_dir : str
        Directory containing image files.
    device : torch.device
        Device to place returned tensors on.
    image_size : Optional[Tuple[int, int]]
        If provided, images are resized to (height, width).

    Returns
    -------
    Tuple[List[str], List[np.ndarray], List[torch.Tensor]]
        Paths, NumPy arrays (H, W, C in RGB), and float tensors scaled 0-1.
    """
    image_paths = [os.path.join(image_dir, f) for f in sorted(os.listdir(image_dir)) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    images_np: List[np.ndarray] = []
    images_tensors: List[torch.Tensor] = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        if image_size is not None:
            img = img.resize((image_size[1], image_size[0]))
        img_np = np.array(img)
        images_np.append(img_np)
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        images_tensors.append(img_tensor.to(device))
    return image_paths, images_np, images_tensors
