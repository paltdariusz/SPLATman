import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from utils.data_utils import load_images


@pytest.fixture
def temp_image_dir():
    """Creates a temporary directory with a dummy image for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dir_path = Path(tmpdir)
        img_path = dir_path / "test_image.png"
        dummy_img = Image.new("RGB", (100, 150), color="red")
        dummy_img.save(img_path)
        yield dir_path


def test_load_images_from_dir(temp_image_dir):
    """Tests loading images from a directory path."""
    torch_imgs, np_imgs = load_images(temp_image_dir)

    assert len(torch_imgs) == 1
    assert len(np_imgs) == 1

    # Test torch tensor
    assert isinstance(torch_imgs[0], torch.Tensor)
    assert torch_imgs[0].shape == (3, 150, 100)  # C, H, W
    assert torch_imgs[0].dtype == torch.float32
    assert torch_imgs[0].max() <= 1.0
    assert torch_imgs[0].min() >= 0.0

    # Test numpy array
    assert isinstance(np_imgs[0], np.ndarray)
    assert np_imgs[0].shape == (150, 100, 3)  # H, W, C
    assert np_imgs[0].dtype == np.uint8
    assert np_imgs[0].max() <= 255
    assert np_imgs[0].min() >= 0


def test_load_images_with_resizing(temp_image_dir):
    """Tests the resizing functionality."""
    new_size = (50, 75)
    torch_imgs, np_imgs = load_images(temp_image_dir, resize_to=new_size)

    assert len(torch_imgs) == 1
    assert len(np_imgs) == 1

    # Test torch tensor
    assert torch_imgs[0].shape == (3, 75, 50)  # C, H, W

    # Test numpy array
    assert np_imgs[0].shape == (75, 50, 3)  # H, W, C


def test_load_images_from_list(temp_image_dir):
    """Tests loading images from a list of paths."""
    image_paths = list(temp_image_dir.glob("*.png"))
    torch_imgs, np_imgs = load_images(image_paths)

    assert len(torch_imgs) == 1
    assert len(np_imgs) == 1
    assert np_imgs[0].shape == (150, 100, 3)


def test_load_images_invalid_path():
    """Tests behavior with an invalid path."""
    with pytest.raises(ValueError):
        load_images("non_existent_path_string")


def test_load_images_empty_dir(temp_image_dir):
    """Tests behavior with an empty directory."""
    empty_subdir = temp_image_dir / "empty"
    empty_subdir.mkdir()
    torch_imgs, np_imgs = load_images(empty_subdir)
    assert len(torch_imgs) == 0
    assert len(np_imgs) == 0
