from typing import Union

import torch


def get_device(preferred: Union[str, torch.device, None] = None) -> torch.device:
    """Return a valid torch.device.

    If *preferred* is provided and CUDA is available, return the corresponding CUDA
    device. Otherwise, fall back to CPU.
    """
    if preferred is None:
        preferred = "cuda" if torch.cuda.is_available() else "cpu"

    # If the preferred device is a string like "cuda" or "cuda:0", honour it if available.
    if isinstance(preferred, str):
        if preferred.startswith("cuda") and torch.cuda.is_available():
            return torch.device(preferred)
        return torch.device(preferred)

    return torch.device(preferred)
