import torch

def get_device(preferred: str = "cuda") -> torch.device:
    """Return a torch.device based on availability and preference."""
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
