"""
Stage 0.A: Human Segmentation
Wrapper for P3M (ViTAE-S) or Detectron2.
"""

from typing import Callable, Optional

import numpy as np
import torch
from torchvision import transforms

# from PIL import Image
# import torchvision.transforms as T
# For MODNet: from a MODNet implementation (e.g., ZHKKKe/MODNet)
# from third_party.MODNet.src.models.modnet import MODNet
# For Detectron2:
# from detectron2.engine import DefaultPredictor
# from detectron2.config import get_cfg

# We assume the P3M repository has been cloned into third_party/P3M_Net
# The import is deferred and wrapped in a try/except to avoid hard failures when the
# optional third-party dependency is not yet available (e.g. during CI unit tests).
build_model: Optional[Callable[..., torch.nn.Module]] = None
try:
    from third_party.P3M_Net.core.network import build_model  # type: ignore
except ModuleNotFoundError:  # pragma: no cover – optional dependency
    pass  # build_model remains None


class HumanSegmenter:
    def __init__(self, config: dict, device: torch.device):
        self.config = config
        self.device = device
        self.model_name = config.get("name", "P3M-ViTAE-S").lower()
        self.model = self._load_model()
        self.transform = self._get_transform()

    def _load_model(self):
        if self.model_name == "p3m-vitae-s":
            # The original P3M-Net code expects a different config structure.
            # We are simplifying here to load the vitae_s model directly.
            # The num_classes and other params are handled within the model definition.
            if build_model is None:
                raise ModuleNotFoundError("P3M-Net is not available")
            model = build_model("vitae")
            model.load_state_dict(
                torch.load(self.config["checkpoint_path"], map_location="cpu")["model"]
            )
            model = model.to(self.device)
            model.eval()
            return model
        elif self.model_name == "detectron2":
            # Placeholder: Load Detectron2 model
            # cfg = get_cfg()
            # cfg.merge_from_file(self.config['config_file'])
            # cfg.MODEL.WEIGHTS = self.config['model_weights']
            # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7 # example
            # cfg.MODEL.DEVICE = str(self.device)
            # predictor = DefaultPredictor(cfg)
            # return predictor
            print(
                f"Placeholder: Detectron2 model loaded using {self.config['config_file']}"
            )
            return None  # Replace with actual model
        else:
            raise ValueError(f"Unsupported segmentation model: {self.model_name}")

    def _get_transform(self):
        if self.model_name == "p3m-vitae-s":
            return transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        return None

    def segment_single_image(self, image_np: np.ndarray) -> np.ndarray:
        """
        Segments a single image.
        Args:
            image_np (np.ndarray): Input image as HxWxC NumPy array (RGB, 0-255).
        Returns:
            np.ndarray: Alpha matte as HxW NumPy array (float32, 0-1).
        """
        if self.model is None:
            return np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.float32)

        if self.model_name == "p3m-vitae-s":
            h, w, _ = image_np.shape
            image_tensor = self.transform(image_np).unsqueeze(0).to(self.device)

            with torch.no_grad():
                # The P3M-Net model returns a dictionary of outputs
                outputs = self.model(image_tensor)
                alpha_pred = outputs["phas"]

            # Resize alpha matte back to original image size
            alpha_pred_resized = torch.nn.functional.interpolate(
                alpha_pred, size=(h, w), mode="bilinear", align_corners=False
            )
            alpha_matte = alpha_pred_resized[0, 0, ...].cpu().numpy()
            return np.clip(alpha_matte, 0, 1)

        elif self.model_name == "detectron2":
            # Placeholder: Detectron2 inference
            # outputs = self.model(image_np) # Detectron2 takes BGR by default
            # instances = outputs["instances"].to("cpu")
            # person_masks = instances[instances.pred_classes == 0].pred_masks # COCO person class is 0
            # if len(person_masks) == 0:
            #     return np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.uint8)
            # combined_mask = torch.any(person_masks, dim=0).numpy().astype(np.uint8)
            # return combined_mask
            pass
        return np.zeros(
            (image_np.shape[0], image_np.shape[1]), dtype=np.float32
        )  # Fallback

    def get_binary_mask(self, alpha_matte: np.ndarray) -> np.ndarray:
        """Converts a soft alpha matte to a binary mask based on a threshold."""
        threshold = self.config.get("binary_threshold", 0.9)
        binary_mask = (alpha_matte > threshold).astype(np.uint8)
        return binary_mask

    def segment_batch(
        self, images_np: list[np.ndarray]
    ) -> tuple[list[np.ndarray], list[torch.Tensor]]:
        """
        Segments a batch of images.
        Returns:
            Tuple: (list of HxW alpha mattes [np.ndarray], list of 1xHxW binary masks [torch.Tensor])
        """
        mattes_np = []
        masks_tensors = []
        for img_np in images_np:
            alpha_matte = self.segment_single_image(img_np)
            binary_mask = self.get_binary_mask(alpha_matte)
            mattes_np.append(alpha_matte)
            masks_tensors.append(
                torch.from_numpy(binary_mask).unsqueeze(0).to(self.device).float()
            )
        return mattes_np, masks_tensors

    def save_masks(self, masks_np: list[np.ndarray], output_dir: str):
        # Placeholder: Save masks to disk (e.g., as PNG images)
        pass
