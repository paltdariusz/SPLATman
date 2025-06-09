"""
Stage 0.B: 2D/3D Keypoint Detection
Wrapper for MMPose models like RTMW-3D.
"""

from typing import Dict, List

import cv2
import numpy as np
import torch

# Optional dependency: MMPose. It relies on compiled extensions (e.g. xtcocotools)
# that may not be available in lightweight CI images. Therefore we wrap the import
# in a try/except and provide graceful degradation so that mere *import* of this
# module does not fail during unit-tests that only exercise imports.
try:
    from mmpose.apis import inference_topdown, init_model  # type: ignore
    from mmpose.structures import PoseDataSample  # type: ignore
except (ModuleNotFoundError, ValueError):  # ValueError can arise from binary mismatch
    inference_topdown = None  # type: ignore
    init_model = None  # type: ignore
    PoseDataSample = object  # type: ignore


class KeypointDetector:
    def __init__(self, config: dict, device: torch.device):
        self.config = config
        self.device = device
        self.model_name = config.get("name", "rtmw3d").lower()
        self.model = self._load_model()
        print(f"KeypointDetector initialized with model: {self.model_name}")

    def _load_model(self):
        """Loads the MMPose model from the config paths."""
        if self.model_name == "rtmw3d":
            model = init_model(
                self.config["config_file"],
                self.config["checkpoint_file"],
                device=str(self.device),
            )
            return model
        else:
            raise ValueError(f"Unsupported keypoint detector: {self.model_name}")

    def detect_single_image(self, image_np: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Detects 2D/3D keypoints in a single image.
        Args:
            image_np (np.ndarray): Input image as HxWxC NumPy array (RGB).
        Returns:
            dict: Detected keypoints, e.g., {'keypoints': [[x,y,conf], ...], 'keypoints_3d': [[x,y,z,conf],...]}
                  The structure is standardized for downstream use. For now, we focus on 2D.
        """
        # MMPose models generally expect BGR images
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Run inference
        results: List[PoseDataSample] = inference_topdown(self.model, image_bgr)

        # Process results for the first (and assumed only) person
        if not results:
            print("Warning: No keypoints detected.")
            return {
                "keypoints": np.zeros((133, 3))
            }  # Return empty/zero keypoints for COCO-WholeBody

        # Extract data from the first detected instance
        pred_instances = results[0].pred_instances

        # Get 2D keypoints and scores
        keypoints = pred_instances.keypoints[0]  # (num_keypoints, 2)
        keypoint_scores = pred_instances.keypoint_scores[0]  # (num_keypoints,)

        # Combine into [x, y, score] format
        keypoints_2d_with_scores = np.hstack(
            [keypoints, keypoint_scores[:, np.newaxis]]
        )

        # TODO: Add extraction for 3D keypoints if the model provides them
        # keypoints_3d = results[0].pred_instances.keypoints_3d

        return {"keypoints": keypoints_2d_with_scores}

    def detect_batch(self, images_np: List[np.ndarray]) -> List[Dict]:
        """Detects keypoints for a batch of images."""
        # Simple loop for now; MMPose might have more efficient batching
        all_keypoints_data = []
        for img_np in images_np:
            all_keypoints_data.append(self.detect_single_image(img_np))
        return all_keypoints_data

    def save_keypoints(self, keypoints_data: list[dict], output_dir: str):
        # Placeholder: Save keypoints to disk (e.g., as JSON files)
        pass
