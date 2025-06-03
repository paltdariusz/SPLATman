"""
Stage 0.C: Camera Parameter Estimation using COLMAP.
"""
import os
import subprocess
# from src.utils.colmap_wrappers import read_cameras_binary, read_images_binary # etc.
# from src.utils.geometry import qvec2rotmat # If needed
# from pytorch3d.renderer import FoVPerspectiveCameras # or PerspectiveCameras

class CameraEstimator:
    def __init__(self, colmap_executable_path: str, options: dict):
        self.colmap_path = colmap_executable_path
        self.options = options

    def run_sfm(self, image_dir: str, output_dir: str, mask_dir: str = None):
        """
        Runs COLMAP Structure-from-Motion pipeline.
        """
        os.makedirs(output_dir, exist_ok=True)
        db_path = os.path.join(output_dir, "database.db")
        sparse_dir = os.path.join(output_dir, "sparse")
        os.makedirs(sparse_dir, exist_ok=True)

        # 1. Feature extraction
        cmd_feature_ext = [
            self.colmap_path, "feature_extractor",
            "--database_path", db_path,
            "--image_path", image_dir,
            "--ImageReader.single_camera", "1", # Assuming shared intrinsics
            # Add more ImageReader options as per config (e.g., camera_model)
        ]
        if mask_dir and self.options.get('use_masks', False):
            cmd_feature_ext.extend(["--ImageReader.mask_path", mask_dir])
        
        print(f"Running COLMAP feature_extractor: {' '.join(cmd_feature_ext)}")
        # subprocess.run(cmd_feature_ext, check=True)

        # 2. Feature matching
        cmd_matcher = [
            self.colmap_path, self.options.get('matcher', 'exhaustive') + "_matcher",
            "--database_path", db_path,
            # Add matcher options
        ]
        print(f"Running COLMAP matcher: {' '.join(cmd_matcher)}")
        # subprocess.run(cmd_matcher, check=True)

        # 3. Incremental mapping (sparse reconstruction)
        cmd_mapper = [
            self.colmap_path, "mapper",
            "--database_path", db_path,
            "--image_path", image_dir,
            "--output_path", sparse_dir,
            # Add mapper options
        ]
        print(f"Running COLMAP mapper: {' '.join(cmd_mapper)}")
        # subprocess.run(cmd_mapper, check=True)
        print(f"COLMAP SfM finished. Outputs in {sparse_dir}")
        # For simplicity, this example assumes COLMAP generates a single model in sparse/0

    def check_sfm_exists(self, sfm_dir: str) -> bool:
        # Check if key COLMAP output files exist (e.g., sparse/0/cameras.bin)
        return os.path.exists(os.path.join(sfm_dir, "sparse", "0", "cameras.bin"))

    def load_colmap_cameras(self, sfm_dir: str, device: str = "cpu"):
        """
        Loads camera parameters from COLMAP's output (sparse/0 directory).
        Converts them into a PyTorch3D-compatible format.
        Returns:
            list[pytorch3d.renderer.CamerasBase]: List of camera objects.
        """
        # Placeholder: This involves reading cameras.bin, images.bin, points3D.bin
        # from colmap_wrappers and converting to PyTorch3D camera objects.
        # Example:
        # camdata = read_cameras_binary(os.path.join(sfm_dir, "sparse/0/cameras.bin"))
        # imdata = read_images_binary(os.path.join(sfm_dir, "sparse/0/images.bin"))
        # Rs = []
        # Ts = []
        # K_mats = [] # Intrinsics
        # image_sizes = []
        # for img_id in sorted(imdata.keys()):
        #     img_info = imdata[img_id]
        #     cam_info = camdata[img_info.camera_id]
        #     R = qvec2rotmat(img_info.qvec) # World-to-Camera
        #     T = img_info.tvec           # World-to-Camera
        #     # Convert R, T to PyTorch3D convention (Camera-to-World for extrinsics, or use world_to_view_transform)
        #     # Construct K matrix from cam_info.params (fx, fy, cx, cy)
        #     # K_matrix = ...
        #     # Rs.append(torch.tensor(R_pytorch3d_convention, dtype=torch.float32))
        #     # Ts.append(torch.tensor(T_pytorch3d_convention, dtype=torch.float32))
        #     # K_mats.append(torch.tensor(K_matrix, dtype=torch.float32))
        #     # image_sizes.append((cam_info.height, cam_info.width))
        #
        # if not Rs: return None
        #
        # cameras = PerspectiveCameras( # or FoVPerspectiveCameras
        #     R=torch.stack(Rs), T=torch.stack(Ts), K=torch.stack(K_mats),
        #     image_size=image_sizes, device=device
        # )
        # return cameras
        print(f"Placeholder: Loading COLMAP cameras from {sfm_dir}")
        return None # Replace with actual PyTorch3D cameras list