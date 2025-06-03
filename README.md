# 3D Body Reconstruction Pipeline (SMPL-X + Gaussian Splatting)

This project implements an end-to-end pipeline for reconstructing 3D human avatars from sparse multi-view images, leveraging SMPL-X as a parametric body model and 3D Gaussian Splatting for high-fidelity rendering and geometry refinement.

## Features

*   **Stage 0: Preprocessing:** Human segmentation (MODNet/Detectron2), 2D keypoint detection (HRNet/OpenPose), camera calibration (COLMAP).
*   **Stage 1: SMPL-X Fitting:** Initial estimation with PIXIE, followed by multi-view optimization (SMPLify-X style).
*   **Stage 2: Texture Baking:** Coarse texture projection onto the fitted SMPL-X mesh.
*   **Stage 3: Gaussian Initialization:** Seeding 3D Gaussians from the textured SMPL-X mesh.
*   **Stage 4-8: Gaussian Optimization & Refinement:**
    *   Differentiable rendering with `gsplat`.
    *   Optimization with photometric, perceptual (LPIPS), silhouette, and SMPL-X attachment losses.
    *   Adaptive Density Control (splitting/pruning Gaussians).
    *   Color filling for unseen areas and regularization.
*   **Stage 9: Output:** Export final Gaussian cloud and optionally a triangle mesh.
*   **Stage 10: Quality Checks:** Automated evaluation (PSNR, Silhouette IoU).
*   Configuration via YAML files.
*   Designed for HPC/SLURM environments.

## Project Structure
3D-Body_Rec/
├── main_pipeline.py # Main script
├── configs/ # Configuration files
├── src/ # Source code (preprocessing, fitting, gaussians, utils, eval)
├── scripts/ # Helper scripts (SLURM job, model download)
├── data/ # Input data
├── results/ # Output data
├── pretrained_models/ # Pretrained model weights
├── third_party/ # External libraries (e.g., gsplat submodule)
├── README.md
└── requirements.txt


## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd project_3d_reconstruction
    ```

2.  **(Optional) Initialize Submodules:** If `gsplat` or other libraries are included as submodules:
    ```bash
    git submodule update --init --recursive
    ```

3.  **Create Environment & Install Dependencies:**
    Using Conda (recommended):
    ```bash
    conda create -n 3d_recon_env python=3.9 # Or desired Python version
    conda activate 3d_recon_env
    # Install PyTorch with CUDA support (adjust for your CUDA version)
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia 
    pip install -r requirements.txt
    # Install gsplat (if not a submodule, or if it has specific build steps)
    # pip install third_party/gsplat # Example
    ```
    `requirements.txt` should include: `pyyaml, numpy, opencv-python, Pillow, smplx, pytorch3d, torchmetrics, lpips, pycolmap (if pip installable), mmpose, mmcv (for MMPose)`, etc.

4.  **Download Pretrained Models:**
    Run the script to download necessary models (SMPL-X, VPoser, PIXIE, MODNet, HRNet, LPIPS weights) and place them in the `pretrained_models/` directory according to paths in `configs/pipeline_config.yaml`.
    ```bash
    bash scripts/download_models.sh 
    ```
    You will need to register on the SMPL-X website to download the SMPL-X model files and VPoser.

5.  **Install COLMAP:**
    Ensure COLMAP is installed and accessible via the command line (or update `colmap_path` in the config).

## Usage

1.  **Prepare Input Data:**
    Place your subject's multi-view images in `data/YOUR_SUBJECT_ID/images/`.
    If you have camera intrinsics, you can place them in `data/YOUR_SUBJECT_ID/optional_camera_intrinsics.json`.

2.  **Configure the Pipeline:**
    Edit `configs/pipeline_config.yaml`:
    *   Set `data.subject_id` to `YOUR_SUBJECT_ID`.
    *   Verify paths to pretrained models and tools.
    *   Adjust pipeline parameters as needed.

3.  **Run the Pipeline:**
    *   **Locally:**
        ```bash
        python main_pipeline.py --config configs/pipeline_config.yaml
        ```
    *   **On SLURM:**
        Edit `scripts/run_pipeline_slurm.sbatch` to set your environment activation and project path.
        Then submit the job:
        ```bash
        sbatch scripts/run_pipeline_slurm.sbatch
        ```

4.  **View Results:**
    Outputs will be saved in `results/YOUR_SUBJECT_ID/` organized by stage.
    The final Gaussian cloud (e.g., `.ply`) can be viewed with a compatible 3DGS viewer.

## Citation
If you use components from this pipeline, please cite the respective original publications for SMPL-X, PIXIE, MODNet, HRNet/OpenPose, COLMAP, Gaussian Splatting, LPIPS, PyTorch3D, etc.