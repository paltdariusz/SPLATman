#!/bin/bash
# This script downloads the pretrained models required for the pipeline.

# --- Configuration ---
MODELS_DIR="pretrained_models"

# Define model subdirectories
SMPLX_DIR="$MODELS_DIR/smplx"
VPOSER_DIR="$MODELS_DIR/vposer"
PIXIE_DIR="$MODELS_DIR/pixie"
P3M_DIR="$MODELS_DIR/p3m"
RTMW3D_DIR="$MODELS_DIR/rtmw3d"
LPIPS_DIR="$MODELS_DIR/lpips"

# Define model URLs
RTMW3D_CHECKPOINT_URL="https://download.openmmlab.com/mmpose/v1/wholebody_3d_keypoint/rtmw3d/rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.pth"
RTMW3D_CONFIG_URL="https://raw.githubusercontent.com/open-mmlab/mmpose/refs/heads/main/projects/rtmpose3d/configs/rtmw3d-x_8xb32_cocktail14-384x288.py"
LPIPS_VGG_URL="https://github.com/richzhang/PerceptualSimilarity/raw/refs/heads/master/lpips/weights/v0.1/vgg.pth"
P3M_VITAE_S_GDRIVE_ID="1QbSjPA_Mxs7rITp_a9OJiPeFRDwxemqK" # P3M-ViTAE-S.pth
P3M_BACKBONE_GDRIVE_ID="1V2xt0BWCVx550Ll7GGfquvopTLseX9gY"

# --- Helper Functions ---
function print_info {
    echo -e "\e[34m[INFO]\e[0m $1"
}

function print_success {
    echo -e "\e[32m[SUCCESS]\e[0m $1"
}

function print_warning {
    echo -e "\e[33m[WARNING]\e[0m $1"
}

function print_error {
    echo -e "\e[31m[ERROR]\e[0m $1"
}

function download_file {
    local url=$1
    local dest=$2
    local dest_filename=$(basename "$dest")
    if [ -f "$dest" ]; then
        print_info "File $dest_filename already exists. Skipping download."
    else
        print_info "Downloading $dest_filename from $url..."
        wget -q --show-progress -O "$dest" "$url"
        if [ $? -eq 0 ]; then
            print_success "Downloaded $dest_filename successfully."
        else
            print_error "Failed to download $dest_filename."
            exit 1
        fi
    fi
}

function download_gdrive_file {
    local file_id=$1
    local dest_filename=$2
    local dest_dir=$3
    local dest_path="$dest_dir/$dest_filename"

    if [ -f "$dest_path" ]; then
        print_info "File $dest_filename already exists. Skipping download."
    else
        print_info "Attempting to download $dest_filename from Google Drive..."
        # Check if gdown is installed
        if ! command -v gdown &> /dev/null; then
            print_warning "gdown is not installed. Please install it with 'pip install gdown'."
            print_warning "Alternatively, manually download the file with ID $file_id from Google Drive and place it at $dest_path."
            return 1
        fi

        gdown --id "$file_id" -O "$dest_path"
        if [ $? -eq 0 ]; then
            print_success "Downloaded $dest_filename successfully."
        else
            print_error "Failed to download $dest_filename using gdown."
            print_warning "Please manually download the file with ID $file_id from Google Drive and place it at $dest_path."
            return 1
        fi
    fi
}


# --- Main Script ---

# Create directories
print_info "Creating model directories under $MODELS_DIR/"
mkdir -p "$SMPLX_DIR"
mkdir -p "$VPOSER_DIR"
mkdir -p "$PIXIE_DIR"
mkdir -p "$P3M_DIR"
mkdir -p "$RTMW3D_DIR"
mkdir -p "$LPIPS_DIR"

echo ""

# Automated downloads
print_info "----------------------- AUTOMATED DOWNLOADS ------------------------"

# Download RTMW-3D model and config
download_file "$RTMW3D_CHECKPOINT_URL" "$RTMW3D_DIR/rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.pth"
download_file "$RTMW3D_CONFIG_URL" "$RTMW3D_DIR/rtmw3d-x_8xb64-270e_cocktail14-384x288.py"

# Download LPIPS VGG
download_file "$LPIPS_VGG_URL" "$LPIPS_DIR/vgg.pth"

# Download P3M
download_gdrive_file "$P3M_VITAE_S_GDRIVE_ID" "P3M-ViTAE-S.pth" "$P3M_DIR"

echo ""

# Manual download instructions
print_warning "------------------------- MANUAL ACTION REQUIRED -------------------------"
print_warning "The following models require manual download after accepting license agreements."
echo ""
print_info "1. SMPL-X Models:"
print_info "   - Go to: https://smpl-x.is.tue.mpg.de/"
print_info "   - Register, and download the 'SMPL-X v1.1' models."
print_info "   - Place the contents (e.g., SMPLX_NEUTRAL.npz, SMPLX_MALE.npz, etc.) into:"
print_info "     -> $SMPLX_DIR/"
echo ""
print_info "2. VPoser Model:"
print_info "   - The VPoser model is usually downloaded from the same SMPL-X project page."
print_info "   - Download 'VPoser v2.0' checkpoint."
print_info "   - Place the '.ckpt' file into:"
print_info "     -> $VPOSER_DIR/"
echo ""
print_info "3. PIXIE Model:"
print_info "   - Go to: https://pixie.is.tue.mpg.de/"
print_info "   - Register, and download the PIXIE model checkpoint."
print_info "   - Place the '.tar' file into:"
print_info "     -> $PIXIE_DIR/"
print_warning "--------------------------------------------------------------------------"

echo ""
read -p "Press [Enter] to continue..."
echo ""

print_success "Model download script finished."
print_warning "Please ensure you have manually downloaded and placed the required models (SMPL-X, VPoser, PIXIE)."
