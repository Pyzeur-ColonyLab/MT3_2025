#!/bin/bash
#
# MT3 Checkpoint Setup Script for Brev NVIDIA Instance
# Downloads T5X checkpoint from Google Cloud Storage and converts to PyTorch format
#
# Usage: bash setup_mt3_checkpoint.sh
#

set -e  # Exit on error

echo "========================================================================"
echo "🎵 MT3 Checkpoint Setup for Brev Instance"
echo "========================================================================"
echo ""

# Configuration
CHECKPOINT_DIR="checkpoints/mt3"
OUTPUT_DIR="."
CHECKPOINT_URL="gs://magentadata/models/mt3/checkpoints/mt3/"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}ℹ${NC}  $1"
}

print_success() {
    echo -e "${GREEN}✓${NC}  $1"
}

print_error() {
    echo -e "${RED}✗${NC}  $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC}  $1"
}

# Step 1: Check if checkpoint already exists
echo ""
print_status "Step 1/5: Checking for existing checkpoint..."
if [ -f "mt3_converted.pth" ]; then
    print_warning "mt3_converted.pth already exists!"
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_status "Keeping existing checkpoint. Exiting."
        exit 0
    fi
fi

# Step 2: Install required dependencies
echo ""
print_status "Step 2/5: Installing required Python packages..."
pip install -q zarr numcodecs gsutil 2>/dev/null || {
    print_warning "Some packages may already be installed"
}
print_success "Dependencies ready"

# Step 3: Check for gsutil
echo ""
print_status "Step 3/5: Checking gsutil availability..."
if ! command -v gsutil &> /dev/null; then
    print_error "gsutil not found!"
    print_status "Installing Google Cloud SDK..."

    # Install gsutil via pip (simpler than full SDK)
    pip install gsutil

    if ! command -v gsutil &> /dev/null; then
        print_error "Failed to install gsutil"
        print_status "Please install manually: pip install gsutil"
        exit 1
    fi
fi
print_success "gsutil is available"

# Step 4: Download T5X checkpoint
echo ""
print_status "Step 4/5: Downloading MT3 checkpoint from Google Cloud Storage..."
print_warning "This may take 5-10 minutes depending on connection speed"
print_status "Checkpoint size: ~200-500 MB"
echo ""

mkdir -p "$CHECKPOINT_DIR"

# Download with progress
print_status "Downloading from: $CHECKPOINT_URL"
if gsutil -m cp -r "$CHECKPOINT_URL" "$CHECKPOINT_DIR/" 2>&1 | grep -E "(Copying|Building|Completed)"; then
    print_success "Checkpoint downloaded successfully"
else
    print_error "Download failed!"
    echo ""
    print_status "This usually happens because gsutil needs Google Cloud authentication."
    print_status "Don't worry! We have alternative methods."
    echo ""
    print_status "📋 Alternative Options:"
    echo ""
    echo "  1️⃣  Try the alternative download script:"
    echo "      bash download_checkpoint_alternative.sh"
    echo ""
    echo "  2️⃣  Follow the manual download guide:"
    echo "      cat MANUAL_CHECKPOINT_DOWNLOAD.md"
    echo "      (or open it in a text editor)"
    echo ""
    echo "  3️⃣  Setup Google Cloud authentication:"
    echo "      curl https://sdk.cloud.google.com | bash"
    echo "      exec -l \$SHELL"
    echo "      gcloud init --skip-diagnostics"
    echo "      Then re-run this script"
    echo ""
    echo "  4️⃣  Download pre-converted checkpoint:"
    echo "      Check: https://github.com/kunato/mt3-pytorch/releases"
    echo "      Or: https://huggingface.co/models?search=mt3"
    echo ""
    print_warning "See MANUAL_CHECKPOINT_DOWNLOAD.md for detailed instructions"
    exit 1
fi

# Verify checkpoint structure
echo ""
print_status "Verifying checkpoint structure..."
PARAM_COUNT=$(find "$CHECKPOINT_DIR" -type d -name "target.*" | wc -l)
print_status "Found $PARAM_COUNT parameter folders"

if [ "$PARAM_COUNT" -lt 100 ]; then
    print_warning "Expected ~147 parameter folders, found only $PARAM_COUNT"
    print_warning "Checkpoint may be incomplete"
fi

# Step 5: Convert to PyTorch
echo ""
print_status "Step 5/5: Converting T5X checkpoint to PyTorch format..."
echo ""

python3 t5x_converter_fixed.py "$CHECKPOINT_DIR/mt3" "$OUTPUT_DIR"

# Check if conversion succeeded
if [ -f "mt3_converted.pth" ]; then
    echo ""
    echo "========================================================================"
    print_success "MT3 CHECKPOINT SETUP COMPLETE!"
    echo "========================================================================"
    echo ""

    # Show file info
    FILE_SIZE=$(du -h mt3_converted.pth | cut -f1)
    print_success "Checkpoint file: mt3_converted.pth ($FILE_SIZE)"

    if [ -f "config.json" ]; then
        print_success "Configuration: config.json"
    fi

    if [ -f "parameter_mapping.txt" ]; then
        print_success "Parameter mapping: parameter_mapping.txt"
    fi

    echo ""
    print_status "Next steps:"
    echo "  1. Verify the checkpoint:"
    echo "     python3 -c \"import torch; c=torch.load('mt3_converted.pth'); print('Parameters:', c['metadata']['total_params'])\""
    echo ""
    echo "  2. Test MT3 inference:"
    echo "     python example_inference.py audio.wav --checkpoint mt3_converted.pth"
    echo ""
    echo "  3. Or use the Jupyter notebook:"
    echo "     jupyter notebook MT3_Test_Notebook.ipynb"
    echo ""

    # Optional: Cleanup
    echo ""
    read -p "Do you want to delete the intermediate checkpoint directory (~500MB)? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$CHECKPOINT_DIR"
        print_success "Checkpoint directory cleaned up"
    fi

else
    echo ""
    echo "========================================================================"
    print_error "CONVERSION FAILED"
    echo "========================================================================"
    echo ""
    print_status "Troubleshooting:"
    echo "  1. Check if zarr is installed: pip install zarr"
    echo "  2. Verify checkpoint structure: ls -la $CHECKPOINT_DIR/mt3/"
    echo "  3. Check the conversion guide: cat t5x_conversion_guide.md"
    echo ""
    exit 1
fi