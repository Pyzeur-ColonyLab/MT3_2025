#!/bin/bash
#
# Alternative MT3 Checkpoint Download Script
# Uses direct HTTP/HTTPS sources instead of gsutil
#
# Usage: bash download_checkpoint_alternative.sh
#

set -e

echo "========================================================================"
echo "🎵 MT3 Checkpoint Alternative Download"
echo "========================================================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${BLUE}ℹ${NC}  $1"; }
print_success() { echo -e "${GREEN}✓${NC}  $1"; }
print_error() { echo -e "${RED}✗${NC}  $1"; }
print_warning() { echo -e "${YELLOW}⚠${NC}  $1"; }

CHECKPOINT_DIR="checkpoints/mt3"

# Method 1: Try Hugging Face Hub
try_huggingface() {
    print_status "Method 1: Trying Hugging Face Hub..."

    # Check if git-lfs is installed
    if ! command -v git-lfs &> /dev/null; then
        print_status "Installing git-lfs..."
        sudo apt-get update -qq
        sudo apt-get install -y git-lfs
        git lfs install
    fi

    print_status "Cloning MT3 from Hugging Face..."
    if git clone https://huggingface.co/spaces/SungBeom/mt3 temp_mt3_download 2>&1 | grep -E "(Cloning|Receiving|Resolving)"; then

        # Find the checkpoint directory
        if [ -d "temp_mt3_download/checkpoints/mt3" ]; then
            print_status "Extracting checkpoint..."
            mkdir -p "$CHECKPOINT_DIR"
            cp -r temp_mt3_download/checkpoints/mt3/* "$CHECKPOINT_DIR/"
            rm -rf temp_mt3_download
            print_success "Downloaded from Hugging Face"
            return 0
        else
            print_warning "Checkpoint not found in expected location"
            rm -rf temp_mt3_download
            return 1
        fi
    else
        print_warning "Hugging Face download failed"
        rm -rf temp_mt3_download 2>/dev/null
        return 1
    fi
}

# Method 2: Direct download from known mirrors
try_direct_download() {
    print_status "Method 2: Trying direct download..."

    # List of potential direct download URLs
    URLS=(
        "https://storage.googleapis.com/magentadata/models/mt3/checkpoints/mt3/"
    )

    mkdir -p "$CHECKPOINT_DIR"

    for url in "${URLS[@]}"; do
        print_status "Trying: $url"
        if wget -q --spider "$url" 2>/dev/null; then
            print_status "URL accessible, downloading..."
            if wget -r -np -nH --cut-dirs=4 -R "index.html*" "$url" -P "$CHECKPOINT_DIR/"; then
                print_success "Downloaded successfully"
                return 0
            fi
        fi
    done

    print_warning "Direct download failed"
    return 1
}

# Method 3: Manual instructions
provide_manual_instructions() {
    echo ""
    echo "========================================================================"
    print_error "AUTOMATIC DOWNLOAD FAILED"
    echo "========================================================================"
    echo ""
    print_status "Please download the checkpoint manually using one of these methods:"
    echo ""

    echo "📥 METHOD A: Using gsutil with authentication"
    echo "────────────────────────────────────────────"
    echo "1. Install Google Cloud SDK:"
    echo "   curl https://sdk.cloud.google.com | bash"
    echo "   exec -l \$SHELL"
    echo ""
    echo "2. Initialize gcloud (no account needed for public data):"
    echo "   gcloud init --skip-diagnostics"
    echo ""
    echo "3. Download checkpoint:"
    echo "   gsutil -m cp -r gs://magentadata/models/mt3/checkpoints/mt3/ checkpoints/mt3/"
    echo ""

    echo "📥 METHOD B: Download pre-converted PyTorch checkpoint"
    echo "────────────────────────────────────────────────────"
    echo "1. Check kunato's repository:"
    echo "   https://github.com/kunato/mt3-pytorch"
    echo ""
    echo "2. Look for releases or pre-converted checkpoints"
    echo ""
    echo "3. Download mt3_converted.pth directly if available"
    echo ""

    echo "📥 METHOD C: Use wget with public bucket"
    echo "─────────────────────────────────────────"
    echo "Try downloading individual files:"
    echo "   mkdir -p checkpoints/mt3/mt3"
    echo "   cd checkpoints/mt3/mt3"
    echo "   # Then download files listed in the checkpoint manifest"
    echo ""

    echo "📥 METHOD D: Alternative sources"
    echo "─────────────────────────────────"
    echo "Check these sources:"
    echo "  - https://magenta.tensorflow.org/datasets/mt3"
    echo "  - https://github.com/magenta/mt3 (check releases)"
    echo "  - Kaggle datasets (search for 'mt3 checkpoint')"
    echo ""

    print_status "After manual download, run:"
    echo "   python3 t5x_converter_fixed.py checkpoints/mt3/mt3 ."
    echo ""
}

# Main execution
echo ""
print_status "Attempting to download MT3 checkpoint..."
echo ""

# Try each method
if try_huggingface; then
    print_success "Download successful via Hugging Face!"

    # Verify checkpoint structure
    PARAM_COUNT=$(find "$CHECKPOINT_DIR" -type d -name "target.*" | wc -l)
    print_status "Found $PARAM_COUNT parameter folders"

    if [ "$PARAM_COUNT" -lt 100 ]; then
        print_warning "Expected ~147 folders, found only $PARAM_COUNT"
        print_warning "Checkpoint may be incomplete"
    else
        echo ""
        print_status "Proceeding to conversion..."
        python3 t5x_converter_fixed.py "$CHECKPOINT_DIR" .
        exit $?
    fi

elif try_direct_download; then
    print_success "Download successful via direct URL!"
    echo ""
    print_status "Proceeding to conversion..."
    python3 t5x_converter_fixed.py "$CHECKPOINT_DIR" .
    exit $?
else
    provide_manual_instructions
    exit 1
fi