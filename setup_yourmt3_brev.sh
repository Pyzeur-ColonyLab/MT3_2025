#!/bin/bash
#
# Setup YourMT3 on Brev NVIDIA instance
#

set -e

echo "==========================================="
echo "YourMT3 Setup for Brev Instance"
echo "==========================================="

# Check if on Brev
if [[ ! $(hostname) =~ brev ]]; then
    echo "⚠️  Warning: This script is designed for Brev instances"
    echo "   Current hostname: $(hostname)"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Install dependencies
echo ""
echo "1. Installing Python dependencies..."
pip install pytorch-lightning>=2.2.1 --quiet
pip install transformers==4.45.1 --quiet
pip install einops mido --quiet
pip install pretty_midi --quiet
echo "   ✅ Dependencies installed"

# Clone YourMT3 Space
echo ""
echo "2. Cloning YourMT3 HuggingFace Space..."
if [ ! -d "yourmt3_space" ]; then
    git clone https://huggingface.co/spaces/mimbres/YourMT3 yourmt3_space
    echo "   ✅ Space cloned"
else
    echo "   ℹ️  Space already exists, skipping clone"
fi

# Check HF authentication
echo ""
echo "3. Checking HuggingFace authentication..."
if huggingface-cli whoami &>/dev/null; then
    echo "   ✅ Already logged in to HuggingFace"
else
    echo "   ⚠️  Not logged in to HuggingFace"
    echo "   Please run: huggingface-cli login"
    echo "   Then re-run this script"
    exit 1
fi

# Download checkpoint
echo ""
echo "4. Downloading YourMT3 checkpoint (536 MB)..."
echo "   This may take a few minutes..."

CHECKPOINT_PATH="yourmt3_space/amt/logs/2024/mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops/checkpoints/last.ckpt"

if [ -f "$CHECKPOINT_PATH" ]; then
    SIZE=$(stat -f%z "$CHECKPOINT_PATH" 2>/dev/null || stat -c%s "$CHECKPOINT_PATH" 2>/dev/null)
    if [ "$SIZE" -gt 500000000 ]; then
        echo "   ✅ Checkpoint already downloaded ($(numfmt --to=iec-i --suffix=B $SIZE 2>/dev/null || echo ${SIZE} bytes))"
    else
        echo "   ⚠️  Checkpoint file exists but is too small, re-downloading..."
        rm -f "$CHECKPOINT_PATH"
    fi
fi

if [ ! -f "$CHECKPOINT_PATH" ] || [ "$SIZE" -lt 500000000 ]; then
    cd yourmt3_space
    huggingface-cli download mimbres/YourMT3 \
        "amt/logs/2024/mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops/checkpoints/last.ckpt" \
        --repo-type space \
        --local-dir .
    cd ..
    echo "   ✅ Checkpoint downloaded"
fi

# Verify checkpoint
echo ""
echo "5. Verifying setup..."
if [ -f "$CHECKPOINT_PATH" ]; then
    SIZE=$(stat -f%z "$CHECKPOINT_PATH" 2>/dev/null || stat -c%s "$CHECKPOINT_PATH" 2>/dev/null)
    echo "   ✅ Checkpoint: $(numfmt --to=iec-i --suffix=B $SIZE 2>/dev/null || echo ${SIZE} bytes)"
else
    echo "   ❌ Checkpoint not found!"
    exit 1
fi

if [ -f "02.HowardShore-TheShire.flac" ]; then
    echo "   ✅ Audio file: 02.HowardShore-TheShire.flac"
else
    echo "   ⚠️  Audio file not found: 02.HowardShore-TheShire.flac"
    echo "      Please upload the audio file to test transcription"
fi

# Check CUDA
echo ""
echo "6. Checking CUDA availability..."
python3 -c "import torch; print('   ✅ CUDA available:', torch.cuda.is_available()); print('   Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

echo ""
echo "==========================================="
echo "✅ Setup Complete!"
echo "==========================================="
echo ""
echo "Next steps:"
echo "1. Run test: python3 test_yourmt3.py"
echo "2. Expected output: > 0 notes (vs MT3's 0 notes)"
echo "3. MIDI will be saved to: model_output/TheShire_YourMT3.mid"
echo ""
echo "Estimated inference time: 2-3 minutes on A10G GPU"
echo ""
