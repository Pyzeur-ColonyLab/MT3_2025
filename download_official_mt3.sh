#!/bin/bash
#
# Download official MT3 checkpoint from Google Cloud Storage
#

set -e

echo "=========================================="
echo "MT3 Official Checkpoint Download"
echo "=========================================="

# Check if gsutil is available
if ! command -v gsutil &> /dev/null; then
    echo "Installing gsutil..."
    pip install gsutil --quiet
fi

# Create directory
CHECKPOINT_DIR="checkpoints/mt3_official"
mkdir -p "$CHECKPOINT_DIR"

echo ""
echo "Downloading MT3 ISMIR2021 checkpoint..."
echo "Source: gs://magentadata/models/mt3/checkpoints/mt3/"
echo "Target: $CHECKPOINT_DIR"
echo ""

# Download the checkpoint
# MT3 has several checkpoints, we'll use the main one
gsutil -m cp -r "gs://magentadata/models/mt3/checkpoints/mt3/*" "$CHECKPOINT_DIR/"

echo ""
echo "✅ Download complete!"
echo ""
echo "Checkpoint contents:"
ls -lh "$CHECKPOINT_DIR" | head -20

echo ""
echo "Next steps:"
echo "1. Run: python3 t5x_converter_fixed.py $CHECKPOINT_DIR"
echo "2. This will create: mt3_converted.pth"
echo "3. Fix dimensions: python3 fix_checkpoint_dimensions.py"
echo "4. Test: python3 test_real_music_transcription.py"
