#!/bin/bash
#
# Setup dependencies for Stems PoC Test
# Installs Demucs for stem separation
#

set -e

echo "=========================================="
echo "Stems PoC Test - Setup Script"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "setup_yourmt3_brev.sh" ]; then
    echo "❌ Error: Run this script from the MT3 directory"
    echo "   cd ~/MT3_2025"
    exit 1
fi

echo "📦 Installing Demucs for stem separation..."
echo ""

# Install Demucs (includes torch dependencies)
pip install demucs --quiet

# Verify installation
if command -v demucs &> /dev/null; then
    echo "✅ Demucs installed successfully!"
    demucs_version=$(pip show demucs | grep Version | awk '{print $2}')
    echo "   Version: $demucs_version"
else
    echo "❌ Demucs installation failed!"
    echo "   Try manually: pip install demucs"
    exit 1
fi

# Check Jupyter notebook dependencies (should already be installed)
echo ""
echo "🔍 Checking Jupyter dependencies..."

missing_deps=()

# Check ipywidgets
if ! python3 -c "import ipywidgets" 2>/dev/null; then
    missing_deps+=("ipywidgets")
fi

# Check matplotlib
if ! python3 -c "import matplotlib" 2>/dev/null; then
    missing_deps+=("matplotlib")
fi

# Check scipy
if ! python3 -c "import scipy" 2>/dev/null; then
    missing_deps+=("scipy")
fi

# Install missing dependencies
if [ ${#missing_deps[@]} -ne 0 ]; then
    echo "   Installing missing: ${missing_deps[*]}"
    pip install "${missing_deps[@]}" --quiet
    echo "   ✅ Dependencies installed"
else
    echo "   ✅ All dependencies already installed"
fi

echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Open Brev Jupyter interface in your browser"
echo "2. Navigate to: Stems_PoC_Test.ipynb"
echo "3. Run all cells to test stems vs full mix"
echo ""
echo "What the PoC will do:"
echo "   🎵 Transcribe full mix (baseline)"
echo "   🎸 Separate into 4 stems (bass, drums, other, vocals)"
echo "   📊 Transcribe each stem separately"
echo "   📈 Compare accuracy improvements"
echo ""
echo "Expected results:"
echo "   >10% improvement → Proceed with fine-tuning ✅"
echo "   5-10% improvement → Investigate further ⚠️"
echo "   <5% improvement → Reconsider approach ❌"
echo ""
