#!/bin/bash
#
# Setup dependencies for YourMT3 Jupyter notebook (Brev edition)
#

set -e

echo "==========================================="
echo "YourMT3 Notebook Dependencies Setup"
echo "==========================================="
echo ""
echo "Note: Brev already has Jupyter installed!"
echo "Installing additional dependencies only..."
echo ""

# Install notebook dependencies
echo "Installing Python packages..."
pip install ipywidgets matplotlib scipy --quiet
echo "   ✅ ipywidgets, matplotlib, scipy installed"

# Install FluidSynth for MIDI playback
echo ""
echo "Installing FluidSynth for MIDI playback..."
if command -v apt-get &> /dev/null; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq fluidsynth fluid-soundfont-gm 2>&1 | grep -v "^WARNING"
    echo "   ✅ FluidSynth installed"
elif command -v yum &> /dev/null; then
    sudo yum install -y -q fluidsynth
    echo "   ✅ FluidSynth installed"
else
    echo "   ⚠️  Could not install FluidSynth automatically"
    echo "      MIDI playback may not work"
    echo "      Install manually: apt-get install fluidsynth"
fi

echo ""
echo "==========================================="
echo "✅ Setup Complete!"
echo "==========================================="
echo ""
echo "Next steps:"
echo "1. Open Brev Jupyter interface in your browser"
echo "2. Navigate to: YourMT3_Interactive_Test.ipynb"
echo "3. Run all cells to start testing!"
echo ""
echo "Features:"
echo "- 🎵 Select and transcribe audio files"
echo "- 🎼 Visualize piano roll"
echo "- 🎧 Play original audio"
echo "- 🎹 Play generated MIDI"
echo "- 📊 Compare results"
echo ""
