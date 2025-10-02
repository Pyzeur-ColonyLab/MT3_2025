#!/bin/bash
#
# Setup Jupyter notebook environment for YourMT3 testing
#

set -e

echo "==========================================="
echo "Jupyter Notebook Setup for YourMT3"
echo "==========================================="

echo ""
echo "Installing Jupyter and dependencies..."

# Install Jupyter and widgets
pip install jupyter ipywidgets --quiet
pip install matplotlib scipy --quiet

# Enable widgets extension
jupyter nbextension enable --py widgetsnbextension --sys-prefix

echo "   ✅ Jupyter installed"

# Install FluidSynth for MIDI playback
echo ""
echo "Installing FluidSynth for MIDI playback..."
if command -v apt-get &> /dev/null; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq fluidsynth fluid-soundfont-gm
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
echo "To start Jupyter notebook:"
echo "1. jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser"
echo "2. Open: YourMT3_Interactive_Test.ipynb"
echo "3. Run all cells to start testing!"
echo ""
echo "Features:"
echo "- 🎵 Select and transcribe audio files"
echo "- 🎼 Visualize piano roll"
echo "- 🎧 Play original audio"
echo "- 🎹 Play generated MIDI"
echo "- 📊 Compare results"
echo ""
