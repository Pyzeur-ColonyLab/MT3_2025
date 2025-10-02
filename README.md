# YourMT3 Music Transcription

Audio-to-MIDI transcription using YourMT3+ model on NVIDIA GPU.

## 🎯 Quick Start

### Setup (First Time)

```bash
# Clone repo
git clone https://github.com/Pyzeur-ColonyLab/MT3_2025.git
cd MT3_2025

# Login to HuggingFace
huggingface-cli login

# Run setup
bash setup_yourmt3_brev.sh
```

### Test Transcription

```bash
# Command line test
python3 test_yourmt3.py

# Interactive Jupyter notebook
bash setup_jupyter.sh
# Then open YourMT3_Interactive_Test.ipynb in Brev
```

## 📁 Files

- **test_yourmt3.py** - Command line transcription script
- **YourMT3_Interactive_Test.ipynb** - Interactive notebook with audio playback
- **setup_yourmt3_brev.sh** - Automated setup for Brev
- **setup_jupyter.sh** - Jupyter dependencies setup

## 📚 Documentation

- [Quick Start Guide](README_YOURMT3.md)
- [Complete Setup](YOURMT3_SETUP.md)
- [Technical Analysis](YOURMT3_ANALYSIS.md)
- [Jupyter Notebook Guide](README_JUPYTER.md)

## ✅ Tested Results

**Test**: "The Shire" by Howard Shore (149s orchestral)

```
Duration: 149.48s
Notes: 1,425
Instruments: 9
Inference time: 17.8s (A10G GPU)
```

**Quality**: ~6/10
- ✅ Main melody recognizable
- ⚠️ Some instrument confusion
- ⚠️ Polyphonic sections simplified

## 🔧 Requirements

- Python 3.10+
- CUDA-capable GPU (recommended)
- HuggingFace account (for checkpoint download)

## 📖 Model

YourMT3+ (2024) - Multi-instrument music transcription
- Paper: https://arxiv.org/abs/2407.04822
- Demo: https://huggingface.co/spaces/mimbres/YourMT3
- Architecture: Perceiver-TF + MoE + Multi-channel T5

---

Assisted by Claude Code
