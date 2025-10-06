# 🎵 MT3 Music Transcription System

**Audio-to-MIDI transcription with YourMT3, stem separation, and instrument recognition**

This repository provides tools for accurate music-to-MIDI transcription using state-of-the-art models, with advanced features for improving accuracy and identifying actual instruments.

---

## 🚀 Quick Start (Turnkey Setup)

### Prerequisites
- **Hardware**: GPU instance (tested on Brev.dev with A10G)
- **OS**: Linux/macOS with CUDA support
- **Python**: 3.8+

### Setup

```bash
# Clone repository
git clone https://github.com/Pyzeur-ColonyLab/MT3_2025.git
cd MT3_2025

# Run setup script (installs YourMT3, Demucs, dependencies)
bash setup_yourmt3_brev.sh

# Optional: Setup Jupyter for interactive notebooks
bash setup_jupyter.sh
```

**That's it!** The system is ready to use.

---

## 📦 Features

### ✅ 1. MIDI Transcription (YourMT3)
Convert audio files to MIDI with multi-instrument support.

- **Model**: YPTF.MoE+Multi (noPS) - 536M parameters
- **Input**: MP3, WAV, FLAC, M4A audio files
- **Output**: MIDI files with instrument detection
- **Performance**: ~2-3 minutes for 3-minute audio on A10G GPU

**Usage**:
```bash
cd yourmt3_space
python transcribe_audio.py --audio your_music.mp3
```

Or use the Jupyter notebook: `Stems_PoC_Test_Fixed.ipynb`

**Setup on new instance**:
```bash
bash setup_yourmt3_brev.sh
bash setup_jupyter.sh  # Optional
```

---

### ✅ 2. Stems-Based Transcription (PoC Validated)
Improve transcription accuracy using stem separation.

**Pipeline**: Audio → Demucs (4 stems) → YourMT3 per stem → Merge

**Setup**:
```bash
bash setup_poc.sh  # Installs Demucs
```

**Usage**: Open `Stems_PoC_Test_Fixed.ipynb` in Jupyter
1. Select audio file
2. Run transcription
3. Compare full mix vs stems results

**Results**: Validated approach for complex polyphonic music

📖 **Documentation**: See original PoC guide in git history for detailed analysis

---

### 🚧 3. Instrument Recognition (In Development)
Identify actual instruments from MIDI transcription using audio timbre analysis.

**Approach**: MIDI notes → Audio segments → YAMNet timbre matching → Instrument labels

**Target**:
- 85% accuracy
- 25 instrument categories (electric guitar, piano, kick drum, etc.)
- <10 min processing time

**Outputs**:
- Aggregated instrument percentages
- Timeline of instrument presence
- Enhanced MIDI with correct instrument labels

📖 **Full Specification**: [instrument_recognition/SPECIFICATION.md](./instrument_recognition/SPECIFICATION.md)

**Setup on new instance**:
```bash
# Install instrument recognition dependencies
pip install -r instrument_recognition/requirements.txt

# Launch Jupyter
jupyter notebook Instrument_Recognition_PoC.ipynb
```

**Status**: Phase 0 (MIDI transcription) complete, Phase 1-5 in development

---

## 📂 Repository Structure

```
MT3_2025/
├── README.md                           # This file
├── setup_yourmt3_brev.sh              # Main setup script
├── setup_jupyter.sh                   # Jupyter setup
│
├── yourmt3_space/                     # YourMT3 installation
│   ├── model_helper.py                # Model loading utilities
│   ├── amt/                           # Core transcription engine
│   └── logs/2024/*/checkpoints/       # Pre-trained models (auto-downloaded)
│
├── Stems_PoC_Test_Fixed.ipynb        # Working stems transcription notebook
│
├── instrument_recognition/            # Instrument identification system (new)
│   └── SPECIFICATION.md               # Technical specification
│
└── .gitignore                         # Git ignore patterns
```

---

## 🛠️ Usage Examples

### Basic MIDI Transcription

**Command Line**:
```bash
cd yourmt3_space
python -m amt.inference.transcribe \
  --audio_path /path/to/music.mp3 \
  --output_dir ./output_midi
```

**Python API**:
```python
from model_helper import load_model_checkpoint, transcribe

# Load model
model = load_model_checkpoint(args=default_args, device="cuda")

# Transcribe
audio_info = {
    "filepath": "music.mp3",
    "track_name": "my_track",
    # ... (see notebook for full format)
}
midi_path = transcribe(model, audio_info)
```

**Jupyter Notebook**: Use `Stems_PoC_Test_Fixed.ipynb` for interactive use

---

### Stems-Based Transcription

```python
# 1. Separate stems with Demucs (in notebook)
from demucs.pretrained import get_model
from demucs.apply import apply_model

demucs_model = get_model('htdemucs')
# ... (see notebook for full implementation)

# 2. Transcribe each stem
for stem_name, stem_path in stems.items():
    midi_path = transcribe(model, stem_audio_info)

# 3. Merge MIDI files (recomposition)
merged_midi = merge_midi_files(stem_midi_paths)
```

**Full example**: `Stems_PoC_Test_Fixed.ipynb` Cell 13

---

## 📊 Model Information

### YourMT3 Models Available

The setup automatically downloads 5 pre-trained models:

| Model | Size | Type | Description |
|-------|------|------|-------------|
| **mc13_256_g4_all_v7_mt3f...nops** | 536M | noPS | Standard (current default) |
| mc13_256_g4_all_v7_mt3f...ps2 | 724M | PS | Pitch-shift trained (±2 semitones) |
| notask_all_cross_v6...nops | 518M | YMT3+ | Extended instrument set |
| ptf_all_cross...full_plus | 345M | Single | Single-channel decoder |
| ptf_mc13_256...nl26_sb | 517M | Multi PS | Multi-channel, no MoE |

**Switching models**: Edit `checkpoint` variable in Cell 7 of notebook

---

## 🎯 Use Cases

### Music Production
- Extract MIDI from audio recordings
- Analyze song structures and arrangements
- Create sheet music from audio

### Music Analysis
- Identify instruments in recordings
- Study composition techniques
- Generate training data for ML models

### Audio Processing
- Separate complex mixes into stems
- Improve polyphonic transcription accuracy
- Build instrument-specific models

---

## 🔧 Configuration

### Environment Variables
```bash
export CUDA_VISIBLE_DEVICES=0  # GPU selection
export YOURMT3_CACHE_DIR=./cache  # Model cache location
```

### Model Parameters (in notebook Cell 7)
```python
checkpoint = "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops@last.ckpt"
precision = '16'  # 16-bit precision (faster, recommended)
device = "cuda"   # or "cpu"
```

### Demucs Configuration
```python
demucs_model = get_model('htdemucs')  # 4-stem separation
# Alternatives: 'mdx_extra' (higher quality), 'htdemucs_6s' (6 stems)
```

---

## 📈 Performance Benchmarks

**Hardware**: Brev.dev A10G GPU (24GB VRAM)

| Task | Audio Length | Processing Time | Memory Usage |
|------|--------------|-----------------|--------------|
| Basic transcription | 3 min | ~2-3 min | ~8GB |
| Stems + transcription | 3 min | ~5-7 min | ~12GB |
| Instrument recognition | 3 min | ~8-10 min (target) | ~14GB |

---

## 🐛 Troubleshooting

### Model Download Issues
```bash
# Manual download if automatic fails
cd yourmt3_space/amt/logs/2024
# Models auto-download on first use
```

### CUDA Out of Memory
```python
# Reduce precision or batch size
precision = '16'  # Use 16-bit instead of 32-bit
# Or process shorter audio segments
```

### Jupyter Kernel Issues
```bash
# Restart kernel
jupyter notebook --ip=0.0.0.0 --no-browser
# Or: Kernel → Restart in Jupyter interface
```

### Import Errors
```bash
# Reinstall dependencies
bash setup_yourmt3_brev.sh
```

---

## 🔬 Advanced Features

### Custom Model Training
For fine-tuning YourMT3 on custom datasets, see original fine-tuning guide in git history.

### Batch Processing
```python
audio_files = glob.glob("audio/*.mp3")
for audio_file in audio_files:
    midi_path = transcribe(model, audio_info)
```

### MIDI Post-Processing
```python
import pretty_midi
midi = pretty_midi.PrettyMIDI(midi_path)
# Modify instruments, timing, velocity, etc.
midi.write("output_modified.mid")
```

---

## 📚 Documentation

- **Instrument Recognition**: [instrument_recognition/SPECIFICATION.md](./instrument_recognition/SPECIFICATION.md)
- **YourMT3 Repository**: [GitHub](https://github.com/mimbres/YourMT3)
- **Demucs**: [GitHub](https://github.com/facebookresearch/demucs)

---

## 🤝 Contributing

This is a research project. For questions or improvements, please open an issue on GitHub.

---

## 📄 License

This project uses:
- **YourMT3**: Apache 2.0 License
- **Demucs**: MIT License
- **This codebase**: See repository for details

---

## 🙏 Acknowledgments

- **YourMT3**: [mimbres/YourMT3](https://github.com/mimbres/YourMT3)
- **Demucs**: [facebookresearch/demucs](https://github.com/facebookresearch/demucs)
- **AudioSet/YAMNet**: Google Research

---

## 📞 Support

**Issues**: Open a GitHub issue
**Environment**: Tested on Brev.dev A10G instances

---

*Last updated: October 3, 2025*
*Repository: https://github.com/Pyzeur-ColonyLab/MT3_2025*
