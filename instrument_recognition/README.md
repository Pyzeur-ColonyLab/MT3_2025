# 🎸 Instrument Recognition System

Identify actual instruments in music by analyzing audio timbre and refining MIDI transcription.

---

## 🚀 Setup Instructions (GPU Instance)

### Prerequisites
- Brev.dev A10G instance (or equivalent GPU)
- YourMT3 already installed (from main MT3 setup)
- CUDA available
- Python 3.8+

### Installation Steps

**1. Navigate to MT3 directory**
```bash
cd MT3
```

**2. Install Instrument Recognition dependencies**
```bash
pip install -r instrument_recognition/requirements.txt
```

This installs:
- TensorFlow (for YAMNet model)
- TensorFlow Hub (model loading)
- Librosa (audio processing)
- CREPE (optional pitch detection)

**3. Verify installation**
```bash
python -c "import tensorflow as tf; import tensorflow_hub as hub; print('✅ TensorFlow:', tf.__version__)"
python -c "import librosa; print('✅ Librosa:', librosa.__version__)"
```

**4. Launch Jupyter**
```bash
jupyter notebook --ip=0.0.0.0 --no-browser
```

**5. Open notebook**
- Navigate to `Instrument_Recognition_PoC.ipynb`
- Upload audio files to MT3 directory
- Run cells in order

---

## 📋 Pipeline Overview

### Phase 0: MIDI Transcription ✅
- **Input**: Audio file (.mp3, .wav, etc.)
- **Process**: YourMT3 transcription
- **Output**: MIDI file with instrument guesses

### Phase 1: YAMNet Setup 🚧
- **Input**: MIDI + Audio
- **Process**: Load YAMNet model, create instrument mapping
- **Output**: Ready for timbre analysis

### Phase 2: Note Isolation 🚧
- **Input**: MIDI piano roll
- **Process**: Extract isolated notes, score quality
- **Output**: Clean audio segments per note

### Phase 3: Timbre Matching 🚧
- **Input**: Isolated note audio
- **Process**: YAMNet classification, aggregation
- **Output**: Instrument predictions per MIDI track

### Phase 4: Output Generation 🚧
- **Output B**: Aggregated percentages (e.g., "Electric Guitar: 45%, Piano: 30%")
- **Output C**: Timeline (e.g., "0-30s: Piano, Bass | 30-60s: Guitar, Drums")
- **Visualization**: Piano roll with correct instruments

### Phase 5: Evaluation 🚧
- Manual accuracy testing
- Error analysis
- Threshold tuning
- **Target**: 85% accuracy

---

## 📁 File Structure

```
instrument_recognition/
├── README.md                          # This file
├── SPECIFICATION.md                   # Full technical specification
├── requirements.txt                   # Python dependencies
├── Instrument_Recognition_PoC.ipynb   # Main notebook
├── yamnet_to_level2_mapping.json      # Instrument mapping (create in Phase 1)
├── evaluation_results/                # Test results (Phase 5)
└── reference_database/                # Optional: Custom instrument samples
```

---

## 🎯 Quick Start

**Step 1**: Install dependencies
```bash
pip install -r instrument_recognition/requirements.txt
```

**Step 2**: Open notebook
```bash
jupyter notebook Instrument_Recognition_PoC.ipynb
```

**Step 3**: Upload audio file to MT3 directory

**Step 4**: Run Phase 0 to transcribe audio to MIDI

**Step 5**: Continue with Phases 1-5 for instrument identification

---

## 🔧 Configuration

### Target Instrument Categories (25)

**Drums/Percussion (8)**:
- Kick Drum, Snare Drum, Closed Hi-hat, Open Hi-hat
- Crash Cymbal, Ride Cymbal, Tom Drum, Electronic Drum

**Bass (3)**:
- Electric Bass, Synth Bass, Acoustic Bass

**Guitar (3)**:
- Electric Guitar (Clean), Electric Guitar (Distorted), Acoustic Guitar

**Keys (4)**:
- Acoustic Piano, Electric Piano, Synthesizer Lead, Synthesizer Pad

**Other (7)**:
- Orchestral Strings, Brass, Woodwinds, Vocals, Vocal Sample, Sound Effects, Unknown

### Performance Targets
- **Accuracy**: ≥ 85%
- **Processing Time**: < 10 minutes for 3-minute audio
- **Note Coverage**: ≥ 80% of notes identified

---

## 📚 References

- **Specification**: See `SPECIFICATION.md` for full technical details
- **YAMNet**: [TensorFlow Hub](https://tfhub.dev/google/yamnet/1)
- **YourMT3**: [GitHub](https://github.com/mimbres/YourMT3)
- **AudioSet**: [Google Research](https://research.google.com/audioset/)

---

## 🐛 Troubleshooting

### TensorFlow Installation Issues
```bash
# If TensorFlow fails to install
pip install --upgrade pip
pip install tensorflow==2.13.0 --no-cache-dir
```

### CUDA Out of Memory
- Reduce batch size in YAMNet inference
- Process shorter audio segments
- Use CPU fallback (slower but works)

### YAMNet Model Download Issues
- Ensure internet connectivity
- Model auto-downloads on first use (~4MB)
- Check TensorFlow Hub cache: `~/.keras/datasets/`

---

*Last updated: October 6, 2025*
