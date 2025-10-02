# 🎵 YourMT3 Interactive Jupyter Notebook

Interactive testing environment for YourMT3 music transcription model.

## 🎯 Features

- **🎵 Audio Selection**: Choose from available audio files
- **🎼 Transcription**: One-click transcription with YourMT3
- **📊 Visualization**: Piano roll view of transcribed notes
- **🎧 Audio Playback**: Listen to original audio
- **🎹 MIDI Playback**: Listen to generated MIDI (synthesized)
- **📈 Comparison**: Side-by-side quality assessment
- **💾 Export**: Download MIDI files for DAW use
- **🔄 Batch Testing**: Test multiple files at once

## 🚀 Quick Start

### 1. Setup (First Time Only)

```bash
# On Brev instance
cd ~/MT3_2025

# Install Jupyter and dependencies
bash setup_jupyter.sh
```

### 2. Access Brev Jupyter Interface

Brev provides a built-in Jupyter interface:

1. **Open Brev dashboard** in your browser
2. **Click "Jupyter" tab** or navigate to your instance's Jupyter URL
3. **Navigate to** `MT3_2025/` directory
4. **Open** `YourMT3_Interactive_Test.ipynb`

> **Note**: No need to start Jupyter manually - Brev handles this automatically!

## 📖 How to Use

### Basic Workflow

1. **Run Setup Cells** (1-3)
   - Imports and model loading (~15 seconds)

2. **Select Audio File** (Cell 4)
   - Choose from dropdown

3. **Click "Transcribe Audio"** (Cell 5)
   - Wait ~0.12s per second of audio
   - View detailed results

4. **Visualize** (Cell 6)
   - Interactive piano roll
   - Adjust time range with slider

5. **Play & Compare** (Cells 7-8)
   - Original audio playback
   - Generated MIDI playback

6. **Assess Quality** (Cell 9)
   - View comparison table
   - Answer quality questions

### Advanced Features

**Batch Testing** (Cell 11)
```python
# Test multiple files
batch_results = batch_transcribe(audio_files[:5])
```

**Custom Visualization**
```python
# Plot specific time range
plot_piano_roll(midi, max_time=60)  # First 60 seconds
```

## 🎼 Interpreting Results

### Statistics

- **Total Notes**: Number of MIDI notes generated
- **Instruments**: Number of detected instrument tracks
- **Note Density**: Notes per second (indicates complexity)

### Instrument Programs

| Program | Instrument Type |
|---------|----------------|
| 0 | Acoustic Grand Piano |
| 24 | Acoustic Guitar (nylon) |
| 25 | Acoustic Guitar (steel) |
| 32-39 | Bass |
| 40-47 | Strings |
| 48-55 | Ensemble |
| 56-63 | Brass |
| 64-71 | Reed |
| 128 | Drums (percussion) |

Full list: [MIDI Program Numbers](https://www.midi.org/specifications-old/item/gm-level-1-sound-set)

### Quality Assessment

**Good Transcription** (7-10/10):
- ✅ Main melody clearly recognizable
- ✅ Rhythm matches original
- ✅ Instrument detection accurate
- ✅ Minimal extra/missing notes

**Acceptable** (4-6/10):
- ⚠️ Melody mostly correct
- ⚠️ Some rhythm errors
- ⚠️ Some instrument confusion
- ⚠️ Some missing/extra notes

**Poor** (1-3/10):
- ❌ Melody unrecognizable
- ❌ Major rhythm errors
- ❌ Wrong instruments
- ❌ Many missing/extra notes

## 🎹 MIDI Playback

### Requirements

MIDI playback requires **FluidSynth**:

```bash
# Ubuntu/Debian
sudo apt-get install fluidsynth fluid-soundfont-gm

# Or use setup script
bash setup_jupyter.sh
```

### Troubleshooting

**No sound when playing MIDI:**
1. Check FluidSynth is installed: `which fluidsynth`
2. Check soundfont exists: `ls /usr/share/sounds/sf2/`
3. If missing, install: `sudo apt-get install fluid-soundfont-gm`

**Alternative: Download MIDI**
If playback doesn't work, download the MIDI file and play in your DAW:
```bash
# From notebook output
scp ubuntu@brev-xxx:~/MT3_2025/yourmt3_space/model_output/*.mid .
```

## 📊 Example Results

### Test Case: "The Shire" (Howard Shore)

```
Duration: 149.48s
Notes: 1,425
Instruments: 9

Top Instruments:
🎹 Acoustic Guitar (steel): 931 notes
🎹 Acoustic Guitar (nylon): 285 notes
🎹 Acoustic Grand Piano: 138 notes
🎺 Trumpet: 20 notes
🥁 Synth Drum: 9 notes
```

**Quality Score**: ~6/10
- ✅ Main melody recognizable
- ⚠️ Some instrument confusion
- ⚠️ Polyphonic sections simplified

## 🔧 Customization

### Test Different Audio

Upload any audio file to `~/MT3_2025/`:
```bash
# Upload from local machine
scp your_audio.mp3 ubuntu@brev-xxx:~/MT3_2025/

# Refresh notebook cell 4 to see new file
```

### Adjust Model Parameters

In cell 2, modify batch size for speed/memory tradeoff:
```python
# In model_helper.py line 141
# Default: bsz=8
# Lower for less memory: bsz=4
# Higher for speed (if VRAM allows): bsz=16
```

### Export Visualizations

```python
# Save piano roll as image
fig = plot_piano_roll(midi, max_time=60)
fig.savefig('piano_roll.png', dpi=300, bbox_inches='tight')
```

## 📝 Tips for Best Results

### Audio Quality

- **Supported formats**: MP3, WAV, FLAC, M4A, OGG
- **Recommended**: 16kHz or higher sample rate
- **Duration**: Works best with < 3 minutes
  - Longer files work but take more time
  - Very long files may hit memory limits

### Music Types

**Works Well** ✅:
- Solo instruments (piano, guitar)
- Small ensembles (2-4 instruments)
- Clear polyphony
- Acoustic instruments

**Challenging** ⚠️:
- Heavy distortion/effects
- Dense orchestral (>8 instruments)
- Extreme polyphony
- Electronic/synthesized sounds

## 🐛 Troubleshooting

### Common Issues

**Cell execution hangs:**
- Restart kernel: Kernel → Restart
- Re-run setup cells

**Out of memory:**
- Close other notebooks
- Reduce batch size in model
- Use shorter audio clips

**MIDI sounds wrong:**
- Check instrument assignments
- Try different soundfont
- Download MIDI and play in DAW

**No audio files appear:**
- Check files are in correct directory
- Re-run cell 4
- Upload new files to `~/MT3_2025/`

## 📚 References

- **YourMT3 Paper**: https://arxiv.org/abs/2407.04822
- **Demo**: https://huggingface.co/spaces/mimbres/YourMT3
- **MIDI Specification**: https://www.midi.org/specifications
- **Pretty MIDI Docs**: https://craffel.github.io/pretty-midi/

## 💡 Next Steps

After testing:

1. **Evaluate Quality**: Use multiple songs to assess accuracy
2. **Document Findings**: Note which types of music work best
3. **Integration**: Decide if quality meets production needs
4. **Alternative Models**: Test Basic-Pitch or other models for comparison

---

**Happy Testing!** 🎵

For issues or questions, check:
- `YOURMT3_SETUP.md` - Detailed setup guide
- `YOURMT3_ANALYSIS.md` - Technical analysis
- `README_YOURMT3.md` - Quick reference
