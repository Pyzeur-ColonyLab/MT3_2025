# YourMT3 Workflow Documentation

Complete technical documentation of the YourMT3 music transcription system workflow.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Complete Workflow](#complete-workflow)
3. [Architecture Details](#architecture-details)
4. [Instrument Recognition](#instrument-recognition)
5. [Extending the Model](#extending-the-model)

---

## System Overview

YourMT3 is an end-to-end audio-to-MIDI transcription system that converts polyphonic music audio into MIDI files with multi-instrument recognition.

**Model Configuration Used**: `mc13_full_plus_256`
- **Architecture**: Perceiver-TF encoder + MoE (Mixture of Experts) + Multi-channel T5 decoder
- **Checkpoint**: `mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops`
- **Precision**: 16-bit floating point (bf16-mixed)
- **Decoding Channels**: 13 parallel channels (12 instrument classes + drums)

---

## Complete Workflow

### Phase 1: Audio Input Processing

```
Input Audio (any format: MP3, WAV, FLAC, etc.)
    ↓
Convert to Mono (average all channels)
    ↓
Resample to 16kHz (model's native sample rate)
    ↓
Slice into segments (input_frames per segment)
    ↓
Convert to float32 tensors
    ↓
Move to GPU (if available)
```

**Technical Details:**
- **Sample Rate**: 16,000 Hz
- **Audio Codec**: Spectrogram (not mel-spectrogram)
- **Hop Length**: 300 frames
- **Input Frames**: Variable length, sliced into fixed segments
- **Batch Processing**: Process multiple segments in parallel (batch size = 8)

**Code Reference:** `model_helper.py:126-137`

---

### Phase 2: Model Inference

```
Audio Segments (GPU tensors)
    ↓
Perceiver-TF Encoder
    ├─ Self-Cross-Attention (SCA) with query residual
    ├─ Mixture of Experts (8 experts, top-2 routing)
    ├─ RoPE (Rotary Position Embedding) with partial PE
    ├─ RMS normalization
    └─ SiLU activation
    ↓
Latent Representation (26 latents, d_latent dimension)
    ↓
Multi-channel T5 Decoder (13 parallel channels)
    ├─ Channel 0: Piano
    ├─ Channel 1: Chromatic Percussion
    ├─ Channel 2: Organ
    ├─ Channel 3: Guitar
    ├─ Channel 4: Bass
    ├─ Channel 5: Strings
    ├─ Channel 6: Brass
    ├─ Channel 7: Reed
    ├─ Channel 8: Pipe
    ├─ Channel 9: Synth Lead
    ├─ Channel 10: Synth Pad
    ├─ Channel 11: Singing Voice
    └─ Channel 12: Drums
    ↓
Token Sequences (per channel, max 256 tokens per channel)
```

**Technical Details:**
- **Encoder Type**: Perceiver-TF (Perceiver with Transformer)
- **Decoder Type**: Multi-T5 (13 parallel T5 decoders)
- **Latents**: 26 learnable latent vectors
- **MoE Configuration**: 8 experts, top-2 selection
- **Widening Factor**: 4 (feed-forward layer expansion)
- **Hidden Activation**: SiLU (Sigmoid Linear Unit)
- **Position Encoding**: RoPE (Rotary Position Embedding)

**Code Reference:** `model_helper.py:140-142`, `config/task.py:95-103`

---

### Phase 3: Post-Processing & Detokenization

```
Token Sequences (13 channels × N segments)
    ↓
Detokenize per channel
    ├─ Convert tokens to note events
    ├─ Handle tie events (note continuations)
    └─ Add time offsets based on segment start times
    ↓
Note Events (per channel)
    ├─ onset: note start time (seconds)
    ├─ offset: note end time (seconds)
    ├─ pitch: MIDI pitch (0-127)
    ├─ velocity: note velocity (0-127)
    └─ program: instrument program number (0-127 or 128 for drums)
    ↓
Merge tied notes (notes spanning multiple segments)
    ↓
Mix all channels (combine all instrument tracks)
    ↓
Write MIDI file
    ├─ Create MIDI tracks per instrument
    ├─ Apply inverse vocabulary mapping
    └─ Save as .mid file
```

**Technical Details:**
- **Tokenizer**: Task-specific tokenizer from TaskManager
- **Max Tokens per Channel**: 256 tokens
- **Vocabulary Size**: ~1536 tokens (event types + pitch values + timing)
- **Tie Handling**: Automatic merging of notes across segment boundaries
- **MIDI Output**: Multi-track MIDI file with instrument assignments

**Code Reference:** `model_helper.py:144-166`

---

## Architecture Details

### Perceiver-TF Encoder

**Purpose**: Compress variable-length audio spectrograms into fixed-size latent representations.

**Components**:
1. **Pre-encoder**: Convolutional layer for initial audio feature extraction
2. **Cross-Attention**: Maps audio features to latent space (26 latents)
3. **Self-Attention**: Processes latent representations
4. **Mixture of Experts (MoE)**:
   - 8 expert networks
   - Top-2 routing (each token routed to 2 experts)
   - Widening factor: 4
5. **Normalization**: RMS (Root Mean Square) normalization
6. **Activation**: SiLU (Sigmoid Linear Unit)

**Advantages over T5 encoder**:
- Handles variable-length inputs efficiently
- Reduces computational cost through latent bottleneck
- Better performance on audio spectrograms

### Multi-channel T5 Decoder

**Purpose**: Generate MIDI events for each instrument class in parallel.

**Configuration**:
- **13 parallel decoders** (one per instrument class)
- **Max sequence length**: 256 tokens per channel
- **Shared parameters**: All decoders share the same T5 weights
- **Channel-specific outputs**: Each decoder specializes in its instrument class

**Why Multi-channel?**
- **Parallel processing**: All instruments decoded simultaneously
- **No interference**: Each instrument has dedicated attention
- **Better polyphony**: No confusion between simultaneous instruments
- **Efficient**: Reuses T5 architecture with minimal overhead

### Vocabulary Mapping

**Token Types**:
1. **Note Events**: `note_on`, `note_off`, `tie`
2. **Pitch Values**: 0-127 (MIDI pitch range)
3. **Time Shifts**: Quantized time offsets
4. **Velocity Values**: 0-127 (note velocity)
5. **Program Changes**: Instrument selection within class

**Vocabulary Presets**:
- `MT3_FULL_PLUS`: 34 instrument classes + drums + singing voice (used by our model)
- `GM_INSTR_CLASS_PLUS`: 11 instrument classes + drums + singing voice
- `GM_INSTR_FULL`: 128 GM instruments (all General MIDI instruments)

---

## Instrument Recognition

### Model Configuration: `mc13_full_plus_256`

This configuration uses **MT3_FULL_PLUS** vocabulary with 13 decoding channels.

### Complete Instrument List

The model recognizes **36 instrument classes** organized into 13 channels:

#### Channel 0: Piano (2 classes)
1. **Acoustic Piano** - Programs: 0, 1, 3, 6, 7
   - Acoustic Grand Piano (0)
   - Bright Acoustic Piano (1)
   - Honky-tonk Piano (3)
   - Harpsichord (6)
   - Clavinet (7)

2. **Electric Piano** - Programs: 2, 4, 5
   - Electric Grand Piano (2)
   - Electric Piano 1 (4)
   - Electric Piano 2 (5)

#### Channel 1: Chromatic Percussion (1 class)
3. **Chromatic Percussion** - Programs: 8-15
   - Celesta (8)
   - Glockenspiel (9)
   - Music Box (10)
   - Vibraphone (11)
   - Marimba (12)
   - Xylophone (13)
   - Tubular Bells (14)
   - Dulcimer (15)

#### Channel 2: Organ (1 class)
4. **Organ** - Programs: 16-23
   - Drawbar Organ (16)
   - Percussive Organ (17)
   - Rock Organ (18)
   - Church Organ (19)
   - Reed Organ (20)
   - Accordion (21)
   - Harmonica (22)
   - Tango Accordion (23)

#### Channel 3: Guitar (3 classes)
5. **Acoustic Guitar** - Programs: 24-25
   - Acoustic Guitar (nylon) (24)
   - Acoustic Guitar (steel) (25)

6. **Clean Electric Guitar** - Programs: 26-28
   - Electric Guitar (jazz) (26)
   - Electric Guitar (clean) (27)
   - Electric Guitar (muted) (28)

7. **Distorted Electric Guitar** - Programs: 29-31
   - Overdriven Guitar (29)
   - Distortion Guitar (30)
   - Guitar Harmonics (31)

#### Channel 4: Bass (2 classes)
8. **Acoustic Bass** - Programs: 32, 35
   - Acoustic Bass (32)
   - Fretless Bass (35)

9. **Electric Bass** - Programs: 33, 34, 36, 37, 38, 39
   - Electric Bass (finger) (33)
   - Electric Bass (pick) (34)
   - Slap Bass 1 (36)
   - Slap Bass 2 (37)
   - Synth Bass 1 (38)
   - Synth Bass 2 (39)

#### Channel 5: Strings (10 classes)
10. **Violin** - Program: 40
11. **Viola** - Program: 41
12. **Cello** - Program: 42
13. **Contrabass** - Program: 43
14. **Orchestral Harp** - Program: 46
15. **Timpani** - Program: 47
16. **String Ensemble** - Programs: 44, 45, 48, 49
    - Tremolo Strings (44)
    - Pizzicato Strings (45)
    - String Ensemble 1 (48)
    - String Ensemble 2 (49)
17. **Synth Strings** - Programs: 50, 51
    - Synth Strings 1 (50)
    - Synth Strings 2 (51)
18. **Choir and Voice** - Programs: 52, 53, 54
    - Choir Aahs (52)
    - Voice Oohs (53)
    - Synth Choir (54)
19. **Orchestra Hit** - Program: 55

#### Channel 6: Brass (4 classes)
20. **Trumpet** - Programs: 56, 59
    - Trumpet (56)
    - Muted Trumpet (59)
21. **Trombone** - Program: 57
22. **Tuba** - Program: 58
23. **French Horn** - Program: 60
24. **Brass Section** - Programs: 61, 62, 63
    - Brass Section (61)
    - Synth Brass 1 (62)
    - Synth Brass 2 (63)

#### Channel 7: Reed (4 classes)
25. **Soprano/Alto Sax** - Programs: 64, 65
    - Soprano Sax (64)
    - Alto Sax (65)
26. **Tenor Sax** - Program: 66
27. **Baritone Sax** - Program: 67
28. **Oboe** - Program: 68
29. **English Horn** - Program: 69
30. **Bassoon** - Program: 70
31. **Clarinet** - Program: 71

#### Channel 8: Pipe (1 class)
32. **Pipe** - Programs: 72-79
    - Piccolo (72)
    - Flute (73)
    - Recorder (74)
    - Pan Flute (75)
    - Bottle Blow (76)
    - Shakuhachi (77)
    - Whistle (78)
    - Ocarina (79)

#### Channel 9: Synth Lead (1 class)
33. **Synth Lead** - Programs: 80-87
    - Lead 1 (square) (80)
    - Lead 2 (sawtooth) (81)
    - Lead 3 (calliope) (82)
    - Lead 4 (chiff) (83)
    - Lead 5 (charang) (84)
    - Lead 6 (voice) (85)
    - Lead 7 (fifths) (86)
    - Lead 8 (bass + lead) (87)

#### Channel 10: Synth Pad (1 class)
34. **Synth Pad** - Programs: 88-95
    - Pad 1 (new age) (88)
    - Pad 2 (warm) (89)
    - Pad 3 (polysynth) (90)
    - Pad 4 (choir) (91)
    - Pad 5 (bowed) (92)
    - Pad 6 (metallic) (93)
    - Pad 7 (halo) (94)
    - Pad 8 (sweep) (95)

#### Channel 11: Singing Voice (2 classes)
35. **Singing Voice** - Program: 100
36. **Singing Voice (chorus)** - Program: 101

#### Channel 12: Drums (17 drum notes)
**GM Drum Notes** (assigned to MIDI channel 10, program 128 internally):
- **Kick Drum** - Notes: 36, 35
- **Snare X-stick** - Notes: 37, 2
- **Snare Drum** - Notes: 38, 40
- **Closed Hi-Hat** - Notes: 42, 44, 22
- **Open Hi-Hat** - Notes: 46, 26
- **Cowbell** - Note: 56
- **High Floor Tom** - Note: 43
- **Low Floor Tom** - Note: 41
- **Low Tom** - Note: 45
- **Low-Mid Tom** - Note: 47
- **Mid Tom** - Note: 48
- **Low Tom (Rim)** - Note: 50
- **Mid Tom (Rim)** - Note: 58
- **Ride** - Note: 51
- **Ride (Bell)** - Note: 53
- **Ride (Edge)** - Note: 59
- **Chinese Cymbal** - Note: 52
- **Crash Cymbal** - Notes: 49, 57
- **Splash Cymbal** - Note: 55

### Instruments NOT Supported

The following GM instruments (96-127) are **not included** in the MT3_FULL_PLUS vocabulary:

**Sound Effects** (Programs 96-127):
- FX 1-8 (rain, soundtrack, crystal, atmosphere, brightness, goblins, echoes, sci-fi)
- Ethnic instruments (Sitar, Banjo, Shamisen, Koto, Kalimba, Bagpipe, Fiddle, Shanai)
- Percussive effects (Tinkle Bell, Agogo, Steel Drums, Woodblock, Taiko Drum, Melodic Tom)
- Synth Drum, Reverse Cymbal, Guitar Fret Noise, Breath Noise
- Environmental sounds (Seashore, Bird Tweet, Telephone Ring, Helicopter, Applause, Gunshot)

**Why excluded?**
These instruments are rare in typical music training datasets and would increase model complexity without significant practical benefit.

---

## Extending the Model

### Can the Instrument List Be Extended?

**Short Answer**: Yes, but it requires **retraining** the model, not just fine-tuning.

### Why Retraining is Required

The instrument vocabulary is **baked into the model architecture** at multiple levels:

1. **Decoder Channels**: The model has 13 fixed decoding channels, one per instrument class group
2. **Vocabulary Mapping**: Token embeddings are trained for specific program numbers
3. **Program-to-Channel Mapping**: `program2channel_vocab_source` defines which programs decode on which channels
4. **Output Layer**: Final layer size matches the vocabulary size

**You cannot** simply add new instruments by:
- Modifying the vocabulary file only
- Fine-tuning the existing checkpoint
- Adding new MIDI program numbers

### How to Extend (Retraining Required)

#### Option 1: Modify Existing Vocabulary

**Use Case**: Add instruments within the 96-127 range that are currently excluded.

**Steps**:

1. **Edit Vocabulary** (`amt/src/config/vocabulary.py`):
```python
# Add to MT3_FULL dictionary
MT3_FULL = {
    # ... existing entries ...
    "Sitar": [104],
    "Banjo": [105],
    "Bagpipe": [109],
    # ... add your instruments ...
}
```

2. **Update Task Configuration** (`amt/src/config/task.py`):
```python
# Ensure your task uses the updated vocabulary
task_cfg = {
    "your_custom_task": {
        "name": "your_custom_task",
        "train_program_vocab": MT3_FULL,  # or your custom vocab
        "train_drum_vocab": drum_vocab_presets["gm"],
        "num_decoding_channels": 13,  # or adjust if needed
        "max_note_token_length_per_ch": 256,
    }
}
```

3. **Prepare Training Data**:
   - Collect MIDI files with annotations for your new instruments
   - Ensure MIDI files use correct program numbers (104, 105, 109, etc.)
   - Create paired audio-MIDI dataset

4. **Retrain Model** (`amt/src/train.py`):
```bash
python train.py your_custom_task \
    -p your_project \
    -tk your_custom_task \
    -enc perceiver-tf \
    -dec multi-t5 \
    -nl 26 \
    # ... other training arguments ...
```

**Challenges**:
- Requires large dataset (~1000+ hours of audio with MIDI annotations)
- Training takes weeks on multiple GPUs (A100 recommended)
- Requires significant computational resources

#### Option 2: Create New Channel

**Use Case**: Add a completely new instrument category (e.g., "Ethnic Instruments").

**Steps**:

1. **Increase Decoding Channels**:
```python
# In task.py
"mc14_custom": {
    "name": "mc14_custom",
    "num_decoding_channels": 14,  # was 13, now 14
    "train_program_vocab": CUSTOM_VOCAB_WITH_NEW_CLASS,
    # ...
}
```

2. **Define New Vocabulary**:
```python
# In vocabulary.py
ETHNIC_INSTRUMENTS = {
    "Sitar": [104],
    "Banjo": [105],
    "Shamisen": [106],
    "Koto": [107],
    "Bagpipe": [109],
}

MT3_FULL_PLUS_ETHNIC = MT3_FULL_PLUS.copy()
MT3_FULL_PLUS_ETHNIC["Ethnic"] = list(range(104, 112))
```

3. **Update Channel Mapping**:
```python
program2channel_vocab_source = {
    "Piano": 0,
    # ... existing mappings ...
    "Singing Voice": 11,
    "Drums": 12,
    "Ethnic": 13,  # new channel
}
```

4. **Retrain from Scratch**:
   - Architecture changes require full retraining
   - Cannot load existing checkpoint (channel mismatch)
   - Requires complete dataset with new instrument annotations

#### Option 3: Fine-tune for Better Accuracy (NOT for new instruments)

**Use Case**: Improve accuracy on existing instruments for your specific music genre.

**What you CAN do**:
- Fine-tune existing checkpoint on genre-specific data (classical, jazz, rock, etc.)
- Improve accuracy for specific instruments already in the vocabulary
- Adapt to specific audio characteristics (microphone types, room acoustics, etc.)

**What you CANNOT do**:
- Add new instrument classes not in the original vocabulary
- Change the number of decoding channels
- Modify program number mappings

**Fine-tuning Example**:
```bash
python train.py mc13_full_plus_256 \
    -p fine_tune_classical \
    -tk mc13_full_plus_256 \
    --resume_from_checkpoint /path/to/pretrained.ckpt \
    --learning_rate 1e-5 \  # lower LR for fine-tuning
    --epochs 10 \  # fewer epochs
    # ... other arguments ...
```

### Practical Recommendations

**If you need new instruments**:
1. **Use existing similar instruments**: If you need "Sitar", try "Acoustic Guitar" (similar timbre)
2. **Post-process MIDI**: Change program numbers after transcription
3. **Combine outputs**: Use multiple specialized models (one for Western, one for Ethnic)

**If you must retrain**:
1. **Start with small experiments**: Test with subset of data first
2. **Use transfer learning**: Start from existing checkpoint (if architectures match)
3. **Budget resources**: Retraining YourMT3 requires significant GPU time and data

**If accuracy is the issue**:
1. **Fine-tune on your genre**: Much faster than full retraining
2. **Augment training data**: Add your specific music style to training set
3. **Adjust inference parameters**: Try different batch sizes, onset tolerances

---

## Summary

### Workflow Overview
```
Audio Input → Preprocessing (16kHz, mono, spectrogram)
           ↓
Perceiver-TF Encoder (26 latents, MoE, RoPE)
           ↓
Multi-channel T5 Decoder (13 parallel channels)
           ↓
Detokenization (tokens → note events → MIDI)
           ↓
MIDI Output (multi-track, instrument-separated)
```

### Instrument Support
- **36 instrument classes** across 13 decoding channels
- Based on **General MIDI** standard (programs 0-101)
- **17 drum note types** (GM Drum Notes)
- Excludes GM programs 96-127 (sound effects, rare ethnic instruments)

### Extension Limitations
- **Cannot** add new instruments without retraining
- **Can** fine-tune for better accuracy on existing instruments
- **Requires** full retraining with new vocabulary and datasets for new instruments
- **Practical alternative**: Use existing similar instruments and post-process MIDI

---

**Model Reference**:
- Paper: [YourMT3: Multi-instrument Music Transcription](https://arxiv.org/abs/2407.04822)
- HuggingFace Space: [mimbres/YourMT3](https://huggingface.co/spaces/mimbres/YourMT3)
- Architecture: Perceiver-TF + MoE + Multi-channel T5

**Code References**:
- Vocabulary: `yourmt3_space/amt/src/config/vocabulary.py`
- Task Config: `yourmt3_space/amt/src/config/task.py`
- Model Helper: `yourmt3_space/model_helper.py`
- Training Script: `yourmt3_space/amt/src/train.py`

Assisted by Claude Code
