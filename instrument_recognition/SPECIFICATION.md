# 🎸 Instrument Recognition System - Technical Specification

**Version**: 1.0
**Date**: October 3, 2025
**Status**: Planning Phase

---

## 📋 Executive Summary

**Objective**: Identify actual instruments in music by analyzing audio timbre and refining MIDI transcription instrument labels.

**Approach**: Extract isolated notes from MIDI transcription → Compare audio timbre against instrument database → Replace MIDI instrument labels with detected instruments.

**Target Accuracy**: 85%
**Performance Budget**: Up to 10 minutes processing time
**Hardware**: GPU A10
**Priority Genres**: Electro, techno, rap, rock (universal capability)

---

## 🎯 Requirements

### Functional Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-1 | Transcribe audio to MIDI using YourMT3 | Critical |
| FR-2 | Extract audio segments corresponding to each MIDI note | Critical |
| FR-3 | Identify isolated notes (minimal polyphony) | High |
| FR-4 | Match audio timbre against instrument database | Critical |
| FR-5 | Generate aggregated instrument percentages (Output B) | Critical |
| FR-6 | Generate timeline of instrument presence (Output C) | Critical |
| FR-7 | Replace MIDI instrument labels with detected instruments | High |
| FR-8 | Visualize results (piano roll with correct instruments) | Medium |
| FR-9 | Support 25 instrument categories (Level 2 granularity) | Critical |
| FR-10 | Allow future extension with custom instruments | Medium |

### Non-Functional Requirements

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-1 | Processing time for 3-minute audio | < 10 minutes |
| NFR-2 | Instrument identification accuracy | ≥ 85% |
| NFR-3 | Note coverage (identified vs total) | ≥ 80% |
| NFR-4 | GPU memory usage | < 16GB |
| NFR-5 | Code maintainability | Well-documented, modular |

---

## 🏗️ System Architecture

### High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 0: Audio → MIDI Transcription (Simplified)                │
├─────────────────────────────────────────────────────────────────┤
│ Input: Audio file (.mp3, .wav, .flac, etc.)                    │
│ → Load YourMT3 model (mc13_256_g4_all_v7...nops)              │
│ → Transcribe audio to MIDI                                     │
│ → Save MIDI file with initial instrument guesses               │
│ Output: MIDI file + Audio file                                 │
│                                                                  │
│ Note: Simplified from Stems_PoC_Test_Fixed.ipynb               │
│       No stems separation, no comparison - just transcription   │
└─────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 1: Audio + MIDI Input & Setup                            │
├─────────────────────────────────────────────────────────────────┤
│ Input: Audio file + MIDI transcription from Phase 0            │
│ → Load MIDI with pretty_midi                                    │
│ → Load audio with torchaudio                                    │
│ → Extract all notes (start, end, pitch, velocity, instrument)  │
│ → Load YAMNet pre-trained model                                │
│ → Load instrument category mapping (521 → 25)                  │
│ Output: Ready for note-by-note analysis                        │
└─────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 2: Note Isolation & Audio Extraction                      │
├─────────────────────────────────────────────────────────────────┤
│ For each note in MIDI:                                          │
│   1. Extract audio segment [start_time - 50ms, end_time + 50ms]│
│   2. Compute isolation quality score:                           │
│      a) Temporal isolation: Count overlapping notes            │
│      b) Pitch isolation: Verify pitch with CREPE (optional)    │
│      c) Energy isolation: Check RMS energy level               │
│   3. Filter: Keep only notes with isolation_score > 0.6        │
│   4. Store: Best N samples per MIDI instrument (N=10-20)       │
│ Output: List of isolated note audio segments                   │
└─────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 3: Timbre Analysis & Instrument Matching                  │
├─────────────────────────────────────────────────────────────────┤
│ For each isolated note sample:                                  │
│   1. Preprocess audio (resample to 16kHz for YAMNet)           │
│   2. Extract YAMNet embeddings/predictions (521 classes)        │
│   3. Map YAMNet classes → Level 2 categories (user-defined)    │
│   4. Get top-K matching instruments with confidence scores      │
│   5. Aggregate: Majority vote across all notes per instrument  │
│ Output: Instrument predictions per MIDI instrument             │
└─────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 4: MIDI Relabeling & Output Generation                    │
├─────────────────────────────────────────────────────────────────┤
│ 1. Replace MIDI instrument programs with detected instruments  │
│    (update inst.program to match new instrument)               │
│                                                                  │
│ 2. Generate Output B: Aggregated Percentages                   │
│    Example: {"electric_guitar": 45%, "piano": 30%, ...}        │
│                                                                  │
│ 3. Generate Output C: Timeline of Instrument Presence          │
│    Example: [(0-10s, ["piano", "bass"]),                       │
│              (10-20s, ["piano", "drums", "guitar"])]           │
│                                                                  │
│ 4. Save enhanced MIDI with corrected instrument labels         │
│                                                                  │
│ 5. Generate visualization:                                      │
│    - Piano roll with instrument-colored notes                  │
│    - Timeline chart showing instrument presence                │
│                                                                  │
│ Output: Enhanced MIDI + Aggregated stats + Timeline + Viz      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Technical Design

### 1. Pre-trained Model Selection

**Primary Model: YAMNet** (Google's audio event classifier)

**Rationale**:
- ✅ Pre-trained on AudioSet (521 audio event classes)
- ✅ ~80-100 instrument-related classes included
- ✅ Fast inference (~50ms per audio segment)
- ✅ TensorFlow Hub integration, easy to use
- ✅ Good coverage for modern electronic instruments
- ✅ No training required, ready to use

**Model Details**:
- Input: 16kHz mono audio, variable length
- Output: 521-dimensional probability vector
- Architecture: MobileNet-based
- Size: ~3.7MB
- Source: TensorFlow Hub (`https://tfhub.dev/google/yamnet/1`)

**Optional Enhancement: CREPE** (Pitch detection)
- Use case: Validate MIDI pitch matches detected pitch
- Benefits: Filter out misaligned notes, improve isolation scoring
- Integration: Optional in Phase 2 (isolation quality scoring)

### 2. Instrument Category Mapping

**Level 2 Categories (25 instruments)**:

#### Drums/Percussion (8 categories)
1. Kick Drum
2. Snare Drum
3. Hi-hat (Closed)
4. Hi-hat (Open)
5. Crash Cymbal
6. Ride Cymbal
7. Tom Drum
8. Electronic Drum (808, 909, etc.)

#### Bass (3 categories)
9. Electric Bass
10. Synth Bass
11. Acoustic Bass

#### Guitar (3 categories)
12. Electric Guitar (Clean)
13. Electric Guitar (Distorted)
14. Acoustic Guitar

#### Keys (4 categories)
15. Piano (Acoustic)
16. Electric Piano
17. Synthesizer (Lead)
18. Synthesizer (Pad)

#### Other (7 categories)
19. Strings (Orchestral)
20. Brass
21. Woodwinds
22. Vocals
23. Vocal Sample/Chop
24. Sound Effects
25. Unknown/Other

**Mapping Strategy**:
- User will create mapping: YAMNet's 521 classes → 25 Level 2 categories
- Stored in JSON file: `yamnet_to_level2_mapping.json`
- Format: `{"Bass drum": ["Kick Drum"], "Snare drum": ["Snare Drum"], ...}`
- Multiple YAMNet classes can map to same Level 2 category
- One-to-many or many-to-one mappings supported

### 3. Note Isolation Algorithm

**Purpose**: Select notes that are acoustically isolated (minimal polyphony) for accurate timbre analysis.

**Algorithm**:
```python
def compute_isolation_score(note, all_notes, audio, sr):
    """
    Score how isolated a note is (0.0 = very polyphonic, 1.0 = monophonic)

    Returns: float in [0.0, 1.0]
    """

    # Component 1: Temporal Isolation (40% weight)
    # Count how many other notes overlap in time
    overlapping_notes = count_notes_overlapping_in_time(note, all_notes)
    temporal_score = 1.0 / (1.0 + overlapping_notes)

    # Component 2: Pitch Isolation (30% weight) - OPTIONAL
    # Verify MIDI pitch matches detected pitch using CREPE
    audio_segment = extract_audio(audio, note.start, note.end, sr)
    detected_pitch = detect_pitch_with_crepe(audio_segment)  # Optional
    pitch_difference = abs(midi_to_hz(note.pitch) - detected_pitch)
    pitch_match_score = 1.0 if pitch_difference < 10 else 0.5  # 10 Hz tolerance

    # Component 3: Energy Isolation (30% weight)
    # Check if note has sufficient energy (not too quiet)
    rms_energy = np.sqrt(np.mean(audio_segment**2))
    energy_threshold = 0.01  # Configurable
    energy_score = min(rms_energy / energy_threshold, 1.0)

    # Weighted combination
    isolation_score = (
        temporal_score * 0.4 +
        pitch_match_score * 0.3 +
        energy_score * 0.3
    )

    return isolation_score

# Usage
isolated_notes = [
    note for note in all_notes
    if compute_isolation_score(note, all_notes, audio, sr) > 0.6
]
```

**Key Parameters**:
- `isolation_threshold`: 0.6 (configurable, 0.5-0.8 recommended)
- `max_overlapping_notes`: 2 (reject if more than 2 notes overlap)
- `min_note_duration`: 0.1s (reject very short notes)
- `padding`: 50ms before/after note for attack/release

### 4. YAMNet Integration

**Implementation**:
```python
import tensorflow_hub as hub
import tensorflow as tf

# Load YAMNet model (once)
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

def classify_note_timbre(audio_segment, sr):
    """
    Classify instrument from audio segment using YAMNet

    Args:
        audio_segment: np.array, mono audio
        sr: int, sample rate (will resample to 16kHz)

    Returns:
        dict: {
            'top_classes': [(class_name, confidence), ...],
            'embeddings': np.array (1024-dim),
            'all_scores': np.array (521-dim)
        }
    """

    # Resample to 16kHz (YAMNet requirement)
    if sr != 16000:
        audio_segment = librosa.resample(audio_segment, orig_sr=sr, target_sr=16000)
        sr = 16000

    # Run YAMNet inference
    scores, embeddings, spectrogram = yamnet_model(audio_segment)

    # Average scores across time frames
    mean_scores = np.mean(scores, axis=0)  # (521,)

    # Get top-K classes
    top_k_indices = np.argsort(mean_scores)[-5:][::-1]
    top_classes = [(class_names[i], mean_scores[i]) for i in top_k_indices]

    return {
        'top_classes': top_classes,
        'embeddings': np.mean(embeddings, axis=0),  # (1024,)
        'all_scores': mean_scores
    }

# Apply to all isolated notes
for note in isolated_notes:
    audio_segment = extract_audio_for_note(note, audio, sr)
    classification = classify_note_timbre(audio_segment, sr)
    note.yamnet_prediction = classification
```

**Output Processing**:
- For each note: Get top-5 YAMNet classes with confidence scores
- Map YAMNet classes → Level 2 categories using user mapping
- Store both raw predictions and mapped categories

### 5. Aggregation Strategy

**Per-Instrument Voting**:
```python
def aggregate_instrument_predictions(notes_per_instrument):
    """
    Aggregate predictions across all notes for each MIDI instrument

    Args:
        notes_per_instrument: dict mapping MIDI instrument → list of notes

    Returns:
        dict: {midi_instrument_id: detected_instrument_name}
    """

    results = {}

    for midi_inst_id, notes in notes_per_instrument.items():
        # Collect all Level 2 predictions with confidence
        predictions = []
        for note in notes:
            if hasattr(note, 'yamnet_prediction'):
                for yamnet_class, confidence in note.yamnet_prediction['top_classes']:
                    # Map to Level 2 category
                    level2_category = yamnet_to_level2_mapping.get(yamnet_class)
                    if level2_category:
                        predictions.append((level2_category, confidence))

        # Weighted voting
        category_scores = {}
        for category, confidence in predictions:
            category_scores[category] = category_scores.get(category, 0) + confidence

        # Select instrument with highest total confidence
        if category_scores:
            best_instrument = max(category_scores.items(), key=lambda x: x[1])
            results[midi_inst_id] = {
                'instrument': best_instrument[0],
                'confidence': best_instrument[1] / len(notes),
                'vote_distribution': category_scores
            }
        else:
            results[midi_inst_id] = {
                'instrument': 'Unknown',
                'confidence': 0.0,
                'vote_distribution': {}
            }

    return results
```

### 6. Output Formats

#### Output B: Aggregated Percentages
```json
{
  "electric_guitar": 0.45,
  "piano": 0.30,
  "kick_drum": 0.12,
  "snare_drum": 0.08,
  "synth_bass": 0.05
}
```

**Calculation**: Based on note count per instrument / total notes

#### Output C: Timeline
```json
[
  {
    "start_time": 0.0,
    "end_time": 10.5,
    "instruments": ["piano", "electric_bass"]
  },
  {
    "start_time": 10.5,
    "end_time": 25.3,
    "instruments": ["piano", "electric_bass", "kick_drum", "snare_drum"]
  },
  {
    "start_time": 25.3,
    "end_time": 40.0,
    "instruments": ["electric_guitar", "kick_drum", "snare_drum", "synth_bass"]
  }
]
```

**Generation**: Group consecutive time windows, detect instrument presence changes

---

## 📊 Success Criteria

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **Accuracy** | ≥ 85% | Manual evaluation on test set (20 tracks) |
| **Note Coverage** | ≥ 80% | Isolated notes / total notes |
| **Processing Time** | < 10 min | For 3-minute audio on GPU A10 |
| **False Positive Rate** | < 15% | Incorrect instrument assignments |
| **Extensibility** | Add 5 custom instruments | Time to add new categories |

---

## 🔄 Extensibility & Future Enhancements

### Adding Custom Instruments

**Method 1: Update Mapping (No retraining)**
- Add new Level 2 category
- Map existing YAMNet classes to new category
- Update `yamnet_to_level2_mapping.json`
- **Time**: ~5 minutes

**Method 2: Reference Database (Your approach)**
- Collect 10-20 audio samples of new instrument
- Extract YAMNet embeddings for each sample
- Store in reference database
- Use cosine similarity for matching
- **Time**: ~1 hour per instrument

**Method 3: Fine-tune YAMNet (Advanced)**
- Collect labeled dataset (100+ samples per instrument)
- Fine-tune YAMNet's final layers
- Requires GPU training (few hours)
- **Time**: ~1 day per instrument family

### Phase 2 Enhancements (Future)

1. **Stem-aware refinement**: Leverage Demucs stems to narrow instrument search space
2. **Temporal consistency**: Smooth predictions over time (same instrument likely continues)
3. **Confidence thresholding**: Mark uncertain predictions as "Unknown"
4. **Multi-model ensemble**: Combine YAMNet + OpenL3 + Custom classifier
5. **Active learning**: User corrects mistakes → improve model

---

## 🛠️ Implementation Phases

### Phase 0: MIDI Transcription ✅
**Status**: Specified
**Deliverable**: `Instrument_Recognition_PoC.ipynb` with Phase 0 complete

- Simplified version of `Stems_PoC_Test_Fixed.ipynb`
- Audio → YourMT3 → MIDI
- No stems, no comparison, just transcription
- Button-based execution for easy testing

### Phase 1: Infrastructure & YAMNet Setup
**Status**: Pending
**Deliverable**: YAMNet model loaded and tested

Tasks:
- [ ] Install dependencies (tensorflow, tensorflow_hub, librosa)
- [ ] Load YAMNet model from TensorFlow Hub
- [ ] Test YAMNet on single audio sample
- [ ] Create `yamnet_to_level2_mapping.json` (user task)
- [ ] Implement mapping function

### Phase 2: Note Isolation
**Status**: Pending
**Deliverable**: Isolated note extraction working

Tasks:
- [ ] Implement `extract_audio_for_note()` function
- [ ] Implement `compute_isolation_score()` function
- [ ] Visualize isolated vs polyphonic notes
- [ ] Filter notes by isolation threshold
- [ ] Validate with manual listening

### Phase 3: Timbre Matching
**Status**: Pending
**Deliverable**: Instrument classification working

Tasks:
- [ ] Implement YAMNet inference pipeline
- [ ] Map YAMNet predictions → Level 2 categories
- [ ] Compute confidence scores per note
- [ ] Aggregate predictions per MIDI instrument
- [ ] Validate accuracy on test tracks

### Phase 4: MIDI Relabeling & Outputs
**Status**: Pending
**Deliverable**: Complete pipeline with all outputs

Tasks:
- [ ] Replace MIDI instrument programs with detected instruments
- [ ] Generate Output B: Aggregated instrument percentages
- [ ] Generate Output C: Timeline of instrument presence
- [ ] Create visualizations (piano roll, timeline chart)
- [ ] Save enhanced MIDI file

### Phase 5: Evaluation & Refinement
**Status**: Pending
**Deliverable**: Tuned system meeting accuracy targets

Tasks:
- [ ] Manual accuracy evaluation on 20 diverse tracks
- [ ] Error analysis (which instruments confused?)
- [ ] Tune isolation thresholds and parameters
- [ ] Optimize for 85% accuracy target
- [ ] Document limitations and edge cases

---

## 📁 File Structure

```
MT3/
├── instrument_recognition/
│   ├── SPECIFICATION.md                    # This document
│   ├── yamnet_to_level2_mapping.json       # User-created mapping (Phase 1)
│   ├── Instrument_Recognition_PoC.ipynb    # Main notebook
│   ├── evaluation_results/                 # Test results (Phase 5)
│   │   ├── accuracy_report.json
│   │   ├── confusion_matrix.png
│   │   └── error_analysis.md
│   └── reference_database/                 # Optional: Custom instrument samples
│       ├── 808_kick_samples/
│       ├── moog_lead_samples/
│       └── embeddings.pkl
├── output_midi/                            # Enhanced MIDI outputs
│   └── instrument_recognition_*.mid
└── yourmt3_space/                          # Existing YourMT3 setup
```

---

## 🔍 Key Decisions & Rationale

| Decision | Rationale |
|----------|-----------|
| **YAMNet over custom model** | Pre-trained, fast, good coverage, no training needed |
| **Level 2 granularity (25 cats)** | Balance between detail and accuracy, sufficient for most use cases |
| **Isolation-first approach** | Clean samples → better predictions, filter out polyphonic noise |
| **User creates mapping** | Domain knowledge required, YAMNet's 521 classes need expert curation |
| **Phase 0 simplification** | Focus on instrument ID, reuse proven transcription from existing PoC |
| **Jupyter notebook format** | Interactive experimentation, easy visualization, quick iteration |
| **No time estimates** | Research phase, uncertainty high, focus on quality over speed |

---

## 🚀 Getting Started

**Prerequisites**:
- Working YourMT3 setup (from `Stems_PoC_Test_Fixed.ipynb`)
- GPU A10 with CUDA
- Python 3.8+
- ~10GB GPU memory available

**Installation** (Phase 1):
```bash
pip install tensorflow tensorflow_hub librosa crepe
```

**First Steps**:
1. Create `yamnet_to_level2_mapping.json` (refer to Section 2)
2. Run Phase 0 of `Instrument_Recognition_PoC.ipynb`
3. Test transcription on sample audio
4. Proceed to Phase 1 implementation

---

## 📚 References

- **YAMNet**: [TensorFlow Hub](https://tfhub.dev/google/yamnet/1)
- **AudioSet**: [Dataset](https://research.google.com/audioset/)
- **YourMT3**: [GitHub](https://github.com/mimbres/YourMT3)
- **CREPE**: [GitHub](https://github.com/marl/crepe)
- **pretty_midi**: [Documentation](https://craffel.github.io/pretty-midi/)

---

## 📝 Notes

- This specification is a living document, update as design evolves
- User will create `yamnet_to_level2_mapping.json` in Phase 1
- Focus on modern genres first (electro, techno, rap), expand later
- Extensibility is key: design for easy addition of custom instruments
- Manual evaluation critical for Phase 5 (accuracy validation)

---

*Document created: October 3, 2025*
*Last updated: October 3, 2025*
*Version: 1.0*
