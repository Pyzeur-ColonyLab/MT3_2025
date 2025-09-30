# MT3 Token Decoder

Decoder module for converting MT3 model output tokens into MIDI files.

## Overview

This module implements the complete token → MIDI pipeline:

```
Model Tokens → Events (Codec) → NoteSequence → MIDI File
```

Based on the original MT3 implementation from Google Magenta.

## Quick Start

### Basic Usage

```python
from decoder.decoder import MT3TokenDecoder

# Create decoder
decoder = MT3TokenDecoder(num_velocity_bins=1)

# Decode tokens to MIDI
decoder.tokens_to_midi(
    tokens=generated_tokens,  # numpy array from model
    output_path="output.mid"
)
```

### Get NoteSequence First

```python
# Get NoteSequence object for inspection
note_sequence = decoder.tokens_to_note_sequence(
    tokens=generated_tokens,
    start_time=0.0
)

# Inspect notes
for note in note_sequence.notes:
    print(f"Pitch: {note.pitch}, Start: {note.start_time}, "
          f"End: {note.end_time}, Velocity: {note.velocity}")

# Save to MIDI later
import note_seq
note_seq.sequence_proto_to_midi_file(note_sequence, "output.mid")
```

### Batch Decoding (Long Audio)

```python
# For audio split into chunks
decoder.batch_tokens_to_midi(
    token_sequences=[tokens1, tokens2, tokens3],
    frame_times=[times1, times2, times3],
    output_path="combined.mid"
)
```

## Configuration

### Velocity Resolution

**Simple velocity (default)**:
```python
decoder = MT3TokenDecoder(num_velocity_bins=1)
# All notes have same velocity (100)
```

**Full velocity range**:
```python
decoder = MT3TokenDecoder(num_velocity_bins=127)
# 127 velocity levels (1-127)
```

### Time Resolution

**Standard (default)**:
```python
decoder = MT3TokenDecoder(steps_per_second=100)
# 10ms time resolution
```

**High resolution**:
```python
decoder = MT3TokenDecoder(steps_per_second=200)
# 5ms time resolution (more precise timing)
```

## Vocabulary System

MT3 uses a dynamic vocabulary based on event types:

### Event Types

| Event Type | Range | Description |
|------------|-------|-------------|
| `shift` | 0-1000 | Time shifts (0-10 seconds at 100 steps/sec) |
| `pitch` | 21-108 | MIDI pitches (88 piano keys) |
| `velocity` | 0-127 | Note velocity (or 0-1 for simplified) |
| `program` | 0-127 | MIDI program (instrument) |
| `drum` | 21-108 | Drum pitch |
| `tie` | 0 | Note continuation marker |

### Vocabulary Size

```python
decoder = MT3TokenDecoder()
print(decoder.get_vocab_size())  # 1536 (for default config)
```

**Breakdown** (default config with num_velocity_bins=1):
- Special tokens: 3 (PAD=0, EOS=1, UNK=2)
- Shift events: 1001 (0-1000 steps)
- Pitch events: 88 (21-108)
- Velocity events: 2 (0-1)
- Tie events: 1
- Program events: 128 (0-127)
- Drum events: 88 (21-108)

**Total**: 3 + 1001 + 88 + 2 + 1 + 128 + 88 = **1311** regular tokens
With extra_ids: 1311 + 225 = **1536** total

## Codec System

### Understanding Events

```python
# Decode single token to event
event = decoder.decode_single_token(token_id=150)
print(f"Type: {event.type}, Value: {event.value}")
# Output: Type: shift, Value: 147
```

### Event Type Ranges

```python
# Get token ID range for event type
pitch_range = decoder.get_event_type_range('pitch')
print(f"Pitch tokens: {pitch_range[0]} to {pitch_range[1]}")
# Output: Pitch tokens: 1001 to 1088
```

### Codec Information

```python
info = decoder.get_codec_info()
print(info)
# {
#     'steps_per_second': 100,
#     'max_shift_steps': 1000,
#     'num_classes': 1311,
#     'num_velocity_bins': 1
# }
```

## Token Sequence Format

### Example Token Sequence

```python
tokens = [
    150,    # shift 147 steps
    1050,   # pitch 60 (middle C)
    200,    # shift 197 steps
    1001,   # velocity 0 (note-off)
    1050,   # pitch 60 (end middle C)
]
```

### Decoding Process

1. **Parse tokens**: Convert token IDs to events using codec
2. **Track state**: Maintain current velocity, program, active notes
3. **Build timeline**: Accumulate time shifts, create note on/off events
4. **Handle ties**: Manage notes continuing across segment boundaries
5. **Create NoteSequence**: Assemble final MIDI representation

## Advanced Usage

### Debug Token Sequence

```python
def debug_tokens(decoder, tokens):
    """Print human-readable token sequence."""
    for i, token_id in enumerate(tokens):
        try:
            event = decoder.decode_single_token(token_id)
            print(f"Token {i:3d}: {token_id:4d} → {event.type:8s} {event.value:3d}")
        except:
            print(f"Token {i:3d}: {token_id:4d} → INVALID")

debug_tokens(decoder, tokens[:20])
```

### Custom Codec Configuration

```python
decoder = MT3TokenDecoder(
    steps_per_second=200,      # 5ms resolution
    max_shift_seconds=5,       # Max 5 second shifts
    num_velocity_bins=32,      # 32 velocity levels
)
```

### Extract Specific Instruments

```python
# Decode to NoteSequence first
ns = decoder.tokens_to_note_sequence(tokens)

# Extract piano track (program 0)
from decoder import note_sequences
piano_track = note_sequences.extract_track(
    ns, program=0, is_drum=False
)

# Save piano only
import note_seq
note_seq.sequence_proto_to_midi_file(piano_track, "piano_only.mid")
```

## Module Components

### Core Modules

- **`decoder.py`**: Main decoder class (`MT3TokenDecoder`)
- **`event_codec.py`**: Event encoding/decoding (`Codec`, `Event`, `EventRange`)
- **`vocabularies.py`**: Vocabulary configuration and token mapping
- **`note_sequences.py`**: NoteSequence manipulation and encoding specs
- **`run_length_encoding.py`**: Event sequence encoding/decoding
- **`metrics_utils.py`**: Prediction combination and evaluation utilities

### Encoding Specs

Three encoding specifications supported:

1. **`NoteOnsetEncodingSpec`**: Only note onsets, no durations
2. **`NoteEncodingSpec`**: Onsets and offsets (note durations)
3. **`NoteEncodingWithTiesSpec`**: Onsets, offsets, and ties (used by default)

Ties allow notes to continue across segment boundaries when processing long audio.

## Dependencies

```
note-seq>=0.0.5     # MIDI and NoteSequence utilities
pretty_midi>=0.2.0  # MIDI file I/O
numpy>=1.20.0       # Array operations
```

## Integration with MT3 Pipeline

### Complete Pipeline

```python
from preprocessing import AudioPreprocessor
from models import MT3Model
from decoder.decoder import MT3TokenDecoder

# 1. Preprocess audio
preprocessor = AudioPreprocessor()
features = preprocessor.process_file("audio.wav")
encoder_input = preprocessor.prepare_encoder_input(features)

# 2. Generate tokens
model = MT3Model.from_checkpoint("mt3_converted.pth")
tokens = model.generate(input_ids=encoder_input)

# 3. Decode to MIDI
decoder = MT3TokenDecoder()
decoder.tokens_to_midi(tokens, "output.mid")
```

### Or Use Inference Handler

```python
from inference import MT3Inference

# One-line transcription
inference = MT3Inference("mt3_converted.pth")
result = inference.transcribe("audio.wav", "output.mid")
```

## Troubleshooting

### Invalid Token Errors

```python
# Check if tokens are in valid range
vocab_size = decoder.get_vocab_size()
assert tokens.max() < vocab_size, f"Token {tokens.max()} >= vocab size {vocab_size}"
```

### Empty MIDI Output

```python
# Check if tokens contain actual note events
ns = decoder.tokens_to_note_sequence(tokens)
print(f"Number of notes: {len(ns.notes)}")

# If 0 notes, inspect token sequence
if len(ns.notes) == 0:
    print("Token sequence:")
    for token in tokens[:50]:  # First 50 tokens
        event = decoder.decode_single_token(token)
        print(f"  {event.type}: {event.value}")
```

### Timing Issues

```python
# Check time resolution
info = decoder.get_codec_info()
time_resolution = 1.0 / info['steps_per_second']
print(f"Time resolution: {time_resolution*1000:.1f}ms")

# Increase if timing is too coarse
decoder = MT3TokenDecoder(steps_per_second=200)  # 5ms resolution
```

## References

- [MT3 Paper](https://arxiv.org/abs/2111.03017): Multi-Task Multitrack Music Transcription
- [MT3 GitHub](https://github.com/magenta/mt3): Original TensorFlow implementation
- [note-seq](https://github.com/magenta/note-seq): MIDI and music utilities

## License

Based on MT3 implementation from Google Magenta (Apache 2.0 License)