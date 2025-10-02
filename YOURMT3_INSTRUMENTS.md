# YourMT3 Supported Instruments

Complete list of instruments recognized by YourMT3 model (`mc13_full_plus_256` configuration).

---

## Summary

- **36 instrument classes** organized into 13 decoding channels
- Based on **General MIDI** standard (programs 0-101)
- **17 drum note types** (GM Drum Notes)
- Supports **multi-instrument polyphonic** transcription

---

## Melodic Instruments (34 Classes)

### Channel 0: Piano
| Class | GM Programs | Instruments |
|-------|-------------|-------------|
| Acoustic Piano | 0, 1, 3, 6, 7 | Acoustic Grand Piano, Bright Acoustic Piano, Honky-tonk Piano, Harpsichord, Clavinet |
| Electric Piano | 2, 4, 5 | Electric Grand Piano, Electric Piano 1, Electric Piano 2 |

### Channel 1: Chromatic Percussion
| Class | GM Programs | Instruments |
|-------|-------------|-------------|
| Chromatic Percussion | 8-15 | Celesta, Glockenspiel, Music Box, Vibraphone, Marimba, Xylophone, Tubular Bells, Dulcimer |

### Channel 2: Organ
| Class | GM Programs | Instruments |
|-------|-------------|-------------|
| Organ | 16-23 | Drawbar Organ, Percussive Organ, Rock Organ, Church Organ, Reed Organ, Accordion, Harmonica, Tango Accordion |

### Channel 3: Guitar
| Class | GM Programs | Instruments |
|-------|-------------|-------------|
| Acoustic Guitar | 24-25 | Acoustic Guitar (nylon), Acoustic Guitar (steel) |
| Clean Electric Guitar | 26-28 | Electric Guitar (jazz), Electric Guitar (clean), Electric Guitar (muted) |
| Distorted Electric Guitar | 29-31 | Overdriven Guitar, Distortion Guitar, Guitar Harmonics |

### Channel 4: Bass
| Class | GM Programs | Instruments |
|-------|-------------|-------------|
| Acoustic Bass | 32, 35 | Acoustic Bass, Fretless Bass |
| Electric Bass | 33, 34, 36-39 | Electric Bass (finger), Electric Bass (pick), Slap Bass 1, Slap Bass 2, Synth Bass 1, Synth Bass 2 |

### Channel 5: Strings
| Class | GM Programs | Instruments |
|-------|-------------|-------------|
| Violin | 40 | Violin |
| Viola | 41 | Viola |
| Cello | 42 | Cello |
| Contrabass | 43 | Contrabass |
| Orchestral Harp | 46 | Orchestral Harp |
| Timpani | 47 | Timpani |
| String Ensemble | 44, 45, 48, 49 | Tremolo Strings, Pizzicato Strings, String Ensemble 1, String Ensemble 2 |
| Synth Strings | 50, 51 | Synth Strings 1, Synth Strings 2 |
| Choir and Voice | 52-54 | Choir Aahs, Voice Oohs, Synth Choir |
| Orchestra Hit | 55 | Orchestra Hit |

### Channel 6: Brass
| Class | GM Programs | Instruments |
|-------|-------------|-------------|
| Trumpet | 56, 59 | Trumpet, Muted Trumpet |
| Trombone | 57 | Trombone |
| Tuba | 58 | Tuba |
| French Horn | 60 | French Horn |
| Brass Section | 61-63 | Brass Section, Synth Brass 1, Synth Brass 2 |

### Channel 7: Reed
| Class | GM Programs | Instruments |
|-------|-------------|-------------|
| Soprano/Alto Sax | 64, 65 | Soprano Sax, Alto Sax |
| Tenor Sax | 66 | Tenor Sax |
| Baritone Sax | 67 | Baritone Sax |
| Oboe | 68 | Oboe |
| English Horn | 69 | English Horn |
| Bassoon | 70 | Bassoon |
| Clarinet | 71 | Clarinet |

### Channel 8: Pipe
| Class | GM Programs | Instruments |
|-------|-------------|-------------|
| Pipe | 72-79 | Piccolo, Flute, Recorder, Pan Flute, Bottle Blow, Shakuhachi, Whistle, Ocarina |

### Channel 9: Synth Lead
| Class | GM Programs | Instruments |
|-------|-------------|-------------|
| Synth Lead | 80-87 | Lead 1 (square), Lead 2 (sawtooth), Lead 3 (calliope), Lead 4 (chiff), Lead 5 (charang), Lead 6 (voice), Lead 7 (fifths), Lead 8 (bass + lead) |

### Channel 10: Synth Pad
| Class | GM Programs | Instruments |
|-------|-------------|-------------|
| Synth Pad | 88-95 | Pad 1 (new age), Pad 2 (warm), Pad 3 (polysynth), Pad 4 (choir), Pad 5 (bowed), Pad 6 (metallic), Pad 7 (halo), Pad 8 (sweep) |

### Channel 11: Vocals
| Class | GM Programs | Instruments |
|-------|-------------|-------------|
| Singing Voice | 100 | Singing Voice (melody) |
| Singing Voice (chorus) | 101 | Singing Voice (chorus/backing vocals) |

---

## Percussion (Channel 12)

### Drum Notes (MIDI Channel 10, Program 128)
| Drum Type | MIDI Notes |
|-----------|------------|
| Kick Drum | 36, 35 |
| Snare X-stick | 37, 2 |
| Snare Drum | 38, 40 |
| Closed Hi-Hat | 42, 44, 22 |
| Open Hi-Hat | 46, 26 |
| Cowbell | 56 |
| High Floor Tom | 43 |
| Low Floor Tom | 41 |
| Low Tom | 45 |
| Low-Mid Tom | 47 |
| Mid Tom | 48 |
| Low Tom (Rim) | 50 |
| Mid Tom (Rim) | 58 |
| Ride | 51 |
| Ride (Bell) | 53 |
| Ride (Edge) | 59 |
| Chinese Cymbal | 52 |
| Crash Cymbal | 49, 57 |
| Splash Cymbal | 55 |

---

## NOT Supported

The following GM instruments (programs 96-127) are **excluded** from the vocabulary:

### Sound Effects & Rare Instruments
- **FX Sounds** (96-103): Rain, Soundtrack, Crystal, Atmosphere, Brightness, Goblins, Echoes, Sci-fi
- **Ethnic Instruments** (104-111): Sitar, Banjo, Shamisen, Koto, Kalimba, Bagpipe, Fiddle, Shanai
- **Percussive Effects** (112-119): Tinkle Bell, Agogo, Steel Drums, Woodblock, Taiko Drum, Melodic Tom, Synth Drum, Reverse Cymbal
- **Noise/Environmental** (120-127): Guitar Fret Noise, Breath Noise, Seashore, Bird Tweet, Telephone Ring, Helicopter, Applause, Gunshot

**Reason**: These instruments are rare in typical music datasets and would increase model complexity without significant benefit.

---

## Usage in MIDI Output

When YourMT3 generates a MIDI file, it assigns instruments using these program numbers. For example:

```python
# Load generated MIDI
import pretty_midi
midi = pretty_midi.PrettyMIDI('output.mid')

for instrument in midi.instruments:
    if instrument.is_drum:
        print(f"Drum track: {len(instrument.notes)} notes")
    else:
        print(f"Program {instrument.program}: {len(instrument.notes)} notes")
        # Program 0 = Acoustic Grand Piano
        # Program 25 = Acoustic Guitar (steel)
        # Program 48 = String Ensemble 1
        # etc.
```

---

## Quick Reference

**Most Common Instruments in Popular Music**:
- Piano (0-7)
- Guitar (24-31)
- Bass (32-39)
- Drums (channel 10)
- Strings (40-55)
- Brass (56-63)
- Synth (80-95)
- Vocals (100-101)

**Classical Orchestra**:
- Strings (40-47): Violin, Viola, Cello, Contrabass
- Woodwinds (64-79): Saxes, Oboe, Bassoon, Clarinet, Flute, Piccolo
- Brass (56-63): Trumpet, Trombone, Tuba, French Horn
- Percussion: Timpani (47), Drums (channel 10)

**Jazz Ensemble**:
- Piano (0-7)
- Guitar (24-31)
- Bass (32-39)
- Drums (channel 10)
- Saxes (64-67)
- Trumpet (56, 59)
- Trombone (57)

**Rock Band**:
- Electric Guitar (26-31)
- Electric Bass (33-39)
- Drums (channel 10)
- Vocals (100-101)
- Synth (80-95)

---

## Extension Possibilities

**Can I add new instruments?**

**Short answer**: Not without retraining the model.

**Why?**
- Instrument vocabulary is baked into the model architecture
- 13 decoding channels are fixed
- Token embeddings are trained for specific program numbers

**What you CAN do**:
- Fine-tune for better accuracy on existing instruments
- Post-process MIDI to change program numbers
- Use similar instruments as substitutes

**What requires retraining**:
- Adding instruments from GM programs 96-127
- Creating new instrument categories
- Changing the number of decoding channels

**See**: `YOURMT3_WORKFLOW.md` section "Extending the Model" for detailed instructions.

---

**Source**: `yourmt3_space/amt/src/config/vocabulary.py` (MT3_FULL_PLUS vocabulary)

**Model**: YourMT3+ (`mc13_full_plus_256`)

**Reference**: [General MIDI Level 1 Sound Set](https://www.midi.org/specifications-old/item/gm-level-1-sound-set)

Assisted by Claude Code
