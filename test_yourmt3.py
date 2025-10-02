#!/usr/bin/env python3
"""
Test YourMT3 with The Shire audio
"""
import sys
import os

# Save original directory
original_dir = os.getcwd()
audio_path_full = os.path.join(original_dir, "02.HowardShore-TheShire.flac")

# Change to yourmt3_space directory (needed for checkpoint paths)
os.chdir('yourmt3_space')

# Add YourMT3 paths
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('amt/src'))

import torch
import torchaudio
from model_helper import load_model_checkpoint, transcribe

print("=" * 60)
print("YourMT3 Test: The Shire Transcription")
print("=" * 60)

# Configuration matching the demo
print("\n1. Loading YourMT3 model...")
model_name = "YPTF.MoE+Multi (noPS)"
checkpoint = "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops@last.ckpt"
project = '2024'
precision = '16'

args = [
    checkpoint,
    '-p', project,
    '-tk', 'mc13_full_plus_256',
    '-dec', 'multi-t5',
    '-nl', '26',
    '-enc', 'perceiver-tf',
    '-sqr', '1',
    '-ff', 'moe',
    '-wf', '4',
    '-nmoe', '8',
    '-kmoe', '2',
    '-act', 'silu',
    '-epe', 'rope',
    '-rp', '1',
    '-ac', 'spec',
    '-hop', '300',
    '-atc', '1',
    '-pr', precision
]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"   Device: {device}")
print(f"   Model: {model_name}")
print(f"   Checkpoint: {checkpoint}")

try:
    model = load_model_checkpoint(args=args, device=device)
    print("   ✅ Model loaded successfully")
except Exception as e:
    print(f"   ❌ Model loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Audio file (use full path from original directory)
audio_path = audio_path_full
print(f"\n2. Loading audio: {audio_path}")

if not os.path.exists(audio_path):
    print(f"   ❌ Audio file not found: {audio_path}")
    sys.exit(1)

# Get audio info
info = torchaudio.info(audio_path)
print(f"   Duration: {info.num_frames / info.sample_rate:.2f}s")
print(f"   Sample rate: {info.sample_rate} Hz")
print(f"   Channels: {info.num_channels}")

# Prepare audio info dict for transcribe function
audio_info = {
    "filepath": audio_path,
    "track_name": "TheShire_YourMT3",
    "sample_rate": int(info.sample_rate),
    "bits_per_sample": int(info.bits_per_sample) if info.bits_per_sample else 16,
    "num_channels": int(info.num_channels),
    "num_frames": int(info.num_frames),
    "duration": int(info.num_frames / info.sample_rate),
    "encoding": str(info.encoding).lower() if hasattr(info, 'encoding') else 'flac',
}

# Transcribe
print("\n3. Transcribing with YourMT3...")
print("   This may take several minutes for 149s audio...")

try:
    midifile = transcribe(model, audio_info)
    print(f"\n✅ Transcription complete!")
    print(f"   MIDI file: {midifile}")

    # Check file size
    if os.path.exists(midifile):
        size = os.path.getsize(midifile)
        print(f"   File size: {size:,} bytes")

        # Load and analyze MIDI
        import pretty_midi
        try:
            midi = pretty_midi.PrettyMIDI(midifile)
            total_notes = sum(len(inst.notes) for inst in midi.instruments)
            print(f"   Total notes: {total_notes}")
            print(f"   Instruments: {len(midi.instruments)}")

            if total_notes == 0:
                print("\n   ⚠️ WARNING: 0 notes detected!")
            else:
                print(f"\n   ✅ SUCCESS: {total_notes} notes detected!")

                # Show instrument breakdown
                for i, inst in enumerate(midi.instruments):
                    if len(inst.notes) > 0:
                        print(f"      Instrument {i}: {len(inst.notes)} notes, program={inst.program}")
        except Exception as e:
            print(f"   ⚠️ Could not analyze MIDI: {e}")
    else:
        print(f"   ❌ MIDI file not created")

except Exception as e:
    print(f"\n❌ Transcription failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("YourMT3 test complete")
print("=" * 60)
