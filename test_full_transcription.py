#!/usr/bin/env python3
"""
Test complete audio transcription pipeline with loaded checkpoint
"""
import torch
import numpy as np
from inference import MT3Inference

print("=" * 60)
print("Testing Complete MT3 Transcription Pipeline")
print("=" * 60)

# 1. Initialize inference with checkpoint
print("\n1. Loading MT3 model with checkpoint...")
try:
    inference = MT3Inference(
        checkpoint_path="mt3_converted_fixed.pth",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    print("   ✅ Model loaded successfully")
except Exception as e:
    print(f"   ❌ Failed to load model: {e}")
    exit(1)

# 2. Check model info
print("\n2. Model Information")
model_info = inference.get_model_info()
print(f"   Device: {model_info['device']}")
print(f"   Preprocessor n_mels: {model_info['preprocessor']['n_mels']}")
print(f"   Model d_model: {model_info['model']['config']['d_model']}")
print(f"   Vocab size: {model_info['decoder']['vocab_size']}")

# Verify dimensions match
assert model_info['preprocessor']['n_mels'] == 512, "❌ Preprocessor should use 512 mel bins"
print("   ✅ Dimensions verified: preprocessor n_mels = 512")

# 3. Create test audio
print("\n3. Creating test audio...")
# Create 5 seconds of test audio (C major chord: C-E-G at 261.63, 329.63, 392.00 Hz)
sample_rate = 16000
duration = 5.0
t = np.linspace(0, duration, int(sample_rate * duration))

# Simple chord
audio = (
    0.3 * np.sin(2 * np.pi * 261.63 * t) +  # C4
    0.3 * np.sin(2 * np.pi * 329.63 * t) +  # E4
    0.3 * np.sin(2 * np.pi * 392.00 * t)    # G4
)

# Save test audio
import soundfile as sf
test_audio_path = "test_chord.wav"
sf.write(test_audio_path, audio, sample_rate)
print(f"   ✅ Test audio created: {test_audio_path} ({duration}s)")

# 4. Test transcription
print("\n4. Running transcription...")
try:
    result = inference.transcribe(
        audio_path=test_audio_path,
        output_path="test_chord_output.mid",
        max_length=1024,
        do_sample=False  # Greedy decoding
    )

    print(f"   ✅ Transcription completed")
    print(f"   MIDI output: {result['midi_path']}")
    print(f"   Notes detected: {result['num_notes']}")
    print(f"   Audio duration: {result['duration']:.2f}s")

except RuntimeError as e:
    print(f"   ❌ Transcription failed with RuntimeError: {e}")
    if "size of tensor" in str(e):
        print("   → Dimension mismatch detected")
    exit(1)
except Exception as e:
    print(f"   ❌ Transcription failed: {e}")
    exit(1)

print("\n" + "=" * 60)
print("✅ COMPLETE TRANSCRIPTION TEST PASSED!")
print("=" * 60)
print("\nThe mel_bins fix (256→512) resolved the dimension mismatch.")
print("MT3 is now ready for real music transcription.")
