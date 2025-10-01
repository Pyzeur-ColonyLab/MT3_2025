#!/usr/bin/env python3
"""
Test MT3 transcription with real music: Howard Shore - The Shire
"""
import torch
from inference import MT3Inference

print("=" * 60)
print("MT3 Real Music Transcription Test")
print("Howard Shore - The Shire")
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

# 2. Verify configuration
print("\n2. Configuration Verification")
model_info = inference.get_model_info()
print(f"   Device: {model_info['device']}")
print(f"   Preprocessor n_mels: {model_info['preprocessor']['n_mels']}")
print(f"   Model parameters: {model_info['model']['parameters']['total']:,}")
assert model_info['preprocessor']['n_mels'] == 512, "❌ Expected 512 mel bins"
print("   ✅ Configuration verified (512 mel bins)")

# 3. Transcribe real music
audio_path = "02.HowardShore-TheShire.flac"
output_path = "TheShire_transcribed.mid"

print(f"\n3. Transcribing: {audio_path}")
print("   This may take a few minutes for a full music file...")

try:
    result = inference.transcribe(
        audio_path=audio_path,
        output_path=output_path,
        max_length=2048,  # Longer for real music
        do_sample=False   # Greedy decoding for deterministic output
    )

    print(f"\n   ✅ Transcription completed successfully!")
    print(f"   Output: {result['midi_path']}")
    print(f"   Notes detected: {result['num_notes']}")
    print(f"   Audio duration: {result['duration']:.2f}s")

    if result['num_notes'] > 0:
        print(f"   Average note density: {result['num_notes'] / result['duration']:.1f} notes/sec")

except RuntimeError as e:
    print(f"\n   ❌ Transcription failed with RuntimeError: {e}")
    if "size of tensor" in str(e):
        print("   → Dimension mismatch error")
        print("   → Check that checkpoint has correct continuous_inputs_projection")
    exit(1)
except FileNotFoundError as e:
    print(f"\n   ❌ Audio file not found: {e}")
    print(f"   → Make sure {audio_path} exists in current directory")
    exit(1)
except Exception as e:
    print(f"\n   ❌ Transcription failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 60)
print("✅ REAL MUSIC TRANSCRIPTION TEST PASSED!")
print("=" * 60)
print("\nMT3 pipeline is fully operational.")
print(f"Check the generated MIDI file: {output_path}")
