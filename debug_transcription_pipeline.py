#!/usr/bin/env python3
"""
Debug MT3 transcription pipeline step by step
"""
import torch
import numpy as np
from inference import MT3Inference

print("=" * 60)
print("MT3 Transcription Pipeline Debug")
print("=" * 60)

# Initialize
print("\n1. Initializing MT3...")
inference = MT3Inference(
    checkpoint_path="mt3_converted_fixed.pth",
    device="cuda" if torch.cuda.is_available() else "cpu"
)
print("   ✅ Model loaded")

# Load and preprocess audio
audio_path = "02.HowardShore-TheShire.flac"
print(f"\n2. Loading audio: {audio_path}")

features = inference.preprocessor.process_file(audio_path)
print(f"   Features shape: {features.shape}")
print(f"   Features range: [{features.min():.3f}, {features.max():.3f}]")
print(f"   Duration estimate: {features.shape[0] * inference.preprocessor.config.hop_length / inference.preprocessor.config.sample_rate:.2f}s")

# Prepare encoder input
print("\n3. Preparing encoder input...")
encoder_input = inference.preprocessor.prepare_encoder_input(features)
encoder_input = {
    k: v.to(inference.device) if isinstance(v, torch.Tensor) else v
    for k, v in encoder_input.items()
}
print(f"   Encoder input shape: {encoder_input['encoder_input'].shape}")
print(f"   Attention mask shape: {encoder_input['attention_mask'].shape}")
print(f"   Sequence length: {encoder_input['seq_lengths'].item()}")

# Generate tokens
print("\n4. Generating tokens...")
print("   This may take a few minutes for 149s of audio...")

with torch.no_grad():
    generated_tokens = inference.model.generate(
        inputs_embeds=encoder_input['encoder_input'],
        attention_mask=encoder_input.get('attention_mask'),
        max_length=2048,  # Longer for real music
        do_sample=False,
        temperature=1.0,
        top_k=None,
        top_p=None,
    )

tokens = generated_tokens.cpu().numpy()[0]
print(f"   ✅ Generated {len(tokens)} tokens")
print(f"   Token range: [{tokens.min()}, {tokens.max()}]")
print(f"   Unique tokens: {len(np.unique(tokens))}")
print(f"   First 20 tokens: {tokens[:20]}")
print(f"   Last 20 tokens: {tokens[-20:]}")

# Count special tokens
pad_token = inference.model.config.pad_token_id
eos_token = inference.model.config.eos_token_id
num_pad = np.sum(tokens == pad_token)
num_eos = np.sum(tokens == eos_token)
print(f"\n   Special tokens:")
print(f"     PAD ({pad_token}): {num_pad} ({num_pad/len(tokens)*100:.1f}%)")
print(f"     EOS ({eos_token}): {num_eos} ({num_eos/len(tokens)*100:.1f}%)")

# Decode to MIDI
print("\n5. Decoding to MIDI...")
output_path = "TheShire_debug.mid"
note_sequence = inference.decoder.tokens_to_midi(
    tokens=tokens,
    output_path=output_path,
)

print(f"   ✅ MIDI saved: {output_path}")
print(f"   Notes in sequence: {len(note_sequence.notes)}")

if len(note_sequence.notes) > 0:
    print(f"   First note: pitch={note_sequence.notes[0].pitch}, start={note_sequence.notes[0].start_time:.2f}s")
    print(f"   Last note: pitch={note_sequence.notes[-1].pitch}, end={note_sequence.notes[-1].end_time:.2f}s")
    print(f"   Total duration: {note_sequence.total_time:.2f}s")
else:
    print("   ⚠️ No notes detected!")
    print("\n   Possible causes:")
    print("     1. Model generates only PAD/EOS tokens")
    print("     2. Decoder doesn't recognize token patterns")
    print("     3. Checkpoint weights not properly loaded")

print("\n" + "=" * 60)
if len(note_sequence.notes) > 0:
    print("✅ Pipeline completed with notes detected")
else:
    print("⚠️ Pipeline completed but NO NOTES detected")
print("=" * 60)
