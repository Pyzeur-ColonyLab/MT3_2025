#!/usr/bin/env python3
"""
Verify MT3 checkpoint is properly loaded and trained for music transcription
"""
import torch
import numpy as np

print("=" * 60)
print("MT3 Checkpoint Validity Check")
print("=" * 60)

# Load checkpoint
checkpoint_path = "mt3_converted_fixed.pth"
print(f"\n1. Loading checkpoint: {checkpoint_path}")

checkpoint = torch.load(checkpoint_path, map_location='cpu')

print(f"   Checkpoint keys: {list(checkpoint.keys())}")
state_dict = checkpoint['model_state_dict']
print(f"   Parameters in checkpoint: {len(state_dict)}")

# Check critical weights
print("\n2. Checking critical weight shapes:")
critical_weights = [
    'shared.weight',
    'encoder.continuous_inputs_projection.weight',
    'lm_head.weight',
    'encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight',
    'decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight',
]

for key in critical_weights:
    if key in state_dict:
        print(f"   ✅ {key}: {state_dict[key].shape}")
    else:
        print(f"   ❌ {key}: MISSING")

# Check lm_head statistics
print("\n3. Analyzing lm_head weights:")
if 'lm_head.weight' in state_dict:
    lm_head = state_dict['lm_head.weight']
    print(f"   Shape: {lm_head.shape} (expected: [vocab_size, d_model])")
    print(f"   Mean: {lm_head.mean().item():.6f}")
    print(f"   Std: {lm_head.std().item():.6f}")
    print(f"   Min: {lm_head.min().item():.6f}")
    print(f"   Max: {lm_head.max().item():.6f}")

    # Check if weights look initialized (not random)
    # Well-trained weights typically have small mean and std
    if abs(lm_head.mean().item()) < 0.1 and 0.1 < lm_head.std().item() < 1.0:
        print("   ✅ Weights look trained (small mean, reasonable std)")
    else:
        print("   ⚠️ Weights might be untrained or incorrectly scaled")

# Check continuous_inputs_projection
print("\n4. Analyzing continuous_inputs_projection:")
if 'encoder.continuous_inputs_projection.weight' in state_dict:
    proj = state_dict['encoder.continuous_inputs_projection.weight']
    print(f"   Shape: {proj.shape} (expected: [512, 512] for d_model)")
    print(f"   Mean: {proj.mean().item():.6f}")
    print(f"   Std: {proj.std().item():.6f}")

    # This is the critical layer for audio input
    # If this is not trained, model won't understand audio
    if abs(proj.mean().item()) < 0.1 and 0.05 < proj.std().item() < 0.5:
        print("   ✅ Projection weights look trained")
    else:
        print("   ⚠️ Projection weights might be untrained")

# Check metadata
print("\n5. Checkpoint metadata:")
if 'metadata' in checkpoint:
    metadata = checkpoint['metadata']
    print(f"   Converted from: {metadata.get('converted_from', 'unknown')}")
    print(f"   Conversion date: {metadata.get('conversion_date', 'unknown')}")
    print(f"   Source: {metadata.get('source', 'unknown')}")

    if 'mt3' in str(metadata.get('source', '')).lower():
        print("   ✅ Checkpoint claims to be MT3")
    else:
        print("   ⚠️ Checkpoint might not be MT3-specific")

# Test forward pass with random input
print("\n6. Testing model forward pass:")
from models import MT3Model, MT3Config

model = MT3Model(MT3Config())
missing, unexpected = model.load_state_dict(state_dict, strict=False)

print(f"   Missing keys: {len(missing)}")
if len(missing) > 0:
    print(f"     {missing[:5]}...")

print(f"   Unexpected keys: {len(unexpected)}")
if len(unexpected) > 0:
    print(f"     {unexpected[:5]}...")

# Generate with random input
print("\n7. Testing generation with random audio features:")
model.eval()
with torch.no_grad():
    # Random audio features: [batch=1, seq=100, mel_bins=512]
    fake_audio = torch.randn(1, 100, 512)

    # Generate 50 tokens
    output = model.generate(
        inputs_embeds=fake_audio,
        max_length=50,
        do_sample=False
    )

    tokens = output[0].cpu().numpy()
    unique = np.unique(tokens)

    print(f"   Generated tokens: {len(tokens)}")
    print(f"   Unique tokens: {len(unique)}")
    print(f"   Token distribution: {unique[:10]}")

    if len(unique) == 1:
        print("   ❌ Model generates ONLY ONE TOKEN - checkpoint is NOT trained for MT3!")
        print(f"      Token {unique[0]} is repeated {len(tokens)} times")
        print("\n   CONCLUSION: This checkpoint is likely a base T5 model,")
        print("               NOT a fine-tuned MT3 model for music transcription.")
    elif len(unique) < 5:
        print(f"   ⚠️ Model generates only {len(unique)} unique tokens - severely undertrained")
    else:
        print(f"   ✅ Model generates {len(unique)} unique tokens - might be trained")

print("\n" + "=" * 60)
print("Checkpoint validation complete")
print("=" * 60)
