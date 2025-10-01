#!/usr/bin/env python3
"""
Test script to verify mel_bins dimension fix (256 → 512)
"""
import torch
import numpy as np
from preprocessing import AudioPreprocessor, AudioPreprocessingConfig
from models import MT3Model, MT3Config

print("=" * 60)
print("Testing mel_bins dimension fix (256 → 512)")
print("=" * 60)

# 1. Test preprocessor configuration
print("\n1. Preprocessor Configuration")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"   Device: {device}")
config = AudioPreprocessingConfig(device=device)
print(f"   n_mels: {config.n_mels}")
assert config.n_mels == 512, f"❌ Expected 512, got {config.n_mels}"
print("   ✅ Preprocessor config: n_mels = 512")

# 2. Test preprocessor output
print("\n2. Preprocessor Output")
preprocessor = AudioPreprocessor(config)
test_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000))  # 1 second
features = preprocessor.compute_features(test_audio)
print(f"   Features shape: {features.shape}")
assert features.shape[1] == 512, f"❌ Expected 512 mel bins, got {features.shape[1]}"
print(f"   ✅ Features: [seq_len={features.shape[0]}, mel_bins={features.shape[1]}]")

# 3. Test encoder input preparation
print("\n3. Encoder Input Preparation")
encoder_input = preprocessor.prepare_encoder_input(features)
enc_shape = encoder_input['encoder_input'].shape
print(f"   Encoder input shape: {enc_shape}")
assert enc_shape[2] == 512, f"❌ Expected 512, got {enc_shape[2]}"
print(f"   ✅ Encoder input: [batch={enc_shape[0]}, seq={enc_shape[1]}, mel_bins={enc_shape[2]}]")

# 4. Test model continuous_inputs_projection
print("\n4. Model Projection Layer")
model_config = MT3Config()
model = MT3Model(model_config).to(device)
proj_weight = model.encoder.continuous_inputs_projection.weight
print(f"   continuous_inputs_projection weight: {proj_weight.shape}")
assert proj_weight.shape == torch.Size([512, 512]), f"❌ Expected [512, 512], got {proj_weight.shape}"
print(f"   ✅ Projection layer: {proj_weight.shape[1]} → {proj_weight.shape[0]}")

# 5. Test complete forward pass (without checkpoint)
print("\n5. Complete Forward Pass Test")
encoder_input_tensor = encoder_input['encoder_input']
print(f"   Input: {encoder_input_tensor.shape}")

try:
    with torch.no_grad():
        encoder_output = model.encoder(inputs_embeds=encoder_input_tensor)
    print(f"   Encoder output: {encoder_output['last_hidden_state'].shape}")
    print("   ✅ Forward pass successful (no dimension mismatch)")
except RuntimeError as e:
    print(f"   ❌ Forward pass failed: {e}")
    exit(1)

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED - mel_bins fix verified!")
print("=" * 60)
print("\nNext: Load checkpoint and test audio transcription")
