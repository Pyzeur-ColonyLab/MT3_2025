#!/usr/bin/env python3
"""
Debug script to identify broadcasting error in MT3 attention
"""
import torch
import numpy as np
from preprocessing import AudioPreprocessor, AudioPreprocessingConfig
from models import MT3Model, MT3Config

print("=" * 60)
print("Debugging MT3 Broadcasting Error")
print("=" * 60)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}\n")

# 1. Create minimal test audio
print("1. Creating test audio (1 second)...")
test_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000))

# 2. Preprocess
print("2. Preprocessing...")
config = AudioPreprocessingConfig(device=device)
preprocessor = AudioPreprocessor(config)
features = preprocessor.compute_features(test_audio)
encoder_input = preprocessor.prepare_encoder_input(features)
encoder_input = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in encoder_input.items()}

print(f"   Encoder input shape: {encoder_input['encoder_input'].shape}")

# 3. Load model (without checkpoint to isolate the issue)
print("\n3. Loading model...")
model_config = MT3Config()
print(f"   decoder_start_token_id: {model_config.decoder_start_token_id}")
print(f"   pad_token_id: {model_config.pad_token_id}")
print(f"   eos_token_id: {model_config.eos_token_id}")

model = MT3Model(model_config).to(device)
model.eval()

# 4. Test encoder
print("\n4. Testing encoder...")
try:
    with torch.no_grad():
        encoder_outputs = model.encoder(
            inputs_embeds=encoder_input['encoder_input'],
            attention_mask=encoder_input.get('attention_mask'),
            return_dict=True
        )
    print(f"   ✅ Encoder output: {encoder_outputs['last_hidden_state'].shape}")
except Exception as e:
    print(f"   ❌ Encoder failed: {e}")
    exit(1)

# 5. Test decoder initialization
print("\n5. Testing decoder initialization...")
batch_size = 1
decoder_input_ids = torch.full(
    (batch_size, 1),
    model_config.decoder_start_token_id,
    dtype=torch.long,
    device=device
)
print(f"   decoder_input_ids shape: {decoder_input_ids.shape}")
print(f"   decoder_input_ids value: {decoder_input_ids}")

# 6. Test first decoder forward (this is where the error likely occurs)
print("\n6. Testing first decoder forward pass...")
try:
    with torch.no_grad():
        outputs = model.forward(
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            attention_mask=encoder_input.get('attention_mask'),
            past_key_values=None,
            use_cache=True,
            return_dict=True
        )
    print(f"   ✅ First forward passed")
    print(f"   Logits shape: {outputs['logits'].shape}")
    print(f"   past_key_values length: {len(outputs['past_key_values']) if outputs['past_key_values'] else 0}")

except RuntimeError as e:
    print(f"   ❌ First forward failed: {e}")
    print("\n   This is where the broadcasting error occurs!")
    print("   Likely issue: position_bias shape mismatch in decoder self-attention")
    exit(1)

# 7. Test second decoder forward with past_key_values
print("\n7. Testing second decoder forward with past_key_values...")
next_token = torch.argmax(outputs['logits'][:, -1, :], dim=-1, keepdim=True)
decoder_input_ids_2 = next_token

print(f"   decoder_input_ids_2 shape: {decoder_input_ids_2.shape}")

try:
    with torch.no_grad():
        outputs_2 = model.forward(
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids_2,
            attention_mask=encoder_input.get('attention_mask'),
            past_key_values=outputs['past_key_values'],
            use_cache=True,
            return_dict=True
        )
    print(f"   ✅ Second forward passed")
    print(f"   Logits shape: {outputs_2['logits'].shape}")

except RuntimeError as e:
    print(f"   ❌ Second forward failed: {e}")
    exit(1)

print("\n" + "=" * 60)
print("✅ All forward passes successful - no broadcasting error!")
print("=" * 60)
