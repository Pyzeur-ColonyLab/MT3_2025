#!/usr/bin/env python3
"""Debug position_bias propagation and is_decoder flag"""
import torch
import numpy as np
from preprocessing import AudioPreprocessor, AudioPreprocessingConfig
from models import MT3Model, MT3Config

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("=" * 60)
print("Position Bias Propagation Debug")
print("=" * 60)
print(f"Device: {device}\n")

# Create minimal test
print("1. Creating test setup...")
config = AudioPreprocessingConfig(device=device)
preprocessor = AudioPreprocessor(config)
test_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000))
features = preprocessor.compute_features(test_audio)
encoder_input = preprocessor.prepare_encoder_input(features)
encoder_input = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in encoder_input.items()}

model = MT3Model(MT3Config()).to(device)
model.eval()

# Encode
print("\n2. Encoding audio...")
with torch.no_grad():
    encoder_outputs = model.encoder(
        inputs_embeds=encoder_input['encoder_input'],
        return_dict=True
    )
print(f"   Encoder output: {encoder_outputs['last_hidden_state'].shape}")

# Check decoder first layer configuration
print("\n3. Checking decoder layer configuration...")
first_layer_attn = model.decoder.block[0].layer[0].SelfAttention
print(f"   First decoder layer (self-attention):")
print(f"     has_relative_attention_bias: {first_layer_attn.has_relative_attention_bias}")
print(f"     is_decoder: {first_layer_attn.is_decoder}")
print(f"     relative_attention_num_buckets: {first_layer_attn.relative_attention_num_buckets}")
print(f"     n_heads: {first_layer_attn.n_heads}")

# First decoder forward
print("\n4. First decoder forward (no cache)...")
decoder_input_ids = torch.tensor([[0]], dtype=torch.long, device=device)
print(f"   decoder_input_ids: {decoder_input_ids.shape}")

with torch.no_grad():
    outputs1 = model.forward(
        encoder_outputs=encoder_outputs,
        decoder_input_ids=decoder_input_ids,
        past_key_values=None,
        use_cache=True,
        return_dict=True
    )

print(f"   ✅ First forward passed")
print(f"   Logits: {outputs1['logits'].shape}")
print(f"   past_key_values: {len(outputs1['past_key_values'])} layers")
if outputs1['past_key_values']:
    print(f"   past_key_values[0][0] (keys): {outputs1['past_key_values'][0][0].shape}")
    print(f"   past_key_values[0][1] (values): {outputs1['past_key_values'][0][1].shape}")

# Second decoder forward with cache
print("\n5. Second decoder forward (with cache)...")
next_token = torch.argmax(outputs1['logits'][:, -1, :], dim=-1, keepdim=True)
print(f"   decoder_input_ids: {next_token.shape}")
print(f"   Using cached keys/values from first forward")

try:
    with torch.no_grad():
        outputs2 = model.forward(
            encoder_outputs=encoder_outputs,
            decoder_input_ids=next_token,
            past_key_values=outputs1['past_key_values'],
            use_cache=True,
            return_dict=True
        )
    print(f"   ✅ Second forward passed")
    print(f"   Logits: {outputs2['logits'].shape}")

    print("\n" + "=" * 60)
    print("✅ ALL FORWARDS PASSED - Broadcasting issue resolved!")
    print("=" * 60)

except RuntimeError as e:
    print(f"   ❌ Second forward failed: {e}")

    # Additional diagnostics
    print("\n6. Additional diagnostics:")
    print(f"   Error indicates position_bias shape mismatch")
    print(f"   Expected: position_bias should be recalculated for each step")
    print(f"   Check: Is position_bias being reused from previous forward?")

    # Check if is_decoder is properly set
    print(f"\n7. Verifying is_decoder flag propagation:")
    for i, block in enumerate(model.decoder.block[:2]):
        attn = block.layer[0].SelfAttention
        print(f"   Decoder block {i}: is_decoder = {attn.is_decoder}")

    print("\n   ⚠️ If is_decoder is False, that's the problem!")
    print("   The attention layer thinks it's an encoder, not a decoder.")
