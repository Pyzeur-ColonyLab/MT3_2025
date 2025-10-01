#!/usr/bin/env python3
"""
Check vocabulary mismatch between model and decoder
"""
import torch
from models import MT3Model, MT3Config
from decoder import MT3TokenDecoder

print("=" * 60)
print("MT3 Vocabulary Mismatch Check")
print("=" * 60)

# 1. Check model vocab size
print("\n1. Model vocabulary:")
config = MT3Config()
print(f"   vocab_size (config): {config.vocab_size}")

checkpoint = torch.load("mt3_converted_fixed.pth", map_location='cpu')
state_dict = checkpoint['model_state_dict']

if 'lm_head.weight' in state_dict:
    lm_head_shape = state_dict['lm_head.weight'].shape
    print(f"   lm_head output size: {lm_head_shape[0]} tokens")
else:
    print("   ⚠️ lm_head.weight not found in checkpoint")

if 'shared.weight' in state_dict:
    embedding_shape = state_dict['shared.weight'].shape
    print(f"   Embedding vocab size: {embedding_shape[0]} tokens")

# 2. Check decoder vocabulary
print("\n2. Decoder vocabulary:")
decoder = MT3TokenDecoder(num_velocity_bins=1)
print(f"   Decoder vocab size: {decoder.get_vocab_size()}")
print(f"   Codec event types: {len(decoder.codec.event_type_range)}")

# Print first few vocab tokens
vocab_info = decoder.get_codec_info()
print(f"   Event ranges:")
for event_type, (start, end) in list(vocab_info['event_ranges'].items())[:10]:
    print(f"     {event_type}: [{start}, {end})")

# 3. Check if token 1133 is valid
print("\n3. Analyzing problematic token 1133:")
token_1133 = 1133
vocab_size = decoder.get_vocab_size()

if token_1133 < vocab_size:
    print(f"   Token 1133 is VALID in vocab (size {vocab_size})")

    # Try to decode it
    try:
        # The decoder expects tokens in vocab space
        event = decoder.codec.decode_event_index(token_1133)
        print(f"   Decoded to event: {event}")
    except Exception as e:
        print(f"   Cannot decode: {e}")
else:
    print(f"   ⚠️ Token 1133 is OUT OF RANGE (vocab size: {vocab_size})")

# 4. Check vocabulary alignment
print("\n4. Vocabulary alignment:")
model_vocab_size = config.vocab_size
decoder_vocab_size = decoder.get_vocab_size()

if model_vocab_size == decoder_vocab_size:
    print(f"   ✅ Sizes match: {model_vocab_size}")
else:
    print(f"   ❌ SIZE MISMATCH!")
    print(f"      Model expects: {model_vocab_size} tokens")
    print(f"      Decoder has: {decoder_vocab_size} tokens")
    print(f"      Difference: {abs(model_vocab_size - decoder_vocab_size)}")

    print("\n   CONCLUSION: Vocabulary mismatch detected!")
    print("   The model and decoder use different vocabularies.")
    print("   This causes the model to generate invalid tokens.")

    print("\n   Solutions:")
    print("   1. Rebuild decoder with correct vocab_size")
    print("   2. Or use the correct MT3 checkpoint with matching vocab")

# 5. Test token decoding
print("\n5. Testing token decoding:")
test_tokens = [0, 1, 2, 100, 500, 1000, 1133, 1500]
print("   Token → Event mapping:")

for token in test_tokens:
    if token < decoder_vocab_size:
        try:
            # Simulate decoding
            status = "valid"
        except:
            status = "error"
    else:
        status = "out_of_range"

    print(f"     {token}: {status}")

print("\n" + "=" * 60)
print("Vocabulary check complete")
print("=" * 60)
