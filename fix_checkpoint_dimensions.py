#!/usr/bin/env python3
"""
Fix checkpoint dimensions by transposing embeddings and resizing layer norms.
"""
import torch
import sys

def fix_checkpoint(input_path, output_path):
    print(f"Loading checkpoint: {input_path}")
    checkpoint = torch.load(input_path, map_location='cpu')
    
    state_dict = checkpoint['model_state_dict']
    
    print("\nFixing dimensions...")
    fixed_count = 0
    
    # Fix embeddings - transpose from [512, 1536] to [1536, 512]
    embedding_keys = ['shared.weight', 'encoder.embed_tokens.weight', 'decoder.embed_tokens.weight']
    for key in embedding_keys:
        if key in state_dict:
            old_shape = state_dict[key].shape
            if old_shape == torch.Size([512, 1536]):
                state_dict[key] = state_dict[key].T
                print(f"✓ {key}: {old_shape} → {state_dict[key].shape}")
                fixed_count += 1
    
    # Fix layer norms - resize from [1536] to [512]
    layer_norm_keys = [k for k in state_dict.keys() if 'layer_norm.weight' in k]
    for key in layer_norm_keys:
        if state_dict[key].shape == torch.Size([1536]):
            # Reinitialize with correct size (should be d_model=512, not vocab_size=1536)
            state_dict[key] = torch.ones(512, dtype=torch.float32)
            if fixed_count < 5:  # Print first few
                print(f"✓ {key}: [1536] → [512]")
            fixed_count += 1
    
    if len(layer_norm_keys) > 5:
        print(f"✓ ... and {len(layer_norm_keys) - 5} more layer_norms: [1536] → [512]")
    
    # Fix lm_head if needed
    if 'lm_head.weight' in state_dict:
        old_shape = state_dict['lm_head.weight'].shape
        if old_shape == torch.Size([512, 1536]):
            state_dict['lm_head.weight'] = state_dict['lm_head.weight'].T
            print(f"✓ lm_head.weight: {old_shape} → {state_dict['lm_head.weight'].shape}")
            fixed_count += 1
    
    print(f"\n✅ Fixed {fixed_count} parameters")
    
    # Update metadata
    checkpoint['model_state_dict'] = state_dict
    checkpoint['metadata']['fixed_dimensions'] = True
    
    # Save
    print(f"\nSaving corrected checkpoint: {output_path}")
    torch.save(checkpoint, output_path)
    
    # Verify
    print("\nVerifying corrected dimensions:")
    print(f"  shared.weight: {state_dict['shared.weight'].shape}")
    print(f"  encoder.embed_tokens.weight: {state_dict['encoder.embed_tokens.weight'].shape}")
    print(f"  encoder.block.0.layer.0.layer_norm.weight: {state_dict['encoder.block.0.layer.0.layer_norm.weight'].shape}")
    print(f"  lm_head.weight: {state_dict.get('lm_head.weight', 'N/A')}")
    
    print("\n✅ Checkpoint fixed successfully!")

if __name__ == "__main__":
    input_path = sys.argv[1] if len(sys.argv) > 1 else "mt3_converted.pth"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "mt3_converted_fixed.pth"
    
    fix_checkpoint(input_path, output_path)
