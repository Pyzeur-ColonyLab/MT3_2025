#!/usr/bin/env python3
"""
MT3 Model Usage Examples

This script demonstrates how to use the MT3 PyTorch implementation
for various tasks including model creation, forward passes, and generation.
"""

import torch
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our MT3 implementation
from mt3_model import MT3Model, MT3Config, create_mt3_model
from checkpoint_utils import (
    load_mt3_checkpoint,
    create_model_from_checkpoint,
    diagnose_checkpoint_compatibility
)


def example_1_basic_model_creation():
    """Example 1: Basic model creation and inspection."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Model Creation")
    print("=" * 60)

    # Create model with default configuration
    print("Creating MT3 model with default configuration...")
    model = create_mt3_model()

    # Get model information
    param_summary = model.get_parameter_summary()
    print(f"\nModel created successfully!")
    print(f"Total parameters: {param_summary['total']:,}")
    print(f"Configuration:")
    print(f"  vocab_size: {model.config.vocab_size}")
    print(f"  d_model: {model.config.d_model}")
    print(f"  encoder_layers: {model.config.num_encoder_layers}")
    print(f"  decoder_layers: {model.config.num_decoder_layers}")
    print(f"  num_heads: {model.config.num_heads}")
    print(f"  d_ff: {model.config.d_ff}")

    return model


def example_2_forward_pass(model):
    """Example 2: Forward pass with dummy data."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Forward Pass")
    print("=" * 60)

    # Set model to evaluation mode
    model.eval()

    # Create dummy inputs
    batch_size = 2
    encoder_seq_len = 256  # Typical audio feature sequence length
    decoder_seq_len = 32   # Short generation sequence

    print(f"Creating dummy inputs...")
    print(f"  Batch size: {batch_size}")
    print(f"  Encoder sequence length: {encoder_seq_len}")
    print(f"  Decoder sequence length: {decoder_seq_len}")

    # Generate random token IDs
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, encoder_seq_len))
    decoder_input_ids = torch.randint(0, model.config.vocab_size, (batch_size, decoder_seq_len))

    print("\nRunning forward pass...")
    with torch.no_grad():  # Disable gradients for inference
        outputs = model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            return_dict=True
        )

    # Inspect outputs
    logits_shape = outputs['logits'].shape
    expected_shape = (batch_size, decoder_seq_len, model.config.vocab_size)

    print(f"Forward pass completed successfully!")
    print(f"  Logits shape: {logits_shape}")
    print(f"  Expected shape: {expected_shape}")
    print(f"  Shape correct: {logits_shape == expected_shape}")

    # Check output ranges
    logits = outputs['logits']
    print(f"  Logits range: [{logits.min():.3f}, {logits.max():.3f}]")
    print(f"  Logits mean: {logits.mean():.3f}")

    return outputs


def example_3_generation(model):
    """Example 3: Autoregressive generation."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Autoregressive Generation")
    print("=" * 60)

    # Set model to evaluation mode
    model.eval()

    # Create encoder input (e.g., audio features)
    batch_size = 1
    encoder_seq_len = 128
    max_gen_length = 50

    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, encoder_seq_len))

    print(f"Input sequence length: {encoder_seq_len}")
    print(f"Maximum generation length: {max_gen_length}")

    # Example 3a: Greedy decoding
    print("\n3a. Greedy decoding:")
    with torch.no_grad():
        generated_greedy = model.generate(
            input_ids=input_ids,
            max_length=max_gen_length,
            do_sample=False,  # Greedy
            early_stopping=True
        )

    print(f"  Generated shape: {generated_greedy.shape}")
    print(f"  Generated tokens: {generated_greedy[0, :10].tolist()}...")  # First 10 tokens

    # Example 3b: Temperature sampling
    print("\n3b. Temperature sampling:")
    with torch.no_grad():
        generated_temp = model.generate(
            input_ids=input_ids,
            max_length=max_gen_length,
            do_sample=True,
            temperature=0.8,
            early_stopping=True
        )

    print(f"  Generated shape: {generated_temp.shape}")
    print(f"  Generated tokens: {generated_temp[0, :10].tolist()}...")

    # Example 3c: Top-k sampling
    print("\n3c. Top-k sampling:")
    with torch.no_grad():
        generated_topk = model.generate(
            input_ids=input_ids,
            max_length=max_gen_length,
            do_sample=True,
            top_k=50,
            temperature=1.0,
            early_stopping=True
        )

    print(f"  Generated shape: {generated_topk.shape}")
    print(f"  Generated tokens: {generated_topk[0, :10].tolist()}...")

    return generated_greedy, generated_temp, generated_topk


def example_4_checkpoint_analysis(checkpoint_path=None):
    """Example 4: Checkpoint analysis and loading."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Checkpoint Analysis")
    print("=" * 60)

    if checkpoint_path is None or not Path(checkpoint_path).exists():
        print("No checkpoint provided or checkpoint not found.")
        print("Skipping checkpoint analysis.")
        print("\nTo test checkpoint loading, run:")
        print("  python example_usage.py --checkpoint path/to/your/checkpoint.pth")
        return None

    print(f"Analyzing checkpoint: {checkpoint_path}")

    # Analyze checkpoint compatibility
    try:
        analysis = diagnose_checkpoint_compatibility(checkpoint_path)

        print(f"\nCheckpoint Analysis:")
        print(f"  File size: {analysis['file_size_mb']:.1f} MB")
        print(f"  Total parameters: {analysis['total_parameters']:,}")
        print(f"  Parameter tensors: {analysis['parameter_count']}")

        if analysis['inferred_config']['vocab_size']:
            print(f"  Inferred vocab_size: {analysis['inferred_config']['vocab_size']}")
        if analysis['inferred_config']['d_model']:
            print(f"  Inferred d_model: {analysis['inferred_config']['d_model']}")

        print(f"  Compatible: {analysis['is_compatible']}")

        if analysis['compatibility_issues']:
            print("  Issues:")
            for issue in analysis['compatibility_issues']:
                print(f"    - {issue}")

    except Exception as e:
        print(f"Failed to analyze checkpoint: {e}")
        return None

    # Try loading checkpoint
    print(f"\nAttempting to load checkpoint...")
    try:
        model_with_checkpoint = create_model_from_checkpoint(checkpoint_path)
        print("✅ Checkpoint loaded successfully!")

        # Test forward pass with loaded model
        print("Testing forward pass with loaded model...")
        model_with_checkpoint.eval()

        with torch.no_grad():
            dummy_input = torch.randint(0, model_with_checkpoint.config.vocab_size, (1, 64))
            outputs = model_with_checkpoint.generate(
                input_ids=dummy_input,
                max_length=20,
                do_sample=False
            )

        print(f"✅ Generation test successful! Output shape: {outputs.shape}")
        return model_with_checkpoint

    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        return None


def example_5_custom_configuration():
    """Example 5: Custom model configuration."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Custom Configuration")
    print("=" * 60)

    # Create custom configuration
    custom_config = MT3Config(
        vocab_size=2048,      # Larger vocabulary
        d_model=768,          # Larger model dimension
        num_encoder_layers=6, # Fewer encoder layers
        num_decoder_layers=6, # Fewer decoder layers
        num_heads=12,         # More attention heads
        d_ff=2048,            # Larger feed-forward
        dropout_rate=0.15,    # Higher dropout
        max_length=512        # Shorter max generation
    )

    print("Custom configuration:")
    print(f"  vocab_size: {custom_config.vocab_size}")
    print(f"  d_model: {custom_config.d_model}")
    print(f"  layers: {custom_config.num_encoder_layers} encoder, {custom_config.num_decoder_layers} decoder")
    print(f"  attention heads: {custom_config.num_heads}")
    print(f"  feed-forward size: {custom_config.d_ff}")
    print(f"  dropout: {custom_config.dropout_rate}")

    # Create model with custom config
    print("\nCreating model with custom configuration...")
    custom_model = MT3Model(custom_config)

    param_summary = custom_model.get_parameter_summary()
    print(f"Custom model parameters: {param_summary['total']:,}")

    # Compare with default model
    default_model = create_mt3_model()
    default_params = default_model.get_parameter_count()

    print(f"Parameter comparison:")
    print(f"  Default model: {default_params:,}")
    print(f"  Custom model:  {param_summary['total']:,}")
    print(f"  Difference:    {param_summary['total'] - default_params:+,}")

    return custom_model


def example_6_memory_and_performance():
    """Example 6: Memory usage and performance analysis."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Memory and Performance Analysis")
    print("=" * 60)

    model = create_mt3_model()
    model.eval()

    # Memory analysis
    print("Memory analysis:")

    # Model size
    param_count = model.get_parameter_count()
    model_size_mb = param_count * 4 / (1024 * 1024)  # Assuming float32
    print(f"  Model parameters: {param_count:,}")
    print(f"  Model size: {model_size_mb:.1f} MB (float32)")

    # Test different batch sizes
    batch_sizes = [1, 2, 4, 8]
    seq_len = 256

    print("\nPerformance test (forward pass timing):")
    print(f"{'Batch Size':<12} {'Time (ms)':<12} {'Tokens/sec':<12}")
    print("-" * 36)

    import time

    for batch_size in batch_sizes:
        input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))
        decoder_input_ids = torch.randint(0, model.config.vocab_size, (batch_size, 32))

        # Warmup
        with torch.no_grad():
            _ = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)

        # Timing
        start_time = time.time()
        with torch.no_grad():
            _ = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        end_time = time.time()

        elapsed_ms = (end_time - start_time) * 1000
        tokens_per_sec = (batch_size * seq_len) / (end_time - start_time)

        print(f"{batch_size:<12} {elapsed_ms:<12.1f} {tokens_per_sec:<12.0f}")

    # Generation timing
    print(f"\nGeneration timing (batch_size=1, max_length=50):")
    input_ids = torch.randint(0, model.config.vocab_size, (1, seq_len))

    start_time = time.time()
    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            max_length=50,
            do_sample=False
        )
    end_time = time.time()

    gen_time = end_time - start_time
    tokens_generated = generated.shape[1]
    tokens_per_sec = tokens_generated / gen_time

    print(f"  Generation time: {gen_time:.3f}s")
    print(f"  Tokens generated: {tokens_generated}")
    print(f"  Generation speed: {tokens_per_sec:.1f} tokens/sec")


def main():
    """Run all examples."""
    print("🎵 MT3 PyTorch Implementation - Usage Examples")
    print("=" * 60)
    print("This script demonstrates the MT3 model capabilities.")
    print("All examples use dummy data for demonstration purposes.")
    print()

    # Parse command line arguments
    import sys
    checkpoint_path = None
    if len(sys.argv) > 1 and sys.argv[1] == '--checkpoint':
        checkpoint_path = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        # Example 1: Basic model creation
        model = example_1_basic_model_creation()

        # Example 2: Forward pass
        outputs = example_2_forward_pass(model)

        # Example 3: Generation
        generated_sequences = example_3_generation(model)

        # Example 4: Checkpoint analysis (if provided)
        checkpoint_model = example_4_checkpoint_analysis(checkpoint_path)

        # Example 5: Custom configuration
        custom_model = example_5_custom_configuration()

        # Example 6: Performance analysis
        example_6_memory_and_performance()

        print("\n" + "=" * 60)
        print("🎉 All examples completed successfully!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Install required dependencies: pip install -r requirements.txt")
        print("2. Test with real audio features and vocabulary")
        print("3. Integrate with MT3 preprocessing and postprocessing")
        print("4. Fine-tune on your specific dataset")

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("Make sure PyTorch is installed: pip install torch")
        sys.exit(1)


if __name__ == "__main__":
    main()