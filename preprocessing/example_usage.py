#!/usr/bin/env python3
"""
MT3 Audio Preprocessing - Example Usage

Demonstrates how to use the MT3 audio preprocessing pipeline
for music transcription tasks. Shows integration with MT3Model.

Examples:
- Single file processing
- Batch processing
- Integration with MT3 model
- Performance optimization
- Memory-efficient processing
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
import soundfile as sf

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from preprocessing.audio_preprocessing import (
    AudioPreprocessor,
    AudioPreprocessingConfig,
    audio_to_frames,
    compute_features
)

# Try to import MT3 model (if available)
try:
    from models.mt3_model import MT3Model, MT3Config
    MT3_AVAILABLE = True
except ImportError:
    print("⚠️  MT3Model not found. Some examples will be skipped.")
    MT3_AVAILABLE = False


def create_test_audio_files(output_dir: str = "./test_audio") -> list:
    """
    Create test audio files for demonstration.

    Args:
        output_dir: Directory to save test files

    Returns:
        List of created file paths
    """
    print("🎵 Creating test audio files...")

    os.makedirs(output_dir, exist_ok=True)
    test_files = []

    # Piano-like sounds (multiple frequencies)
    piano_freqs = [261.63, 329.63, 392.00]  # C4, E4, G4 (C major chord)
    piano_audio = sum(0.3 * np.sin(2 * np.pi * f * np.linspace(0, 3, 48000))
                     for f in piano_freqs)
    piano_file = os.path.join(output_dir, "piano_chord.wav")
    sf.write(piano_file, piano_audio, 16000)
    test_files.append(piano_file)

    # Guitar-like sound (fundamental + harmonics)
    fundamental = 196.00  # G3
    harmonics = [fundamental * i for i in [1, 2, 3, 4]]
    harmonic_weights = [1.0, 0.5, 0.25, 0.125]
    guitar_audio = sum(w * 0.2 * np.sin(2 * np.pi * f * np.linspace(0, 4, 64000))
                      for f, w in zip(harmonics, harmonic_weights))
    guitar_file = os.path.join(output_dir, "guitar_harmonics.wav")
    sf.write(guitar_file, guitar_audio, 16000)
    test_files.append(guitar_file)

    # Drum-like sound (noise burst with exponential decay)
    t = np.linspace(0, 1, 16000)
    drum_audio = np.random.normal(0, 0.3, 16000) * np.exp(-t * 5)
    drum_file = os.path.join(output_dir, "drum_hit.wav")
    sf.write(drum_file, drum_audio, 16000)
    test_files.append(drum_file)

    print(f"✅ Created {len(test_files)} test audio files in {output_dir}")
    return test_files


def example_basic_usage():
    """Basic audio preprocessing example."""
    print("\n" + "="*60)
    print("📝 EXAMPLE 1: Basic Audio Preprocessing")
    print("="*60)

    # Create preprocessor with default configuration
    config = AudioPreprocessingConfig(
        sample_rate=16000,
        n_mels=256,
        hop_length=320,
        device='cpu'  # Use CPU for portability
    )

    preprocessor = AudioPreprocessor(config)
    print(f"✅ Created preprocessor with {config.n_mels} mel bins")

    # Create test audio files
    test_files = create_test_audio_files()

    # Process single file
    print(f"\n🎵 Processing single file: {test_files[0]}")
    start_time = time.time()

    features = preprocessor.process_file(test_files[0])

    processing_time = time.time() - start_time
    print(f"✅ Features computed in {processing_time:.3f}s")
    print(f"   Shape: {features.shape}")
    print(f"   Data type: {features.dtype}")
    print(f"   Device: {features.device}")
    print(f"   Value range: [{features.min():.3f}, {features.max():.3f}]")

    # Prepare for model input
    encoder_input = preprocessor.prepare_encoder_input(features)
    print(f"\n🔧 Prepared encoder input:")
    print(f"   Encoder input shape: {encoder_input['encoder_input'].shape}")
    print(f"   Attention mask shape: {encoder_input['attention_mask'].shape}")
    print(f"   Sequence lengths: {encoder_input['seq_lengths']}")

    # Validate output
    try:
        preprocessor.validate_output(encoder_input)
        print("✅ Output validation passed")
    except Exception as e:
        print(f"❌ Output validation failed: {e}")


def example_batch_processing():
    """Batch processing example."""
    print("\n" + "="*60)
    print("📦 EXAMPLE 2: Batch Processing")
    print("="*60)

    # Create preprocessor
    config = AudioPreprocessingConfig(device='cpu')
    preprocessor = AudioPreprocessor(config)

    # Create test files
    test_files = create_test_audio_files()

    print(f"🎵 Processing {len(test_files)} files in batch...")
    start_time = time.time()

    # Process all files together
    batch_output = preprocessor.process_batch(test_files)

    processing_time = time.time() - start_time
    print(f"✅ Batch processed in {processing_time:.3f}s")
    print(f"   Batch size: {batch_output['encoder_input'].shape[0]}")
    print(f"   Max sequence length: {batch_output['encoder_input'].shape[1]}")
    print(f"   Feature dimensions: {batch_output['encoder_input'].shape[2]}")

    # Show individual sequence lengths
    seq_lengths = batch_output['seq_lengths']
    print(f"   Individual lengths: {seq_lengths.tolist()}")

    # Calculate padding efficiency
    total_elements = batch_output['encoder_input'].numel()
    valid_elements = batch_output['attention_mask'].sum().item() * config.n_mels
    efficiency = valid_elements / total_elements * 100
    print(f"   Padding efficiency: {efficiency:.1f}%")


def example_chunked_processing():
    """Chunked processing for long audio files."""
    print("\n" + "="*60)
    print("✂️  EXAMPLE 3: Chunked Processing (Long Audio)")
    print("="*60)

    # Create longer test audio (30 seconds)
    print("🎵 Creating 30-second test audio...")
    long_audio = []
    for i in range(30):  # 30 seconds
        t = np.linspace(0, 1, 16000)  # 1 second at 16kHz
        freq = 440 * (1 + 0.1 * np.sin(2 * np.pi * 0.1 * i))  # Vibrato
        chunk = 0.3 * np.sin(2 * np.pi * freq * t)
        long_audio.extend(chunk)

    long_audio = np.array(long_audio)
    long_file = "./test_audio/long_audio.wav"
    os.makedirs("./test_audio", exist_ok=True)
    sf.write(long_file, long_audio, 16000)

    # Process with chunking
    config = AudioPreprocessingConfig(
        frame_size=2048,  # ~2 seconds per frame at 16kHz
        overlap_ratio=0.1,  # 10% overlap
        device='cpu'
    )
    preprocessor = AudioPreprocessor(config)

    print(f"📊 Processing with frame size: {config.frame_size} samples")
    start_time = time.time()

    features_list, timestamps = preprocessor.audio_to_frames(long_file)

    processing_time = time.time() - start_time
    print(f"✅ Chunked processing completed in {processing_time:.3f}s")
    print(f"   Total frames: {len(features_list)}")
    print(f"   Frame timestamps: {[f'{t:.1f}s' for t in timestamps[:5]]}... (showing first 5)")

    # Show frame statistics
    frame_lengths = [f.shape[0] for f in features_list]
    print(f"   Frame lengths: min={min(frame_lengths)}, max={max(frame_lengths)}, avg={np.mean(frame_lengths):.1f}")

    # Batch process frames
    batch_input = preprocessor.prepare_encoder_input(features_list, max_seq_len=200)
    print(f"   Batched frames shape: {batch_input['encoder_input'].shape}")


def example_integration_with_mt3():
    """Integration with MT3 model (if available)."""
    if not MT3_AVAILABLE:
        print("\n⚠️  Skipping MT3 integration example (model not available)")
        return

    print("\n" + "="*60)
    print("🤖 EXAMPLE 4: Integration with MT3 Model")
    print("="*60)

    # Create MT3 model
    mt3_config = MT3Config(
        d_model=512,
        num_encoder_layers=8,
        num_decoder_layers=8,
        vocab_size=1536
    )
    model = MT3Model(mt3_config)
    model.eval()

    print(f"✅ Created MT3 model with {sum(p.numel() for p in model.parameters())} parameters")

    # Create preprocessor with matching configuration
    preprocess_config = AudioPreprocessingConfig(
        device='cpu',
        n_mels=256  # Should match model's expected input
    )
    preprocessor = AudioPreprocessor(preprocess_config)

    # Create test audio
    test_files = create_test_audio_files()

    # Full pipeline: Audio → Features → Model
    print(f"\n🔄 Running full pipeline on {test_files[0]}...")

    # 1. Preprocess audio
    features = preprocessor.process_file(test_files[0])
    encoder_input = preprocessor.prepare_encoder_input(features)

    print(f"   Preprocessing: {features.shape} → {encoder_input['encoder_input'].shape}")

    # 2. Run through model encoder
    with torch.no_grad():
        encoder_outputs = model.encoder(
            input_ids=None,  # Not used for audio input
            attention_mask=encoder_input['attention_mask'],
            inputs_embeds=encoder_input['encoder_input']
        )

    print(f"   Encoder output: {encoder_outputs.last_hidden_state.shape}")

    # 3. Create dummy decoder input for demonstration
    batch_size = encoder_input['encoder_input'].shape[0]
    decoder_input = torch.zeros(batch_size, 1, dtype=torch.long)  # Start token

    # 4. Run decoder step
    with torch.no_grad():
        decoder_outputs = model(
            encoder_outputs=encoder_outputs.last_hidden_state,
            encoder_attention_mask=encoder_input['attention_mask'],
            decoder_input_ids=decoder_input
        )

    print(f"   Decoder output: {decoder_outputs.logits.shape}")
    print(f"   Vocabulary size: {decoder_outputs.logits.shape[-1]}")

    # Show top predictions
    logits = decoder_outputs.logits[0, 0]  # First batch, first token
    top_tokens = torch.topk(logits, 5).indices
    print(f"   Top 5 predicted tokens: {top_tokens.tolist()}")

    print("✅ Full MT3 pipeline completed successfully!")


def example_performance_optimization():
    """Performance optimization examples."""
    print("\n" + "="*60)
    print("⚡ EXAMPLE 5: Performance Optimization")
    print("="*60)

    # Test different configurations for performance
    configs = [
        ("Standard", AudioPreprocessingConfig(n_mels=256, hop_length=320)),
        ("Fast", AudioPreprocessingConfig(n_mels=128, hop_length=512)),
        ("High Quality", AudioPreprocessingConfig(n_mels=512, hop_length=160)),
    ]

    # Create test audio
    test_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 5, 80000))  # 5 seconds
    test_file = "./test_audio/performance_test.wav"
    os.makedirs("./test_audio", exist_ok=True)
    sf.write(test_file, test_audio, 16000)

    print("📊 Comparing preprocessing configurations:")

    results = []
    for name, config in configs:
        config.device = 'cpu'  # Ensure consistent device
        preprocessor = AudioPreprocessor(config)

        # Time the preprocessing
        start_time = time.time()
        features = preprocessor.process_file(test_file)
        processing_time = time.time() - start_time

        # Calculate memory usage (approximate)
        memory_usage = features.numel() * 4  # 4 bytes per float32

        results.append({
            'name': name,
            'time': processing_time,
            'shape': features.shape,
            'memory_mb': memory_usage / 1024 / 1024,
            'n_mels': config.n_mels,
            'hop_length': config.hop_length
        })

        print(f"   {name:12s}: {processing_time:.3f}s, {features.shape}, {memory_usage/1024/1024:.1f}MB")

    # Find best trade-offs
    fastest = min(results, key=lambda x: x['time'])
    most_detailed = max(results, key=lambda x: x['shape'][1])  # Most mel bins

    print(f"\n🏆 Fastest: {fastest['name']} ({fastest['time']:.3f}s)")
    print(f"🔍 Most detailed: {most_detailed['name']} ({most_detailed['n_mels']} mel bins)")

    # GPU vs CPU comparison (if CUDA available)
    if torch.cuda.is_available():
        print(f"\n🎯 GPU vs CPU comparison:")

        for device in ['cpu', 'cuda']:
            config = AudioPreprocessingConfig(device=device)
            preprocessor = AudioPreprocessor(config)

            start_time = time.time()
            features = preprocessor.process_file(test_file)
            processing_time = time.time() - start_time

            print(f"   {device.upper():4s}: {processing_time:.3f}s")

    print("✅ Performance analysis completed!")


def example_error_handling():
    """Error handling and validation examples."""
    print("\n" + "="*60)
    print("🛡️  EXAMPLE 6: Error Handling & Validation")
    print("="*60)

    config = AudioPreprocessingConfig(device='cpu')
    preprocessor = AudioPreprocessor(config)

    # Test various error conditions
    print("🧪 Testing error handling:")

    # 1. Missing file
    try:
        preprocessor.load_audio("nonexistent_file.wav")
        print("❌ Should have raised FileNotFoundError")
    except FileNotFoundError:
        print("✅ FileNotFoundError handled correctly")

    # 2. Unsupported format
    try:
        # Create dummy file with wrong extension
        dummy_file = "./test_audio/dummy.txt"
        os.makedirs("./test_audio", exist_ok=True)
        with open(dummy_file, 'w') as f:
            f.write("not audio")

        preprocessor.load_audio(dummy_file)
        print("❌ Should have raised ValueError for unsupported format")
    except ValueError as e:
        if "Unsupported audio format" in str(e):
            print("✅ Unsupported format error handled correctly")
        else:
            print(f"❌ Unexpected ValueError: {e}")
    finally:
        if os.path.exists(dummy_file):
            os.unlink(dummy_file)

    # 3. Invalid configuration
    try:
        invalid_config = AudioPreprocessingConfig(n_mels=-1)
        print("❌ Should have raised AssertionError")
    except AssertionError:
        print("✅ Invalid configuration handled correctly")

    # 4. Output validation
    print("\n🔍 Testing output validation:")

    # Valid output
    test_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000))
    features = preprocessor.compute_features(test_audio)
    encoder_input = preprocessor.prepare_encoder_input(features)

    try:
        preprocessor.validate_output(encoder_input)
        print("✅ Valid output passes validation")
    except Exception as e:
        print(f"❌ Validation failed unexpectedly: {e}")

    # Invalid output (missing keys)
    try:
        invalid_output = {'encoder_input': torch.randn(1, 10, 256)}
        preprocessor.validate_output(invalid_output)
        print("❌ Should have raised ValueError")
    except ValueError as e:
        if "Missing required keys" in str(e):
            print("✅ Missing keys error handled correctly")
        else:
            print(f"❌ Unexpected error: {e}")

    print("✅ Error handling validation completed!")


def cleanup_test_files():
    """Clean up test files."""
    test_dir = "./test_audio"
    if os.path.exists(test_dir):
        import shutil
        shutil.rmtree(test_dir)
        print(f"🧹 Cleaned up test directory: {test_dir}")


def main():
    """Run all examples."""
    print("🎵 MT3 Audio Preprocessing - Example Usage")
    print("=" * 60)
    print("This script demonstrates the MT3 audio preprocessing pipeline")
    print("with various use cases and integration examples.")

    try:
        # Run examples
        example_basic_usage()
        example_batch_processing()
        example_chunked_processing()
        example_integration_with_mt3()
        example_performance_optimization()
        example_error_handling()

        print("\n" + "="*60)
        print("🎉 All examples completed successfully!")
        print("=" * 60)
        print("\nNext steps:")
        print("- Integrate with your MT3 model training pipeline")
        print("- Optimize configuration for your specific use case")
        print("- Add custom audio augmentation if needed")
        print("- Consider GPU acceleration for large-scale processing")

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Clean up
        cleanup_test_files()


if __name__ == "__main__":
    main()