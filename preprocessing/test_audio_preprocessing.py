#!/usr/bin/env python3
"""
Unit Tests for MT3 Audio Preprocessing Pipeline

Comprehensive test suite for audio preprocessing functionality.
Tests all major components with synthetic and real audio data.
"""

import os
import tempfile
from pathlib import Path
import numpy as np
import torch
import pytest
import soundfile as sf

from audio_preprocessing import (
    AudioPreprocessor,
    AudioPreprocessingConfig,
    audio_to_frames,
    compute_features,
    prepare_encoder_input
)


class TestAudioPreprocessingConfig:
    """Test configuration validation and initialization."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AudioPreprocessingConfig()

        assert config.sample_rate == 16000
        assert config.n_mels == 256
        assert config.hop_length == 320
        assert config.win_length == 512
        assert config.n_fft == 1024
        assert config.fmin == 0.0
        assert config.fmax == 8000.0
        assert config.frame_size == 512
        assert config.normalize is True

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = AudioPreprocessingConfig(n_mels=128, hop_length=160)
        assert config.n_mels == 128
        assert config.hop_length == 160

        # Invalid configs should raise assertions
        with pytest.raises(AssertionError):
            AudioPreprocessingConfig(n_mels=-1)

        with pytest.raises(AssertionError):
            AudioPreprocessingConfig(hop_length=0)

        with pytest.raises(AssertionError):
            AudioPreprocessingConfig(overlap_ratio=1.1)

    def test_device_auto_detection(self):
        """Test automatic device detection."""
        config = AudioPreprocessingConfig(device='cpu')
        # Should either stay cpu or switch to cuda if available
        assert config.device in ['cpu', 'cuda']


class TestAudioPreprocessor:
    """Test main AudioPreprocessor class."""

    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor with CPU device for testing."""
        config = AudioPreprocessingConfig(device='cpu')
        return AudioPreprocessor(config)

    @pytest.fixture
    def test_audio_file(self):
        """Create temporary test audio file."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            # Generate 2 seconds of 440Hz sine wave at 16kHz
            sample_rate = 16000
            duration = 2.0
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio = 0.5 * np.sin(2 * np.pi * 440 * t)

            sf.write(f.name, audio, sample_rate)
            yield f.name

        # Cleanup
        os.unlink(f.name)

    def test_initialization(self, preprocessor):
        """Test preprocessor initialization."""
        assert preprocessor.config.sample_rate == 16000
        assert preprocessor.config.n_mels == 256
        assert preprocessor._mel_filters is not None
        assert preprocessor._mel_filters.shape == (256, 513)  # n_mels x (n_fft//2 + 1)

    def test_supported_formats(self, preprocessor):
        """Test supported audio format detection."""
        expected_formats = {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg'}
        assert preprocessor.supported_formats == expected_formats

    def test_load_audio_success(self, preprocessor, test_audio_file):
        """Test successful audio loading."""
        audio, sr = preprocessor.load_audio(test_audio_file)

        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert sr == 16000
        assert len(audio) == 32000  # 2 seconds at 16kHz
        assert -1.0 <= audio.min() <= audio.max() <= 1.0

    def test_load_audio_file_not_found(self, preprocessor):
        """Test error handling for missing files."""
        with pytest.raises(FileNotFoundError):
            preprocessor.load_audio("nonexistent_file.wav")

    def test_load_audio_unsupported_format(self, preprocessor):
        """Test error handling for unsupported formats."""
        with tempfile.NamedTemporaryFile(suffix='.txt') as f:
            with pytest.raises(ValueError, match="Unsupported audio format"):
                preprocessor.load_audio(f.name)

    def test_resample_audio(self, preprocessor):
        """Test audio resampling functionality."""
        # Test case: no resampling needed
        audio = np.random.randn(16000)
        resampled = preprocessor.resample_audio(audio, 16000)
        assert len(resampled) == len(audio)
        np.testing.assert_array_almost_equal(audio, resampled)

        # Test case: actual resampling (8kHz to 16kHz)
        audio_8k = np.random.randn(8000)
        resampled = preprocessor.resample_audio(audio_8k, 8000)
        assert len(resampled) == 16000  # Doubled length

    def test_compute_features_shape(self, preprocessor):
        """Test feature computation output shape."""
        # 2 seconds of audio at 16kHz
        audio = np.random.randn(32000)
        features = preprocessor.compute_features(audio)

        assert isinstance(features, torch.Tensor)
        assert features.dim() == 2
        assert features.shape[1] == preprocessor.config.n_mels  # 256 mel bins

        # Time dimension should be approximately audio_length / hop_length
        expected_time_steps = (len(audio) - 1) // preprocessor.config.hop_length + 1
        assert abs(features.shape[0] - expected_time_steps) <= 1

    def test_compute_features_values(self, preprocessor):
        """Test feature computation produces reasonable values."""
        # Generate pure sine wave
        sample_rate = 16000
        freq = 440  # Hz
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * freq * t)

        features = preprocessor.compute_features(audio)

        # Features should be log-mel values
        assert torch.isfinite(features).all()

        # With normalization, values should be roughly centered around 0
        if preprocessor.config.normalize:
            mean_val = features.mean().item()
            assert -2.0 < mean_val < 2.0

    def test_audio_to_frames(self, preprocessor, test_audio_file):
        """Test audio frame chunking."""
        features_list, timestamps = preprocessor.audio_to_frames(test_audio_file)

        assert len(features_list) == len(timestamps)
        assert len(features_list) > 0

        # All features should have same mel dimension
        for features in features_list:
            assert isinstance(features, torch.Tensor)
            assert features.shape[1] == preprocessor.config.n_mels

        # Timestamps should be monotonically increasing
        assert all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1))
        assert timestamps[0] == 0.0

    def test_prepare_encoder_input_single(self, preprocessor):
        """Test encoder input preparation with single tensor."""
        # Create test features
        features = torch.randn(100, preprocessor.config.n_mels)

        encoder_input = preprocessor.prepare_encoder_input(features)

        assert 'encoder_input' in encoder_input
        assert 'attention_mask' in encoder_input
        assert 'seq_lengths' in encoder_input

        enc_input = encoder_input['encoder_input']
        assert enc_input.shape == (1, 100, preprocessor.config.n_mels)

        attention_mask = encoder_input['attention_mask']
        assert attention_mask.shape == (1, 100)
        assert attention_mask.all()  # All True for single tensor

        seq_lengths = encoder_input['seq_lengths']
        assert seq_lengths.shape == (1,)
        assert seq_lengths[0] == 100

    def test_prepare_encoder_input_batch(self, preprocessor):
        """Test encoder input preparation with batch."""
        # Create test features with different lengths
        features_list = [
            torch.randn(50, preprocessor.config.n_mels),
            torch.randn(75, preprocessor.config.n_mels),
            torch.randn(60, preprocessor.config.n_mels)
        ]

        encoder_input = preprocessor.prepare_encoder_input(features_list)

        enc_input = encoder_input['encoder_input']
        assert enc_input.shape == (3, 75, preprocessor.config.n_mels)  # Max length is 75

        attention_mask = encoder_input['attention_mask']
        assert attention_mask.shape == (3, 75)

        # Check attention masks
        assert attention_mask[0, :50].all() and not attention_mask[0, 50:].any()  # First: 50 valid
        assert attention_mask[1].all()  # Second: all valid (longest)
        assert attention_mask[2, :60].all() and not attention_mask[2, 60:].any()  # Third: 60 valid

        seq_lengths = encoder_input['seq_lengths']
        assert seq_lengths.tolist() == [50, 75, 60]

    def test_process_file(self, preprocessor, test_audio_file):
        """Test complete file processing pipeline."""
        features = preprocessor.process_file(test_audio_file)

        assert isinstance(features, torch.Tensor)
        assert features.dim() == 2
        assert features.shape[1] == preprocessor.config.n_mels

    def test_process_file_with_frames(self, preprocessor, test_audio_file):
        """Test file processing with frame output."""
        features_list, timestamps = preprocessor.process_file(test_audio_file, return_frames=True)

        assert isinstance(features_list, list)
        assert isinstance(timestamps, list)
        assert len(features_list) == len(timestamps)

    def test_process_batch(self, preprocessor):
        """Test batch processing with multiple files."""
        # Create multiple temporary audio files
        temp_files = []
        try:
            for i in range(3):
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    # Different frequencies for each file
                    freq = 220 * (i + 1)  # 220, 440, 660 Hz
                    audio = 0.5 * np.sin(2 * np.pi * freq * np.linspace(0, 1, 16000))
                    sf.write(f.name, audio, 16000)
                    temp_files.append(f.name)

            # Process batch
            batch_output = preprocessor.process_batch(temp_files)

            assert 'encoder_input' in batch_output
            assert 'attention_mask' in batch_output
            assert 'seq_lengths' in batch_output

            enc_input = batch_output['encoder_input']
            assert enc_input.shape[0] == 3  # Batch size
            assert enc_input.shape[2] == preprocessor.config.n_mels

        finally:
            # Cleanup
            for f in temp_files:
                if os.path.exists(f):
                    os.unlink(f)

    def test_validate_output_success(self, preprocessor):
        """Test output validation with valid input."""
        features = torch.randn(50, preprocessor.config.n_mels)
        encoder_input = preprocessor.prepare_encoder_input(features)

        # Should not raise any exception
        assert preprocessor.validate_output(encoder_input) is True

    def test_validate_output_failure(self, preprocessor):
        """Test output validation with invalid input."""
        # Missing keys
        with pytest.raises(ValueError, match="Missing required keys"):
            preprocessor.validate_output({'encoder_input': torch.randn(1, 10, 256)})

        # Wrong dimensions
        invalid_input = {
            'encoder_input': torch.randn(10, 256),  # 2D instead of 3D
            'attention_mask': torch.zeros(1, 10, dtype=torch.bool),
            'seq_lengths': torch.tensor([10])
        }
        with pytest.raises(ValueError, match="should be 3D"):
            preprocessor.validate_output(invalid_input)


class TestStandaloneFunctions:
    """Test standalone utility functions."""

    def test_audio_to_frames_function(self):
        """Test standalone audio_to_frames function."""
        # Create temporary test file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            audio = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000))
            sf.write(f.name, audio, 16000)

            try:
                features_list, timestamps = audio_to_frames(f.name)

                assert len(features_list) > 0
                assert len(timestamps) == len(features_list)
                assert all(isinstance(f, torch.Tensor) for f in features_list)

            finally:
                os.unlink(f.name)

    def test_compute_features_function(self):
        """Test standalone compute_features function."""
        audio = np.random.randn(16000)  # 1 second
        features = compute_features(audio)

        assert isinstance(features, torch.Tensor)
        assert features.dim() == 2
        assert features.shape[1] == 256  # Default n_mels

    def test_prepare_encoder_input_function(self):
        """Test standalone prepare_encoder_input function."""
        features = torch.randn(100, 256)
        encoder_input = prepare_encoder_input(features)

        assert 'encoder_input' in encoder_input
        assert 'attention_mask' in encoder_input
        assert 'seq_lengths' in encoder_input
        assert encoder_input['encoder_input'].shape == (1, 100, 256)


class TestIntegration:
    """Integration tests for complete preprocessing pipeline."""

    def test_end_to_end_processing(self):
        """Test complete end-to-end preprocessing pipeline."""
        # Create test audio file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            # 3 seconds of audio with multiple frequencies
            sample_rate = 16000
            duration = 3.0
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio = (0.3 * np.sin(2 * np.pi * 220 * t) +  # A3
                     0.3 * np.sin(2 * np.pi * 440 * t) +  # A4
                     0.3 * np.sin(2 * np.pi * 880 * t))   # A5

            sf.write(f.name, audio, sample_rate)

            try:
                # Full pipeline
                config = AudioPreprocessingConfig(device='cpu')
                preprocessor = AudioPreprocessor(config)

                # Process file
                features = preprocessor.process_file(f.name)

                # Prepare for encoder
                encoder_input = preprocessor.prepare_encoder_input(features)

                # Validate
                assert preprocessor.validate_output(encoder_input)

                # Check final dimensions
                enc_input = encoder_input['encoder_input']
                assert enc_input.dim() == 3
                assert enc_input.shape[2] == 256  # n_mels

                print(f"✅ End-to-end test passed:")
                print(f"   Input: {duration}s audio at {sample_rate}Hz")
                print(f"   Features shape: {features.shape}")
                print(f"   Encoder input shape: {enc_input.shape}")

            finally:
                os.unlink(f.name)

    def test_memory_efficiency(self):
        """Test memory efficiency with larger audio files."""
        # Create 30-second audio file (should still be processable)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            sample_rate = 16000
            duration = 30.0  # 30 seconds

            # Generate in chunks to avoid memory issues
            chunk_size = sample_rate * 5  # 5-second chunks
            audio_chunks = []

            for i in range(6):  # 6 chunks of 5 seconds each
                t_chunk = np.linspace(0, 5, chunk_size)
                freq = 440 * (1 + i * 0.1)  # Varying frequency
                chunk = 0.3 * np.sin(2 * np.pi * freq * t_chunk)
                audio_chunks.append(chunk)

            audio = np.concatenate(audio_chunks)
            sf.write(f.name, audio, sample_rate)

            try:
                config = AudioPreprocessingConfig(device='cpu', frame_size=1024)
                preprocessor = AudioPreprocessor(config)

                # Process with frames for memory efficiency
                features_list, timestamps = preprocessor.audio_to_frames(f.name)

                assert len(features_list) > 0
                assert len(timestamps) == len(features_list)

                # Process batch of frames
                encoder_input = preprocessor.prepare_encoder_input(features_list)
                assert preprocessor.validate_output(encoder_input)

                print(f"✅ Memory efficiency test passed:")
                print(f"   Processed {duration}s audio in {len(features_list)} frames")
                print(f"   Encoder input shape: {encoder_input['encoder_input'].shape}")

            finally:
                os.unlink(f.name)


if __name__ == "__main__":
    # Run basic functionality tests
    print("Running MT3 Audio Preprocessing Tests")
    print("=" * 50)

    # Test configuration
    print("\n1. Testing configuration...")
    config = AudioPreprocessingConfig()
    print(f"✅ Default config created: {config.sample_rate}Hz, {config.n_mels} mels")

    # Test preprocessor creation
    print("\n2. Testing preprocessor creation...")
    preprocessor = AudioPreprocessor(config)
    print(f"✅ Preprocessor created with device: {preprocessor.config.device}")

    # Test with synthetic audio
    print("\n3. Testing feature computation...")
    test_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 2, 32000))
    features = preprocessor.compute_features(test_audio)
    print(f"✅ Features computed: shape {features.shape}")

    # Test encoder input preparation
    print("\n4. Testing encoder input preparation...")
    encoder_input = preprocessor.prepare_encoder_input(features)
    print(f"✅ Encoder input prepared: shape {encoder_input['encoder_input'].shape}")

    # Validate output
    print("\n5. Testing output validation...")
    try:
        preprocessor.validate_output(encoder_input)
        print("✅ Output validation passed")
    except Exception as e:
        print(f"❌ Output validation failed: {e}")

    print("\n6. Running integration test...")
    test_integration = TestIntegration()
    test_integration.test_end_to_end_processing()

    print("\n🎵 MT3 Audio Preprocessing Pipeline Tests Complete! 🎵")
    print("\nTo run full test suite:")
    print("pytest test_audio_preprocessing.py -v")