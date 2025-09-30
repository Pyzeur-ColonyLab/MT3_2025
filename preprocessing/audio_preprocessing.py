#!/usr/bin/env python3
"""
MT3 Audio Preprocessing Pipeline

Comprehensive audio preprocessing for Music Transcription with Transformers (MT3).
Converts raw audio files to mel-scale spectrograms compatible with MT3 encoder.

Features:
- MT3-compliant mel-scale spectrogram computation (256 mel bins, 16kHz sample rate)
- Memory-efficient chunked processing for long audio files
- Batch processing capabilities
- Multiple audio format support (wav, mp3, flac, etc.)
- GPU acceleration support
- Comprehensive error handling and validation

Author: Claude Code
"""

import os
import warnings
from typing import Optional, List, Tuple, Union, Dict, Any
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
import librosa
import soundfile as sf
from tqdm import tqdm


@dataclass
class AudioPreprocessingConfig:
    """Configuration for MT3 audio preprocessing."""

    # Audio parameters (MT3 standards)
    sample_rate: int = 16000
    n_mels: int = 256  # Use 256 for better quality
    hop_length: int = 320  # Use 320 for efficiency
    win_length: int = 512
    n_fft: int = 1024
    fmin: float = 0.0
    fmax: Optional[float] = 8000.0  # Nyquist for 16kHz

    # Frame processing
    frame_size: int = 512  # Samples for chunking
    overlap_ratio: float = 0.1  # 10% overlap between frames

    # Feature processing
    log_offset: float = 1e-8  # Offset for log-scale computation
    normalize: bool = True
    normalize_mean: float = -4.0  # Typical log-mel mean
    normalize_std: float = 4.0   # Typical log-mel std

    # Processing parameters
    max_length_seconds: float = 300.0  # 5 minutes max
    device: str = 'cpu'  # Will be set to cuda if available
    dtype: torch.dtype = torch.float32

    def __post_init__(self):
        """Validate configuration and set device."""
        if self.fmax is None:
            self.fmax = self.sample_rate / 2

        # Auto-detect device if not explicitly set
        if self.device == 'cpu' and torch.cuda.is_available():
            self.device = 'cuda'

        # Validate parameters
        assert self.n_mels > 0, "n_mels must be positive"
        assert self.hop_length > 0, "hop_length must be positive"
        assert self.sample_rate > 0, "sample_rate must be positive"
        assert 0 <= self.overlap_ratio < 1, "overlap_ratio must be in [0, 1)"
        assert self.max_length_seconds > 0, "max_length_seconds must be positive"


class AudioPreprocessor:
    """
    MT3 Audio Preprocessing Pipeline

    Handles conversion of raw audio files to MT3-compatible mel-scale spectrograms.
    Designed for efficient processing of variable-length audio with memory optimization.

    Example:
        >>> preprocessor = AudioPreprocessor()
        >>> features = preprocessor.process_file("audio.wav")
        >>> encoder_input = preprocessor.prepare_encoder_input(features)
    """

    def __init__(self, config: Optional[AudioPreprocessingConfig] = None):
        """
        Initialize audio preprocessor.

        Args:
            config: Preprocessing configuration. If None, uses defaults.
        """
        self.config = config or AudioPreprocessingConfig()

        # Initialize mel filter bank on correct device
        self._mel_filters = None
        self._init_mel_filters()

        # Supported audio formats
        self.supported_formats = {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg'}

    def _init_mel_filters(self):
        """Initialize mel-scale filter bank for consistent processing."""
        mel_filters = librosa.filters.mel(
            sr=self.config.sample_rate,
            n_fft=self.config.n_fft,
            n_mels=self.config.n_mels,
            fmin=self.config.fmin,
            fmax=self.config.fmax
        )

        # Convert to tensor and move to device
        self._mel_filters = torch.from_numpy(mel_filters).to(
            device=self.config.device,
            dtype=self.config.dtype
        )

    def load_audio(self, audio_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """
        Load audio file with format validation and error handling.

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (audio_data, original_sample_rate)

        Raises:
            FileNotFoundError: If audio file doesn't exist
            ValueError: If audio format not supported or file corrupted
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if audio_path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported audio format: {audio_path.suffix}")

        try:
            # Load audio with librosa for consistent handling
            audio, sr = librosa.load(
                str(audio_path),
                sr=None,  # Keep original sample rate initially
                mono=True,  # Convert to mono
                dtype=np.float32
            )

            # Validate audio data
            if len(audio) == 0:
                raise ValueError(f"Empty audio file: {audio_path}")

            # Check duration
            duration = len(audio) / sr
            if duration > self.config.max_length_seconds:
                warnings.warn(
                    f"Audio duration ({duration:.1f}s) exceeds maximum "
                    f"({self.config.max_length_seconds}s). Will be truncated."
                )
                max_samples = int(self.config.max_length_seconds * sr)
                audio = audio[:max_samples]

            return audio, sr

        except Exception as e:
            raise ValueError(f"Failed to load audio file {audio_path}: {str(e)}")

    def resample_audio(self, audio: np.ndarray, original_sr: int) -> np.ndarray:
        """
        Resample audio to target sample rate if needed.

        Args:
            audio: Audio data
            original_sr: Original sample rate

        Returns:
            Resampled audio data
        """
        if original_sr == self.config.sample_rate:
            return audio

        # Use librosa for high-quality resampling
        return librosa.resample(
            audio,
            orig_sr=original_sr,
            target_sr=self.config.sample_rate,
            res_type='kaiser_best'  # High quality resampling
        )

    def compute_features(self, audio: np.ndarray, sample_rate: int = None) -> Tensor:
        """
        Compute mel-scale spectrogram features from audio.

        Args:
            audio: Audio data (1D numpy array)
            sample_rate: Sample rate (if None, uses config default)

        Returns:
            Features tensor [seq_len, n_mels]
        """
        if sample_rate is None:
            sample_rate = self.config.sample_rate

        # Resample if needed
        if sample_rate != self.config.sample_rate:
            audio = self.resample_audio(audio, sample_rate)

        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio).to(
            device=self.config.device,
            dtype=self.config.dtype
        )

        # Compute STFT
        stft = torch.stft(
            audio_tensor,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            window=torch.hann_window(self.config.win_length).to(
                device=self.config.device, dtype=self.config.dtype
            ),
            return_complex=True
        )

        # Compute magnitude spectrum
        magnitude = torch.abs(stft)  # [n_fft//2 + 1, time_steps]

        # Apply mel filter bank
        mel_spec = torch.matmul(self._mel_filters, magnitude)  # [n_mels, time_steps]

        # Convert to log scale with offset
        log_mel = torch.log(mel_spec + self.config.log_offset)

        # Transpose to [time_steps, n_mels] format
        features = log_mel.transpose(0, 1)

        # Normalize if requested
        if self.config.normalize:
            features = (features - self.config.normalize_mean) / self.config.normalize_std

        return features

    def audio_to_frames(
        self,
        audio_path: Union[str, Path],
        frame_size: Optional[int] = None
    ) -> Tuple[List[Tensor], List[float]]:
        """
        Load audio and chunk into overlapping frames for batch processing.

        Args:
            audio_path: Path to audio file
            frame_size: Frame size in samples (if None, uses config)

        Returns:
            Tuple of (features_list, timestamps_list)
            - features_list: List of feature tensors [seq_len, n_mels]
            - timestamps_list: List of start timestamps in seconds
        """
        if frame_size is None:
            frame_size = self.config.frame_size

        # Load audio
        audio, original_sr = self.load_audio(audio_path)

        # Resample if needed
        audio = self.resample_audio(audio, original_sr)

        # Calculate overlap
        overlap_samples = int(frame_size * self.config.overlap_ratio)
        step_size = frame_size - overlap_samples

        # Generate frame boundaries
        total_samples = len(audio)
        frame_starts = list(range(0, total_samples - frame_size + 1, step_size))

        # Include final frame if there's remaining audio
        if frame_starts[-1] + frame_size < total_samples:
            frame_starts.append(total_samples - frame_size)

        features_list = []
        timestamps_list = []

        for start_idx in tqdm(frame_starts, desc="Processing frames", disable=len(frame_starts) < 10):
            end_idx = min(start_idx + frame_size, total_samples)
            frame_audio = audio[start_idx:end_idx]

            # Pad if necessary (final frame might be shorter)
            if len(frame_audio) < frame_size:
                frame_audio = np.pad(frame_audio, (0, frame_size - len(frame_audio)))

            # Compute features for this frame
            frame_features = self.compute_features(frame_audio)
            features_list.append(frame_features)

            # Calculate timestamp
            timestamp = start_idx / self.config.sample_rate
            timestamps_list.append(timestamp)

        return features_list, timestamps_list

    def prepare_encoder_input(
        self,
        features: Union[Tensor, List[Tensor]],
        max_seq_len: Optional[int] = None
    ) -> Dict[str, Tensor]:
        """
        Prepare features for MT3 encoder input with proper batching and padding.

        Args:
            features: Either single features tensor [seq_len, n_mels] or list of tensors
            max_seq_len: Maximum sequence length for padding (if None, use longest)

        Returns:
            Dictionary with:
            - 'encoder_input': Tensor [batch_size, seq_len, n_mels]
            - 'attention_mask': Tensor [batch_size, seq_len] (1 for real data, 0 for padding)
            - 'seq_lengths': Tensor [batch_size] with actual sequence lengths
        """
        # Handle single tensor input
        if isinstance(features, Tensor):
            features = [features]

        batch_size = len(features)

        # Get sequence lengths
        seq_lengths = torch.tensor([feat.size(0) for feat in features], dtype=torch.long)

        # Determine maximum sequence length
        if max_seq_len is None:
            max_seq_len = seq_lengths.max().item()

        # Initialize tensors
        encoder_input = torch.zeros(
            batch_size, max_seq_len, self.config.n_mels,
            device=self.config.device, dtype=self.config.dtype
        )
        attention_mask = torch.zeros(
            batch_size, max_seq_len,
            device=self.config.device, dtype=torch.bool
        )

        # Fill tensors with data
        for i, feat in enumerate(features):
            seq_len = min(feat.size(0), max_seq_len)
            encoder_input[i, :seq_len] = feat[:seq_len].to(
                device=self.config.device, dtype=self.config.dtype
            )
            attention_mask[i, :seq_len] = True

        return {
            'encoder_input': encoder_input,
            'attention_mask': attention_mask,
            'seq_lengths': seq_lengths.to(self.config.device)
        }

    def process_file(
        self,
        audio_path: Union[str, Path],
        return_frames: bool = False
    ) -> Union[Tensor, Tuple[List[Tensor], List[float]]]:
        """
        Complete preprocessing pipeline for a single audio file.

        Args:
            audio_path: Path to audio file
            return_frames: If True, return frame list; if False, return concatenated features

        Returns:
            Either features tensor [seq_len, n_mels] or (features_list, timestamps)
        """
        if return_frames:
            return self.audio_to_frames(audio_path)

        # Load and process entire file
        audio, original_sr = self.load_audio(audio_path)
        audio = self.resample_audio(audio, original_sr)
        return self.compute_features(audio)

    def process_batch(
        self,
        audio_paths: List[Union[str, Path]],
        max_seq_len: Optional[int] = None
    ) -> Dict[str, Tensor]:
        """
        Process multiple audio files in batch.

        Args:
            audio_paths: List of audio file paths
            max_seq_len: Maximum sequence length for padding

        Returns:
            Batch dictionary ready for encoder
        """
        features_list = []

        for path in tqdm(audio_paths, desc="Processing audio files"):
            features = self.process_file(path)
            features_list.append(features)

        return self.prepare_encoder_input(features_list, max_seq_len)

    def validate_output(self, encoder_input: Dict[str, Tensor]) -> bool:
        """
        Validate encoder input format and dimensions.

        Args:
            encoder_input: Output from prepare_encoder_input

        Returns:
            True if valid, raises ValueError otherwise
        """
        required_keys = {'encoder_input', 'attention_mask', 'seq_lengths'}
        if not required_keys.issubset(encoder_input.keys()):
            raise ValueError(f"Missing required keys. Expected: {required_keys}")

        enc_input = encoder_input['encoder_input']
        attention_mask = encoder_input['attention_mask']
        seq_lengths = encoder_input['seq_lengths']

        # Check dimensions
        if enc_input.dim() != 3:
            raise ValueError(f"encoder_input should be 3D, got {enc_input.dim()}D")

        batch_size, seq_len, n_mels = enc_input.shape

        if n_mels != self.config.n_mels:
            raise ValueError(f"Expected {self.config.n_mels} mel bins, got {n_mels}")

        if attention_mask.shape != (batch_size, seq_len):
            raise ValueError(f"attention_mask shape mismatch: {attention_mask.shape}")

        if seq_lengths.shape != (batch_size,):
            raise ValueError(f"seq_lengths shape mismatch: {seq_lengths.shape}")

        return True


# Standalone functions for compatibility
def audio_to_frames(
    audio_path: Union[str, Path],
    sample_rate: int = 16000,
    frame_size: int = 512,
    config: Optional[AudioPreprocessingConfig] = None
) -> Tuple[List[Tensor], List[float]]:
    """
    Standalone function to load audio and chunk into frames.

    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate
        frame_size: Frame size in samples
        config: Optional preprocessing configuration

    Returns:
        Tuple of (features_list, timestamps_list)
    """
    if config is None:
        config = AudioPreprocessingConfig(sample_rate=sample_rate, frame_size=frame_size)

    preprocessor = AudioPreprocessor(config)
    return preprocessor.audio_to_frames(audio_path, frame_size)


def compute_features(
    audio: np.ndarray,
    sample_rate: int = 16000,
    config: Optional[AudioPreprocessingConfig] = None
) -> Tensor:
    """
    Standalone function to compute mel-scale spectrogram features.

    Args:
        audio: Audio data (1D numpy array)
        sample_rate: Sample rate
        config: Optional preprocessing configuration

    Returns:
        Features tensor [seq_len, n_mels]
    """
    if config is None:
        config = AudioPreprocessingConfig(sample_rate=sample_rate)

    preprocessor = AudioPreprocessor(config)
    return preprocessor.compute_features(audio, sample_rate)


def prepare_encoder_input(
    features: Union[Tensor, List[Tensor]],
    max_seq_len: Optional[int] = None,
    config: Optional[AudioPreprocessingConfig] = None
) -> Dict[str, Tensor]:
    """
    Standalone function to prepare features for MT3 encoder.

    Args:
        features: Features tensor(s)
        max_seq_len: Maximum sequence length
        config: Optional preprocessing configuration

    Returns:
        Dictionary with encoder inputs
    """
    if config is None:
        config = AudioPreprocessingConfig()

    preprocessor = AudioPreprocessor(config)
    return preprocessor.prepare_encoder_input(features, max_seq_len)


if __name__ == "__main__":
    # Example usage and testing
    print("MT3 Audio Preprocessing Pipeline")
    print("=" * 40)

    # Create preprocessor
    config = AudioPreprocessingConfig(device='cpu')  # Force CPU for testing
    preprocessor = AudioPreprocessor(config)

    print(f"Configuration:")
    print(f"  Sample rate: {config.sample_rate} Hz")
    print(f"  Mel bins: {config.n_mels}")
    print(f"  Hop length: {config.hop_length}")
    print(f"  Device: {config.device}")

    # Test with synthetic audio
    print("\nTesting with synthetic audio...")
    test_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 2, 32000))  # 2 seconds at 16kHz

    features = preprocessor.compute_features(test_audio)
    print(f"Features shape: {features.shape}")

    # Test encoder input preparation
    encoder_input = preprocessor.prepare_encoder_input(features)
    print(f"Encoder input shape: {encoder_input['encoder_input'].shape}")
    print(f"Attention mask shape: {encoder_input['attention_mask'].shape}")

    # Validate output
    try:
        preprocessor.validate_output(encoder_input)
        print("✅ Output validation passed")
    except Exception as e:
        print(f"❌ Output validation failed: {e}")

    print("\nPreprocessing pipeline ready for integration!")