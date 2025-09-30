"""
MT3 Audio Preprocessing Pipeline

This package provides audio preprocessing functionality for MT3 (Music Transcription with Transformers).
Converts raw audio files to model-compatible features following MT3 specifications.

Main components:
- AudioPreprocessor: Main preprocessing class
- Feature computation: Mel-scale spectrogram extraction
- Frame chunking: Audio segmentation for batch processing
- Encoder input preparation: Tensor formatting for MT3 model

Usage:
    from MT3.preprocessing import AudioPreprocessor

    preprocessor = AudioPreprocessor()
    features = preprocessor.process_file("audio.wav")
"""

from .audio_preprocessing import (
    AudioPreprocessor,
    audio_to_frames,
    compute_features,
    prepare_encoder_input,
    AudioPreprocessingConfig,
)

__all__ = [
    'AudioPreprocessor',
    'audio_to_frames',
    'compute_features',
    'prepare_encoder_input',
    'AudioPreprocessingConfig',
]

__version__ = '1.0.0'