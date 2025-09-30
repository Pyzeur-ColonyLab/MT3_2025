"""
MT3 Inference Handler

Complete pipeline for audio → MIDI transcription using MT3 model.
Integrates preprocessing, model inference, and token decoding.
"""

import os
from typing import Optional, List, Dict, Any
import torch
import numpy as np

from preprocessing import AudioPreprocessor, AudioPreprocessingConfig
from models import MT3Model, MT3Config
from models.checkpoint_utils import load_mt3_checkpoint
from decoder.decoder import MT3TokenDecoder


class MT3Inference:
    """
    Complete MT3 inference pipeline: audio → spectrogram → tokens → MIDI.

    Example:
        ```python
        # Initialize
        inference = MT3Inference(
            checkpoint_path="mt3_converted.pth",
            device="cuda"
        )

        # Transcribe single file
        inference.transcribe(
            audio_path="piano.wav",
            output_path="piano.mid"
        )

        # Transcribe multiple files
        inference.transcribe_batch(
            audio_files=["song1.wav", "song2.wav"],
            output_dir="output_midi/"
        )
        ```
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        preprocessor_config: Optional[AudioPreprocessingConfig] = None,
        model_config: Optional[MT3Config] = None,
        num_velocity_bins: int = 1,
    ):
        """
        Initialize MT3 inference handler.

        Args:
            checkpoint_path: Path to MT3 checkpoint (.pth file)
            device: Device for inference ('cuda' or 'cpu')
            preprocessor_config: Custom audio preprocessing config (optional)
            model_config: Custom model config (optional)
            num_velocity_bins: Velocity resolution (1=simple, 127=full range)
        """
        self.device = torch.device(device)

        # Initialize preprocessor
        if preprocessor_config is None:
            preprocessor_config = AudioPreprocessingConfig(device=device)
        self.preprocessor = AudioPreprocessor(preprocessor_config)

        # Initialize model
        if model_config is None:
            model_config = MT3Config()
        self.model = MT3Model(model_config)

        # Load checkpoint
        load_result = load_mt3_checkpoint(
            self.model,
            checkpoint_path,
            strict=False
        )

        if not load_result['success']:
            raise RuntimeError(
                f"Failed to load checkpoint: {load_result['message']}\n"
                f"Missing keys: {len(load_result['missing_keys'])}\n"
                f"Unexpected keys: {len(load_result['unexpected_keys'])}"
            )

        self.model.to(self.device)
        self.model.eval()

        # Initialize decoder
        self.decoder = MT3TokenDecoder(num_velocity_bins=num_velocity_bins)

        print(f"✅ MT3 Inference initialized on {device}")
        print(f"   Model: {load_result['parameter_count']:,} parameters loaded")
        print(f"   Vocab size: {self.decoder.get_vocab_size()}")

    def transcribe(
        self,
        audio_path: str,
        output_path: Optional[str] = None,
        max_length: int = 1024,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Transcribe single audio file to MIDI.

        Args:
            audio_path: Path to audio file
            output_path: Path for output MIDI (optional, auto-generated if None)
            max_length: Maximum token sequence length
            do_sample: Use sampling instead of greedy decoding
            temperature: Sampling temperature (if do_sample=True)
            top_k: Top-k sampling (if do_sample=True)
            top_p: Nucleus sampling (if do_sample=True)

        Returns:
            Dictionary with:
                - 'midi_path': Path to generated MIDI file
                - 'note_sequence': NoteSequence object
                - 'num_notes': Number of notes transcribed
                - 'duration': Audio duration in seconds
        """
        # Generate output path if not provided
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            output_path = f"{base_name}_transcribed.mid"

        # 1. Preprocess audio
        features = self.preprocessor.process_file(audio_path)
        encoder_input = self.preprocessor.prepare_encoder_input(features)

        # Move to device
        encoder_input = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in encoder_input.items()
        }

        # 2. Generate tokens
        with torch.no_grad():
            generated_tokens = self.model.generate(
                input_ids=encoder_input['encoder_input'],
                attention_mask=encoder_input.get('attention_mask'),
                max_length=max_length,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

        # Move to CPU and convert to numpy
        tokens = generated_tokens.cpu().numpy()[0]  # [seq_len]

        # 3. Decode to MIDI
        note_sequence = self.decoder.tokens_to_midi(
            tokens=tokens,
            output_path=output_path,
        )

        # Calculate duration
        duration = features.shape[0] / self.preprocessor.config.sample_rate

        return {
            'midi_path': output_path,
            'note_sequence': note_sequence,
            'num_notes': len(note_sequence.notes),
            'duration': duration,
        }

    def transcribe_batch(
        self,
        audio_files: List[str],
        output_dir: str = "output_midi",
        max_length: int = 1024,
        **generation_kwargs
    ) -> List[Dict[str, Any]]:
        """
        Transcribe multiple audio files in batch.

        Args:
            audio_files: List of audio file paths
            output_dir: Directory for output MIDI files
            max_length: Maximum token sequence length
            **generation_kwargs: Additional arguments for generate()

        Returns:
            List of result dictionaries (one per file)
        """
        os.makedirs(output_dir, exist_ok=True)

        results = []
        for audio_path in audio_files:
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}.mid")

            try:
                result = self.transcribe(
                    audio_path=audio_path,
                    output_path=output_path,
                    max_length=max_length,
                    **generation_kwargs
                )
                results.append(result)
                print(f"✅ {audio_path} → {result['num_notes']} notes")
            except Exception as e:
                print(f"❌ Failed: {audio_path} - {e}")
                results.append({
                    'audio_path': audio_path,
                    'error': str(e)
                })

        return results

    def transcribe_long_audio(
        self,
        audio_path: str,
        output_path: Optional[str] = None,
        chunk_length: int = 256,
        max_length: int = 1024,
        **generation_kwargs
    ) -> Dict[str, Any]:
        """
        Transcribe long audio file by chunking.

        For audio files longer than ~30 seconds, this method splits the audio
        into overlapping chunks, transcribes each chunk, and combines results.

        Args:
            audio_path: Path to audio file
            output_path: Path for output MIDI
            chunk_length: Frames per chunk (default: 256 ~= 30s)
            max_length: Maximum token sequence length per chunk
            **generation_kwargs: Additional arguments for generate()

        Returns:
            Dictionary with transcription results
        """
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            output_path = f"{base_name}_transcribed.mid"

        # 1. Preprocess with chunking
        frames, frame_times = self.preprocessor.audio_to_frames(
            audio_path,
            frame_size=chunk_length
        )

        # 2. Process each chunk
        token_sequences = []
        time_sequences = []

        for chunk_frames, chunk_times in zip(frames, frame_times):
            # Prepare encoder input for chunk
            encoder_input = self.preprocessor.prepare_encoder_input(chunk_frames)

            # Move to device
            encoder_input = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in encoder_input.items()
            }

            # Generate tokens
            with torch.no_grad():
                generated = self.model.generate(
                    input_ids=encoder_input['encoder_input'],
                    attention_mask=encoder_input.get('attention_mask'),
                    max_length=max_length,
                    **generation_kwargs
                )

            tokens = generated.cpu().numpy()[0]
            token_sequences.append(tokens)
            time_sequences.append(chunk_times)

        # 3. Combine chunks and decode
        note_sequence = self.decoder.batch_tokens_to_midi(
            token_sequences=token_sequences,
            frame_times=time_sequences,
            output_path=output_path,
        )

        return {
            'midi_path': output_path,
            'note_sequence': note_sequence,
            'num_notes': len(note_sequence.notes),
            'num_chunks': len(token_sequences),
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded model and configuration."""
        return {
            'model': {
                'parameters': self.model.get_parameter_summary(),
                'config': self.model.config.__dict__,
            },
            'preprocessor': {
                'sample_rate': self.preprocessor.config.sample_rate,
                'n_mels': self.preprocessor.config.n_mels,
                'hop_length': self.preprocessor.config.hop_length,
            },
            'decoder': self.decoder.get_codec_info(),
            'device': str(self.device),
        }


def create_inference_handler(
    checkpoint_path: str,
    device: Optional[str] = None,
) -> MT3Inference:
    """
    Convenience function to create MT3 inference handler with defaults.

    Args:
        checkpoint_path: Path to MT3 checkpoint
        device: Device ('cuda' or 'cpu'), auto-detected if None

    Returns:
        Configured MT3Inference instance
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    return MT3Inference(checkpoint_path=checkpoint_path, device=device)