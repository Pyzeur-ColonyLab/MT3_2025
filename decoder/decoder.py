"""
MT3 Token Decoder

Converts MT3 model output tokens into MIDI files using the MT3 codec system.
"""

import functools
from typing import List, Sequence, Mapping, Any, Optional
import numpy as np
import note_seq

from . import (
    event_codec,
    vocabularies,
    note_sequences,
    run_length_encoding,
    metrics_utils,
)


class MT3TokenDecoder:
    """
    Decoder for MT3 model tokens to MIDI files.

    This class handles the complete pipeline:
    1. Token IDs → Events (via Codec)
    2. Events → NoteSequence (via encoding spec)
    3. NoteSequence → MIDI file

    Example:
        ```python
        decoder = MT3TokenDecoder()

        # Decode tokens to MIDI
        decoder.tokens_to_midi(
            tokens=generated_tokens,
            output_path="output.mid",
            start_time=0.0
        )

        # Or get NoteSequence first
        note_sequence = decoder.tokens_to_note_sequence(
            tokens=generated_tokens,
            start_time=0.0
        )
        ```
    """

    def __init__(
        self,
        steps_per_second: int = 100,
        max_shift_seconds: int = 10,
        num_velocity_bins: int = 1,
    ):
        """
        Initialize MT3 token decoder.

        Args:
            steps_per_second: Time resolution (steps per second). Default: 100
            max_shift_seconds: Maximum time shift in seconds. Default: 10
            num_velocity_bins: Number of velocity bins (1 or 127). Default: 1
                - Use 1 for simplified velocity (all notes same velocity)
                - Use 127 for full velocity range
        """
        # Create vocabulary configuration
        self.vocab_config = vocabularies.VocabularyConfig(
            steps_per_second=steps_per_second,
            max_shift_seconds=max_shift_seconds,
            num_velocity_bins=num_velocity_bins,
        )

        # Build codec (event encoder/decoder)
        self.codec = vocabularies.build_codec(self.vocab_config)

        # Create vocabulary
        self.vocab = vocabularies.vocabulary_from_codec(self.codec)

        # Use NoteEncodingWithTiesSpec (handles ties between segments)
        self.encoding_spec = note_sequences.NoteEncodingWithTiesSpec

    def tokens_to_note_sequence(
        self,
        tokens: np.ndarray,
        start_time: float = 0.0,
        max_time: Optional[float] = None,
    ) -> note_seq.NoteSequence:
        """
        Convert token sequence to NoteSequence.

        Args:
            tokens: Array of token IDs from model output [seq_len]
            start_time: Start time offset in seconds
            max_time: Maximum time (events beyond this will be dropped)

        Returns:
            note_seq.NoteSequence object containing decoded notes
        """
        # Ensure tokens are numpy array
        if not isinstance(tokens, np.ndarray):
            tokens = np.array(tokens)

        # Remove special tokens (PAD=0, EOS=1, UNK=2)
        # Model outputs are offset by num_special_tokens
        if tokens.max() >= self.vocab._base_vocab_size:
            # Need to decode from vocabulary space
            tokens = self.vocab.decode(tokens.tolist())
            tokens = np.array([t for t in tokens if t >= 0])  # Remove invalid tokens

        # Prepare predictions in format expected by event_predictions_to_ns
        predictions = [{
            'est_tokens': tokens,
            'start_time': start_time,
            'raw_inputs': np.array([]),  # Not used for simple decoding
        }]

        # Decode to NoteSequence
        result = metrics_utils.event_predictions_to_ns(
            predictions=predictions,
            codec=self.codec,
            encoding_spec=self.encoding_spec,
        )

        return result['est_ns']

    def tokens_to_midi(
        self,
        tokens: np.ndarray,
        output_path: str,
        start_time: float = 0.0,
        max_time: Optional[float] = None,
    ) -> note_seq.NoteSequence:
        """
        Convert token sequence directly to MIDI file.

        Args:
            tokens: Array of token IDs from model output [seq_len]
            output_path: Path where to save MIDI file (e.g., "output.mid")
            start_time: Start time offset in seconds
            max_time: Maximum time (events beyond this will be dropped)

        Returns:
            note_seq.NoteSequence object (also saved to MIDI)
        """
        # Decode to NoteSequence
        ns = self.tokens_to_note_sequence(tokens, start_time, max_time)

        # Save to MIDI
        note_seq.sequence_proto_to_midi_file(ns, output_path)

        return ns

    def batch_tokens_to_midi(
        self,
        token_sequences: Sequence[np.ndarray],
        frame_times: Sequence[np.ndarray],
        output_path: str,
    ) -> note_seq.NoteSequence:
        """
        Convert multiple token sequences (from chunked audio) to single MIDI file.

        This handles the case where a long audio file was split into chunks,
        each chunk was transcribed separately, and now we need to combine them.

        Args:
            token_sequences: List of token arrays from each chunk
            frame_times: List of time arrays for each chunk [num_chunks, chunk_len]
            output_path: Path where to save MIDI file

        Returns:
            Combined note_seq.NoteSequence object
        """
        predictions = []

        for tokens, times in zip(token_sequences, frame_times):
            # Decode from vocabulary if needed
            if isinstance(tokens, np.ndarray) and tokens.max() >= self.vocab._base_vocab_size:
                tokens_decoded = self.vocab.decode(tokens.tolist())
                tokens = np.array([t for t in tokens_decoded if t >= 0])

            # Get start time for this segment
            start_time = times[0] if len(times) > 0 else 0.0

            predictions.append({
                'est_tokens': tokens,
                'start_time': start_time,
                'raw_inputs': np.array([]),
            })

        # Combine all predictions
        result = metrics_utils.event_predictions_to_ns(
            predictions=predictions,
            codec=self.codec,
            encoding_spec=self.encoding_spec,
        )

        ns = result['est_ns']

        # Save to MIDI
        note_seq.sequence_proto_to_midi_file(ns, output_path)

        return ns

    def get_vocab_size(self) -> int:
        """Get vocabulary size (including special tokens)."""
        return self.vocab._base_vocab_size + self.vocab.extra_ids

    def get_codec_info(self) -> dict:
        """Get codec configuration information."""
        return {
            'steps_per_second': self.codec.steps_per_second,
            'max_shift_steps': self.codec.max_shift_steps,
            'num_classes': self.codec.num_classes,
            'num_velocity_bins': self.vocab_config.num_velocity_bins,
        }

    def decode_single_token(self, token_id: int) -> event_codec.Event:
        """
        Decode a single token ID to an Event object.

        Useful for debugging and understanding model outputs.

        Args:
            token_id: Token ID from model output

        Returns:
            Event object with type and value
        """
        # Remove special token offset if needed
        if token_id >= self.vocab._num_special_tokens:
            token_id -= self.vocab._num_special_tokens

        return self.codec.decode_event_index(token_id)

    def get_event_type_range(self, event_type: str) -> tuple:
        """
        Get the token ID range for a specific event type.

        Args:
            event_type: One of 'shift', 'pitch', 'velocity', 'tie', 'program', 'drum'

        Returns:
            (min_id, max_id) tuple
        """
        return self.codec.event_type_range(event_type)


def create_decoder(
    velocity_bins: int = 1,
    steps_per_second: int = 100,
) -> MT3TokenDecoder:
    """
    Convenience function to create MT3 decoder with common settings.

    Args:
        velocity_bins: 1 for simplified velocity, 127 for full range
        steps_per_second: Time resolution (higher = more precise timing)

    Returns:
        Configured MT3TokenDecoder instance
    """
    return MT3TokenDecoder(
        steps_per_second=steps_per_second,
        num_velocity_bins=velocity_bins,
    )