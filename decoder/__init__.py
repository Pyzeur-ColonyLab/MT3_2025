"""
MT3 Token Decoder Module

This module provides functionality to decode MT3 model tokens into MIDI files.
Based on the original MT3 implementation from Google Magenta.
"""

from .event_codec import Event, EventRange, Codec
from .vocabularies import (
    VocabularyConfig,
    GenericTokenVocabulary,
    build_codec,
    vocabulary_from_codec,
    velocity_to_bin,
    bin_to_velocity,
    num_velocity_bins_from_codec,
    DECODED_EOS_ID,
    DECODED_INVALID_ID,
)
from .note_sequences import (
    TrackSpec,
    NoteEventData,
    extract_track,
    trim_overlapping_notes,
    assign_instruments,
    validate_note_sequence,
    note_arrays_to_note_sequence,
    note_sequence_to_onsets,
    note_sequence_to_onsets_and_offsets,
    note_sequence_to_onsets_and_offsets_and_programs,
)

__all__ = [
    # Event codec
    'Event',
    'EventRange',
    'Codec',
    # Vocabularies
    'VocabularyConfig',
    'GenericTokenVocabulary',
    'build_codec',
    'vocabulary_from_codec',
    'velocity_to_bin',
    'bin_to_velocity',
    'num_velocity_bins_from_codec',
    'DECODED_EOS_ID',
    'DECODED_INVALID_ID',
    # Note sequences
    'TrackSpec',
    'NoteEventData',
    'extract_track',
    'trim_overlapping_notes',
    'assign_instruments',
    'validate_note_sequence',
    'note_arrays_to_note_sequence',
    'note_sequence_to_onsets',
    'note_sequence_to_onsets_and_offsets',
    'note_sequence_to_onsets_and_offsets_and_programs',
]

__version__ = '1.0.0'