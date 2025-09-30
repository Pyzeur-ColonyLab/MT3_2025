#!/usr/bin/env python3
"""
MT3 Models Package

PyTorch implementation of MT3 (Music Transcription with Transformers) models.
"""

from .mt3_model import (
    MT3Model,
    MT3Config,
    create_mt3_model,
)

__all__ = [
    'MT3Model',
    'MT3Config',
    'create_mt3_model',
]

__version__ = '1.0.0'