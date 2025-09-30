"""
MT3 Inference Example Script

Demonstrates how to use the MT3 inference pipeline to transcribe audio to MIDI.
"""

import argparse
import os
from inference import MT3Inference


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio to MIDI using MT3"
    )

    parser.add_argument(
        "audio_path",
        type=str,
        help="Path to input audio file (WAV, MP3, FLAC, etc.)"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="mt3_converted.pth",
        help="Path to MT3 checkpoint file (default: mt3_converted.pth)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path for output MIDI file (default: auto-generated)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Device for inference (default: auto-detect)"
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Maximum token sequence length (default: 1024)"
    )

    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use sampling instead of greedy decoding"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature if --sample is used (default: 1.0)"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling if --sample is used"
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Nucleus (top-p) sampling if --sample is used"
    )

    parser.add_argument(
        "--long-audio",
        action="store_true",
        help="Use chunked processing for long audio files (>30s)"
    )

    parser.add_argument(
        "--velocity-bins",
        type=int,
        default=1,
        choices=[1, 127],
        help="Velocity resolution: 1 (simple) or 127 (full range)"
    )

    args = parser.parse_args()

    # Validate audio file exists
    if not os.path.exists(args.audio_path):
        print(f"❌ Error: Audio file not found: {args.audio_path}")
        return

    # Validate checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"❌ Error: Checkpoint not found: {args.checkpoint}")
        print(f"   Expected: {args.checkpoint}")
        print(f"   Please provide correct checkpoint path with --checkpoint")
        return

    print(f"\n{'='*60}")
    print(f"MT3 Audio → MIDI Transcription")
    print(f"{'='*60}\n")

    # Initialize inference handler
    print("🔧 Initializing MT3 inference handler...")
    try:
        inference = MT3Inference(
            checkpoint_path=args.checkpoint,
            device=args.device or ("cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"),
            num_velocity_bins=args.velocity_bins,
        )
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return

    # Transcribe
    print(f"\n🎵 Transcribing: {args.audio_path}")
    print(f"   Output: {args.output or 'auto-generated'}")
    print(f"   Method: {'Chunked processing' if args.long_audio else 'Single pass'}")
    print(f"   Decoding: {'Sampling' if args.sample else 'Greedy'}")

    try:
        if args.long_audio:
            # Use chunked processing for long audio
            result = inference.transcribe_long_audio(
                audio_path=args.audio_path,
                output_path=args.output,
                max_length=args.max_length,
                do_sample=args.sample,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )
        else:
            # Standard single-pass transcription
            result = inference.transcribe(
                audio_path=args.audio_path,
                output_path=args.output,
                max_length=args.max_length,
                do_sample=args.sample,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )

        # Display results
        print(f"\n{'='*60}")
        print(f"✅ Transcription complete!")
        print(f"{'='*60}")
        print(f"   MIDI file: {result['midi_path']}")
        print(f"   Notes: {result['num_notes']}")
        if 'duration' in result:
            print(f"   Duration: {result['duration']:.2f}s")
        if 'num_chunks' in result:
            print(f"   Chunks: {result['num_chunks']}")
        print(f"{'='*60}\n")

        # Show note distribution
        if result['num_notes'] > 0:
            ns = result['note_sequence']
            programs = set(note.program for note in ns.notes if not note.is_drum)
            drums = any(note.is_drum for note in ns.notes)

            print(f"📊 Transcription summary:")
            print(f"   Instruments detected: {len(programs)}")
            if programs:
                print(f"   Program IDs: {sorted(programs)}")
            print(f"   Drums: {'Yes' if drums else 'No'}")
            print()

    except Exception as e:
        print(f"\n❌ Transcription failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()