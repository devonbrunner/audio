#!/usr/bin/env python3
"""
Novice-friendly, production-grade transcription with faster-whisper.

Usage:
  python transcribe.py "path/to/audio_or_video.mp3"
  # Outputs:
  #   <basename>.txt  (clean text)
  #   <basename>.srt  (subtitles with timestamps)
  #   <basename>.vtt  (web captions)

Tips:
- Works with most audio/video formats (mp3, wav, m4a, mp4, mov, etc.)
- Handles long files (streaming decode).
- Auto-detects language and chooses efficient compute settings.
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import timedelta

from faster_whisper import WhisperModel


def format_timestamp(seconds: float) -> str:
    """Format seconds to SRT timestamp (HH:MM:SS,mmm)."""
    if seconds is None:
        return "00:00:00,000"
    td = timedelta(seconds=float(seconds))
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int((td.total_seconds() - total_seconds) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_txt(segments, out_txt: Path):
    with out_txt.open("w", encoding="utf-8") as f:
        for seg in segments:
            text = (seg.text or "").strip()
            if text:
                f.write(text + " ")
    # tidy extra space/newline
    content = out_txt.read_text(encoding="utf-8").strip() + "\n"
    out_txt.write_text(content, encoding="utf-8")


def write_srt(segments, out_srt: Path):
    with out_srt.open("w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            start = format_timestamp(seg.start)
            end = format_timestamp(seg.end)
            text = (seg.text or "").strip()
            if not text:
                continue
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")


def write_vtt(segments, out_vtt: Path):
    with out_vtt.open("w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for seg in segments:
            start = format_timestamp(seg.start).replace(",", ".")
            end = format_timestamp(seg.end).replace(",", ".")
            text = (seg.text or "").strip()
            if not text:
                continue
            f.write(f"{start} --> {end}\n{text}\n\n")


def pick_compute_settings():
    """
    Sensible defaults:
    - device='auto' lets faster-whisper choose GPU if available, else CPU.
    - compute_type:
        * 'int8_float16' is great on GPU (fast + accurate).
        * fallback to 'int8' on CPU for speed.
    """
    device = "auto"
    # Default; faster-whisper will still work if device is CPU
    compute_type = "int8_float16"
    try:
        # Light heuristic: if CUDA not available, use int8 for CPU speed
        import torch
        if not torch.cuda.is_available():
            compute_type = "int8"
    except Exception:
        compute_type = "int8"
    return device, compute_type


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio/video to TXT/SRT/VTT with faster-whisper.")
    parser.add_argument("input_path", help="Path to audio/video file (e.g., .mp3, .wav, .m4a, .mp4)")
    parser.add_argument("--model", default="base", help="Whisper model size: tiny|base|small|medium|large-v3 (default: base)")
    parser.add_argument("--language", default=None, help="Force language code (e.g., en). If omitted, auto-detect.")
    parser.add_argument("--beam_size", type=int, default=5, help="Beam search size (quality/speed tradeoff).")
    parser.add_argument("--vad", action="store_true", help="Enable voice activity detection (may help noisy audio).")
    args = parser.parse_args()

    src = Path(args.input_path).expanduser().resolve()
    if not src.exists():
        print(f"ERROR: File not found: {src}", file=sys.stderr)
        sys.exit(1)

    out_base = src.with_suffix("")  # remove extension
    out_txt = out_base.with_suffix(".txt")
    out_srt = out_base.with_suffix(".srt")
    out_vtt = out_base.with_suffix(".vtt")

    device, compute_type = pick_compute_settings()
    print(f"[INFO] Loading model='{args.model}' device='{device}' compute_type='{compute_type}' ...")
    model = WhisperModel(args.model, device=device, compute_type=compute_type)

    print(f"[INFO] Transcribing: {src.name}")
    segments, info = model.transcribe(
        str(src),
        language=args.language,
        beam_size=args.beam_size,
        vad_filter=args.vad,        # requires Silero VAD (auto-handled by faster-whisper)
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    # Collect all segments (generator -> list) so we can write multiple formats
    segs = list(segments)

    # Write outputs
    write_txt(segs, out_txt)
    write_srt(segs, out_srt)
    write_vtt(segs, out_vtt)

    # Print summary
    detected_lang = args.language or (getattr(info, "language", None) or "unknown")
    print(f"[DONE] Language: {detected_lang}")
    print(f"[DONE] Saved:\n  - {out_txt}\n  - {out_srt}\n  - {out_vtt}")


if __name__ == "__main__":
    main()
