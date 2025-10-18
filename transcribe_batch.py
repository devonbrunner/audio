#!/usr/bin/env python3
"""
Batch Transcriber for accessibility:
- Processes every .mp3 (or .wav, .m4a) file in a folder
- Saves .txt transcripts in an outputs/ directory
- Deletes original audio after successful transcription
- High readability: one sentence per line, line breaks for comfort
"""

from pathlib import Path
from datetime import timedelta
import re
import sys
from faster_whisper import WhisperModel

# ---------- CONFIG ----------
INPUT_DIR = Path("audio")         # Folder containing .mp3/.wav/.m4a files
OUTPUT_DIR = Path("outputs")      # Folder where .txt transcripts will be saved
MODEL_NAME = "small"               # Change to "small" or "medium" for better accuracy
LANGUAGE = "en"                   # or None for auto-detect
DELETE_SOURCE_AFTER = True        # Delete .mp3 after transcription
# -----------------------------

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")

def srt_time(seconds: float) -> str:
    td = timedelta(seconds=float(seconds))
    total = int(td.total_seconds())
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def split_into_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    return [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]

def transcribe_file(model, audio_path: Path, output_path: Path):
    print(f"\n🎧 Transcribing: {audio_path.name}")
    try:
        segments, info = model.transcribe(str(audio_path), language=LANGUAGE, beam_size=5)
        lines = [f"Source file: {audio_path.name}",
                 f"Language: {LANGUAGE or getattr(info, 'language', 'auto')}",
                 "-" * 60, ""]
        for seg in segments:
            for s in split_into_sentences(seg.text or ""):
                lines.append(s)
                lines.append("")  # spacing for readability
        output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
        print(f"✅ Saved transcript: {output_path}")
        if DELETE_SOURCE_AFTER:
            audio_path.unlink()
            print(f"🗑️ Deleted source: {audio_path}")
    except Exception as e:
        print(f"❌ Error transcribing {audio_path.name}: {e}")

def main():
    if not INPUT_DIR.exists():
        print(f"ERROR: Input folder '{INPUT_DIR}' not found.")
        sys.exit(1)

    OUTPUT_DIR.mkdir(exist_ok=True)
    device = "auto"
    compute_type = "int8"
    try:
        import torch
        if torch.cuda.is_available():
            compute_type = "int8_float16"
    except Exception:
        pass

    print(f"[INFO] Loading model '{MODEL_NAME}' ({compute_type}) ...")
    model = WhisperModel(MODEL_NAME, device=device, compute_type=compute_type)

    # Find all audio files
    audio_files = list(INPUT_DIR.glob("*.mp3")) + list(INPUT_DIR.glob("*.wav")) + list(INPUT_DIR.glob("*.m4a"))
    if not audio_files:
        print(f"No audio files found in '{INPUT_DIR}'.")
        sys.exit(0)

    print(f"Found {len(audio_files)} audio files. Starting batch...")
    for audio_path in audio_files:
        output_path = OUTPUT_DIR / (audio_path.stem + ".txt")
        transcribe_file(model, audio_path, output_path)

    print("\n🏁 All transcriptions complete!")
    print(f"Transcripts saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
