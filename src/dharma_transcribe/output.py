"""Stage 7: Multi-format output, corrections dictionary, review queue."""

import json
from datetime import datetime
from pathlib import Path
from typing import cast

from .config import (
    CONFIDENCE_THRESHOLD,
    # Re-exported for tests: tests monkeypatch dharma_transcribe.output.CORRECTIONS_DIR
    # alongside the other output dirs. Not referenced in module code itself.
    CORRECTIONS_DIR,  # noqa: F401
    CORRECTIONS_FILE,
    JSON_DIR,
    REVIEW_DIR,
    SRT_DIR,
    TXT_DIR,
    VTT_DIR,
)


def _format_timestamp(seconds: float) -> str:
    """Format seconds as SRT timestamp (HH:MM:SS,mmm)."""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{ms:03d}"


def _format_timestamp_vtt(seconds: float) -> str:
    """Format seconds as VTT timestamp (HH:MM:SS.mmm)."""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}.{ms:03d}"


def load_corrections() -> dict:
    """Load the corrections dictionary."""
    if CORRECTIONS_FILE.exists():
        return cast(dict, json.loads(CORRECTIONS_FILE.read_text()))
    return {"corrections": []}


def apply_dictionary(transcript: dict) -> dict:
    """Apply dictionary corrections (case-insensitive string replacement)."""
    corrections = load_corrections()
    if not corrections.get("corrections"):
        return transcript

    dict_applied = 0
    for seg in transcript.get("segments", []):
        text = seg.get("text", "")
        for correction in corrections["corrections"]:
            pattern = correction["pattern"]
            replacement = correction["replacement"]
            if pattern.lower() in text.lower():
                new_text = text.replace(pattern, replacement)
                if new_text != text:
                    seg["text_pre_dict"] = text
                    seg["text"] = new_text
                    dict_applied += 1
                    text = new_text

    transcript["dictionary_corrections_applied"] = dict_applied
    return transcript


def generate_json(transcript: dict, source_name: str):
    """Write full structured JSON output."""
    out_path = JSON_DIR / f"{Path(source_name).stem}.json"
    out_path.write_text(json.dumps(transcript, indent=2, ensure_ascii=False))
    return out_path


def generate_srt(transcript: dict, source_name: str):
    """Write SRT subtitle file with speaker labels."""
    lines = []
    for i, seg in enumerate(transcript.get("segments", []), 1):
        start = seg.get("start", 0)
        end = seg.get("end", start + 1)
        text = seg.get("text", "").strip()
        speaker = seg.get("speaker", "")
        if speaker:
            text = f"[{speaker}] {text}"
        lines.append(str(i))
        lines.append(f"{_format_timestamp(start)} --> {_format_timestamp(end)}")
        lines.append(text)
        lines.append("")

    out_path = SRT_DIR / f"{Path(source_name).stem}.srt"
    out_path.write_text("\n".join(lines))
    return out_path


def generate_vtt(transcript: dict, source_name: str):
    """Write VTT subtitle file."""
    lines = ["WEBVTT", ""]
    for seg in transcript.get("segments", []):
        start = seg.get("start", 0)
        end = seg.get("end", start + 1)
        text = seg.get("text", "").strip()
        speaker = seg.get("speaker", "")
        if speaker:
            text = f"<v {speaker}>{text}</v>"
        lines.append(f"{_format_timestamp_vtt(start)} --> {_format_timestamp_vtt(end)}")
        lines.append(text)
        lines.append("")

    out_path = VTT_DIR / f"{Path(source_name).stem}.vtt"
    out_path.write_text("\n".join(lines))
    return out_path


def generate_txt(transcript: dict, source_name: str):
    """Write plain text reading copy."""
    lines = []
    for seg in transcript.get("segments", []):
        text = seg.get("text", "").strip()
        speaker = seg.get("speaker", "")
        if speaker:
            lines.append(f"[{speaker}] {text}")
        else:
            lines.append(text)

    out_path = TXT_DIR / f"{Path(source_name).stem}.txt"
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def generate_review_queue(transcript: dict, source_name: str):
    """Generate review queue for low-confidence segments."""
    review_items = []
    for i, seg in enumerate(transcript.get("segments", [])):
        # Check word-level confidence
        words = seg.get("words", [])
        low_conf_words = [w for w in words if w.get("score", 1.0) < CONFIDENCE_THRESHOLD]

        # Check if LLM flagged low confidence
        llm_conf = seg.get("llm_confidence", "")

        if low_conf_words or llm_conf == "low":
            review_items.append(
                {
                    "segment_id": i,
                    "start": seg.get("start", 0),
                    "end": seg.get("end", 0),
                    "text": seg.get("text", ""),
                    "speaker": seg.get("speaker", ""),
                    "low_confidence_words": len(low_conf_words),
                    "llm_suggestion": seg.get("llm_suggestion", ""),
                    "llm_confidence": llm_conf,
                    "tibetan_second_pass": seg.get("tibetan_second_pass", False),
                }
            )

    if review_items:
        out_path = REVIEW_DIR / f"{Path(source_name).stem}_review.json"
        out_path.write_text(
            json.dumps(
                {
                    "source_file": source_name,
                    "review_items": review_items,
                    "generated_at": datetime.utcnow().isoformat() + "Z",
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return out_path
    return None


def generate_all_outputs(transcript: dict, source_name: str) -> dict:
    """Generate all output formats."""
    # Apply dictionary corrections first
    transcript = apply_dictionary(transcript)

    outputs = {}
    outputs["json"] = str(generate_json(transcript, source_name))
    outputs["srt"] = str(generate_srt(transcript, source_name))
    outputs["vtt"] = str(generate_vtt(transcript, source_name))
    outputs["txt"] = str(generate_txt(transcript, source_name))

    review = generate_review_queue(transcript, source_name)
    if review:
        outputs["review"] = str(review)

    return outputs
