"""Main pipeline orchestrator — chains all 7 stages.

This module is importable as a library. For CLI usage, see cli.py.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import cast

from .align import align_transcript
from .config import MANIFEST_FILE
from .diarize import diarize_transcript
from .ingest import ingest_file
from .llm_correct import llm_correct_transcript
from .output import generate_all_outputs
from .tibetan_second_pass import tibetan_second_pass
from .transcribe import transcribe_with_metadata


def load_manifest() -> dict:
    """Load or create the processing manifest.

    The manifest tracks processed files for idempotency — re-running
    the pipeline skips files already marked as completed.

    Returns:
        Manifest dict with 'files' mapping and 'last_updated' timestamp.
    """
    if MANIFEST_FILE.exists():
        return cast(dict, json.loads(MANIFEST_FILE.read_text()))
    return {"files": {}, "last_updated": None}


def save_manifest(manifest: dict):
    """Save the processing manifest to disk.

    Args:
        manifest: Manifest dict to persist.
    """
    manifest["last_updated"] = datetime.utcnow().isoformat() + "Z"
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))


def process_file(media_path: Path, hf_token: str = "", skip_llm: bool = False) -> dict:
    """Process a single media file through all 7 pipeline stages.

    Args:
        media_path: Path to the input audio/video file.
        hf_token: HuggingFace token for diarization. Empty string skips diarization.
        skip_llm: If True, skip LLM post-correction (Stage 6).

    Returns:
        Manifest entry dict with status, stages completed, outputs, and timing.
    """
    source_name = media_path.name
    print(f"\n{'=' * 60}", flush=True)
    print(f"PROCESSING: {source_name}", flush=True)
    print(f"{'=' * 60}", flush=True)

    start_time = time.time()
    stages_completed: list[str] = []

    # === STAGE 1: INGEST ===
    print("\n[STAGE 1: INGEST]", flush=True)
    metadata = ingest_file(media_path)
    if metadata is None:
        print("  FAILED: ingest returned None", flush=True)
        return {"status": "failed", "stage": "ingest", "error": "ingest failed"}

    print(f"  Duration: {metadata['duration']:.1f}s", flush=True)
    print(f"  WAV: {Path(metadata['wav_path']).name}", flush=True)
    stages_completed.append("ingest")

    # === STAGE 2: TRANSCRIPTION ===
    print("\n[STAGE 2: TRANSCRIPTION]", flush=True)
    transcript = transcribe_with_metadata(metadata["wav_path"], source_name)
    if transcript is None:
        return {"status": "failed", "stage": "transcribe", "error": "transcription returned None"}
    stages_completed.append("transcribe")

    # === STAGE 3: ALIGNMENT ===
    print("\n[STAGE 3: ALIGNMENT]", flush=True)
    transcript = align_transcript(transcript, metadata["wav_path"], hf_token)
    stages_completed.append("align")

    # === STAGE 4: DIARIZATION ===
    print("\n[STAGE 4: DIARIZATION]", flush=True)
    transcript = diarize_transcript(transcript, metadata["wav_path"], hf_token)
    stages_completed.append("diarize")

    # === STAGE 5: TIBETAN SECOND PASS ===
    print("\n[STAGE 5: TIBETAN SECOND PASS]", flush=True)
    transcript = tibetan_second_pass(transcript, metadata["wav_path"])
    stages_completed.append("tibetan_second_pass")

    # === STAGE 6: LLM CORRECTION ===
    if not skip_llm:
        print("\n[STAGE 6: LLM CORRECTION]", flush=True)
        transcript = llm_correct_transcript(transcript)
        stages_completed.append("llm_correct")
    else:
        print("\n[STAGE 6: LLM CORRECTION] SKIPPED", flush=True)

    # === STAGE 7: OUTPUT ===
    print("\n[STAGE 7: OUTPUT]", flush=True)
    outputs = generate_all_outputs(transcript, source_name)
    stages_completed.append("output")
    print(f"  JSON: {outputs.get('json', 'N/A')}", flush=True)
    print(f"  SRT:  {outputs.get('srt', 'N/A')}", flush=True)
    print(f"  VTT:  {outputs.get('vtt', 'N/A')}", flush=True)
    print(f"  TXT:  {outputs.get('txt', 'N/A')}", flush=True)
    if "review" in outputs:
        print(f"  REVIEW: {outputs['review']}", flush=True)

    elapsed = time.time() - start_time
    print(f"\n  TOTAL TIME: {elapsed:.1f}s ({elapsed / 60:.1f} min)", flush=True)

    return {
        "status": "completed",
        "stages": stages_completed,
        "source_file": source_name,
        "checksum": metadata["checksum"],
        "duration": metadata["duration"],
        "processing_time": round(elapsed, 1),
        "outputs": outputs,
        "completed_at": datetime.utcnow().isoformat() + "Z",
    }
