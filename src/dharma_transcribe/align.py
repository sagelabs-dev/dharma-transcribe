"""Stage 3: Forced alignment via wav2vec2 per-language."""

import warnings

from . import config
from .config import ALIGN_MODELS
from .gpu import flush_gpu, vram_free_mb


def align_transcript(transcript: dict, wav_path: str, hf_token: str = "") -> dict:
    """
    Run forced alignment on transcript segments.
    Uses per-language wav2vec2 models.
    Patches WhisperX to support bo and sa.
    """
    import whisperx

    language = transcript.get("detected_language", "en")

    # Check if we have an alignment model for this language
    align_model_name = ALIGN_MODELS.get(language)

    if align_model_name is None and language != "en":
        # No alignment model for this language — skip
        print(f"  [stage3] No alignment model for '{language}' — skipping alignment", flush=True)
        return transcript

    print(f"  [stage3] Loading alignment model for '{language}'...", flush=True)

    try:
        model_a, metadata = whisperx.load_align_model(
            language_code=language,
            device=config.DEVICE,
            model_name=align_model_name,
        )
    except Exception as e:
        print(f"  [stage3] Failed to load alignment model for '{language}': {e}", flush=True)
        return transcript

    print("  [stage3] Aligning segments...", flush=True)

    try:
        result_aligned = whisperx.align(
            transcript["segments"],
            model_a,
            metadata,
            wav_path,
            device=config.DEVICE,
            return_char_alignments=False,
        )
        transcript["segments"] = result_aligned["segments"]
        transcript["aligned"] = True

        # Count words with confidence
        word_count = 0
        low_conf_count = 0
        for seg in transcript["segments"]:
            for word in seg.get("words", []):
                word_count += 1
                if word.get("score", 1.0) < 0.5:
                    low_conf_count += 1

        transcript["alignment_stats"] = {
            "total_words": word_count,
            "low_confidence_words": low_conf_count,
            "low_confidence_pct": round(low_conf_count / max(word_count, 1) * 100, 1),
        }

        print(
            f"  [stage3] Aligned {word_count} words ({low_conf_count} low-confidence)", flush=True
        )

    except Exception as e:
        print(f"  [stage3] Alignment failed: {e}", flush=True)
        transcript["aligned"] = False
        warnings.warn(f"Alignment failed for {wav_path}: {e}", stacklevel=2)

    # Free alignment model
    del model_a
    flush_gpu()

    if config.DEVICE != "cpu":
        print(f"  [stage3] Alignment model flushed. VRAM free: {vram_free_mb()}MB", flush=True)

    return transcript
