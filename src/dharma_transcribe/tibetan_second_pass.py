"""Stage 5: Tibetan second-pass transcription using OpenPecha model."""

from typing import cast

import numpy as np
import torch
import torchaudio

from . import config
from .config import TIBETAN_MODEL_HF
from .gpu import flush_gpu, vram_free_mb


def _slice_audio(wav_path: str, start: float, end: float) -> np.ndarray:
    """Slice audio from WAV file at given timestamps."""
    waveform, sample_rate = torchaudio.load(wav_path)
    start_sample = int(start * sample_rate)
    end_sample = int(end * sample_rate)
    return cast("np.ndarray", waveform[:, start_sample:end_sample].numpy())


def tibetan_second_pass(transcript: dict, wav_path: str) -> dict:
    """
    Re-transcribe Tibetan-detected segments using OpenPecha whisper-small model.
    Pick the better transcription between primary and second-pass.
    """
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    # Find Tibetan segments
    bo_segments = [
        (i, seg)
        for i, seg in enumerate(transcript["segments"])
        if seg.get("language", "") == "bo" or _looks_tibetan(seg.get("text", ""))
    ]

    if not bo_segments:
        print("  [stage5] No Tibetan segments detected — skipping second pass", flush=True)
        return transcript

    print(f"  [stage5] Found {len(bo_segments)} Tibetan segments to re-transcribe", flush=True)

    # Load OpenPecha model
    print(f"  [stage5] Loading {TIBETAN_MODEL_HF}...", flush=True)

    processor = WhisperProcessor.from_pretrained(
        TIBETAN_MODEL_HF,
        language="bo",
        task="transcribe",
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        TIBETAN_MODEL_HF,
    ).to(config.DEVICE)

    model.eval()
    if config.DEVICE != "cpu":
        print(f"  [stage5] Model loaded. VRAM free: {vram_free_mb()}MB", flush=True)

    corrections_made = 0

    for idx, seg in bo_segments:
        try:
            # Slice audio for this segment
            start = seg.get("start", 0)
            end = seg.get("end", start + 30)
            audio_slice = _slice_audio(wav_path, start, end)

            # Process with OpenPecha model
            inputs = processor(
                audio_slice,
                sampling_rate=16000,
                return_tensors="pt",
            ).input_features.to(config.DEVICE)

            with torch.no_grad():
                predicted_ids = model.generate(
                    inputs,
                    max_new_tokens=225,
                )

            second_pass_text = processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True,
            )[0].strip()

            primary_text = seg.get("text", "").strip()

            # Pick the better one
            # Heuristic: if second-pass has Tibetan script characters, prefer it
            if second_pass_text and _has_tibetan_script(second_pass_text):
                if second_pass_text != primary_text:
                    print(f"  [stage5] seg {idx}: replaced primary with OpenPecha", flush=True)
                    print(f"    was: {primary_text[:80]}...", flush=True)
                    print(f"    now: {second_pass_text[:80]}...", flush=True)
                    seg["text_original"] = primary_text
                    seg["text"] = second_pass_text
                    seg["tibetan_second_pass"] = True
                    corrections_made += 1
            else:
                # Second pass didn't produce Tibetan script — keep primary
                seg["tibetan_second_pass_attempted"] = True

        except Exception as e:
            print(f"  [stage5] Failed on segment {idx}: {e}", flush=True)
            seg["tibetan_second_pass_error"] = str(e)

    transcript["tibetan_second_pass"] = {
        "model": TIBETAN_MODEL_HF,
        "segments_checked": len(bo_segments),
        "corrections_made": corrections_made,
    }

    print(
        f"  [stage5] Second pass complete: {corrections_made} corrections "
        f"out of {len(bo_segments)} segments",
        flush=True,
    )

    del model
    del processor
    flush_gpu()
    if config.DEVICE != "cpu":
        print(f"  [stage5] Tibetan model flushed. VRAM free: {vram_free_mb()}MB", flush=True)

    return transcript


def _looks_tibetan(text: str) -> bool:
    """Heuristic: check if text contains Tibetan Unicode range characters."""
    if not text:
        return False
    tibetan_chars = sum(1 for c in text if 0x0F00 <= ord(c) <= 0x0FFF)
    return tibetan_chars > len(text) * 0.1


def _has_tibetan_script(text: str) -> bool:
    """Check if text contains Tibetan Unicode script characters."""
    return any(0x0F00 <= ord(c) <= 0x0FFF for c in text)
