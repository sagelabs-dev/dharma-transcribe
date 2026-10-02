"""Dharma Audio Transcription Pipeline.

A headless multilingual transcription pipeline designed for Buddhist dharma
teachings. Handles mixed-language audio (Tibetan, Sanskrit, English, Japanese)
with per-language forced alignment, speaker diarization, Tibetan second-pass
transcription using dharma-trained models, and optional LLM post-correction.

Stages:
    1. Ingest — ffmpeg audio extraction, checksum idempotency
    2. Transcription — WhisperX large-v3, automatic language detection
    3. Alignment — wav2vec2 per-language forced alignment
    4. Diarization — pyannote speaker identification
    5. Tibetan Second-Pass — OpenPecha dharma-trained model re-transcription
    6. LLM Correction — OpenAI-compatible API post-correction (skippable)
    7. Output — JSON, SRT, VTT, TXT, review queue
"""

__version__ = "0.1.2"
