# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

## [0.1.1] - 2026-09-20

### Fixed
- **Stage 4 (diarization) no longer hangs.** Two root causes, both fixed and
  verified on production (4-session Kongokai series, zero hangs):
  - Guard against pyannote `StatsPool` divide-by-zero on single-frame
    segments (upstream pyannote-audio#1861; our `pyannote_compat.py` shim
    intercepts only the degenerate branch).
  - Fix `multiprocessing.Queue` deadlock in the Stage-4 subprocess wrapper:
    the child blocked in `queue.put()` on results larger than the OS pipe
    buffer while the parent waited in `proc.join()` without draining. The
    queue is now drained while waiting; the result always lands.
- **Stage 4 subprocess watchdog**: hard timeout (`DHARMA_DIARIZE_TIMEOUT`,
  default 3600s) so a stalled diarization pass can never freeze a run
  indefinitely.
- **Stage 6 (LLM correction) treats an unconfigured LLM as disabled**
  instead of attempting (and warning on) every segment: missing
  `DHARMA_LLM_API_URL` / `DHARMA_LLM_API_KEY` now skips the stage cleanly
  in one line. Previously produced one "API error on segment" warning per
  chunk (6,868 warnings in one production run, zero corrections).

### Added

- GitHub Actions CI pipeline (lint, typecheck, test on Python 3.10/3.11/3.12)
- PyPI publishing workflow (trusted publishing via OIDC)
- Pre-commit hooks configuration (ruff + mypy)
- `.env.example` with all configurable environment variables
- `CONTRIBUTING.md` with development setup and testing guide
- `tests/fixtures/` directory with README for test audio samples

## [0.1.0] - 2026-08-23

### Added
- 7-stage pipeline architecture: Ingest → WhisperX → Alignment → Diarization → Tibetan Second-Pass → LLM Correction → Output
- WhisperX large-v3 primary transcription with automatic language detection
- Per-language forced alignment (wav2vec2 for en, ja, bo, sa)
- Speaker diarization via pyannote speaker-diarization-community-1
- Tibetan second-pass transcription using OpenPecha op-whisper_small-ft-v2 (dharma-trained)
- LLM post-correction via any OpenAI-compatible API (cloud or local)
- Output formats: JSON, SRT, VTT, TXT, review queue
- Corrections dictionary for deterministic pre-LLM fixes
- Idempotent manifest for batch processing
- VRAM management with model flushing between stages (6 GB consumer GPU support)
- `--device cpu` flag for CPU-only mode
- `--skip-llm` flag to skip LLM correction
- Environment-variable-based configuration (no hardcoded secrets)
- `src/` package layout with `pyproject.toml` (hatchling build backend)
- 54 unit tests covering all modules
- MIT license

### Security
- Git history scrubbed of all API keys, URLs, and private paths
- All sensitive values read from environment variables
- No credentials in source code or commit history
