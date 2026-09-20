#!/usr/bin/env bash
#
# Run the dharma transcription pipeline over a corpus, under tmux, with live
# output AND a durable log.
#
# Why a script: the previous ad-hoc tmux one-liners redirected stdout to a file
# (`> log 2>&1`), which left the tmux pane empty and unusable for live watching.
# `tee` gives both. This also centralises the environment ritual that is easy to
# get wrong:
#
#   * .env is NOT auto-loaded by the package — `set -a; . ./.env; set +a`.
#   * HF_TOKEN must be exported for Stage 4 diarization; without it the stage is
#     skipped (not failed) and transcripts come out un-diarized.
#   * the venv must be active for whisperx/pyannote to import.
#
# Usage:
#   scripts/run_batch.sh <input-path> [log-name]
#
# Example:
#   scripts/run_batch.sh "/media/.../Kongokai Mandala Series" kongokai
#
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

INPUT="${1:?usage: run_batch.sh <input-path> [log-name]}"
LOG_NAME="${2:-batch}"
LOG="logs/${LOG_NAME}.log"
LOG_DIR="logs"
SESSION="${LOG_NAME}"

mkdir -p "$LOG_DIR"

# --- Environment ritual -------------------------------------------------------
set -a
# shellcheck disable=SC1091
. ./.env
set +a

# Prefer the cached HF token file; fall back to whatever .env provided.
if [[ -f "$HOME/.cache/huggingface/token" ]]; then
  HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
  export HF_TOKEN
fi

# shellcheck disable=SC1091
source venv/bin/activate

# --- Pre-flight ---------------------------------------------------------------
# Downloads and diarization both need external reachability. A dropped default
# route has silently broken download batches before (route-guard should
# self-heal within ~2 min, but fail loudly rather than mysteriously).
if [[ -z "$(ip route show default 2>/dev/null)" ]]; then
  echo "WARNING: no default route — model downloads may fail" >&2
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARNING: HF_TOKEN unset — Stage 4 diarization will be SKIPPED" >&2
else
  echo "HF_TOKEN: present (diarization enabled)"
fi

# --- Launch -------------------------------------------------------------------
# ${PIPESTATUS[0]} captures python's exit code, NOT tee's — a pipe makes `$?`
# report the last command in the pipeline.
echo "Logging to $LOG (tail -f to watch live)"
echo "tmux session: $SESSION"
echo "---"

python -m dharma_transcribe "$INPUT" 2>&1 | tee "$LOG"
echo "DONE=${PIPESTATUS[0]}" >> "$LOG"
