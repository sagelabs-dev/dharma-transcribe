#!/usr/bin/env bash
# Probe: does a given pyannote model complete on a known-hanging audio chunk?
# Usage: probe_diarize_model.sh <model_name|DEFAULT> <wav> <logfile>
set -u
MODEL="${1:-DEFAULT}"; WAV="${2}"; LOG="${3}"
source /home/guan/src/dharma-transcribe/venv/bin/activate
python - "$MODEL" "$WAV" << 'PY' > "$LOG" 2>&1
import sys, time
from whisperx.diarize import DiarizationPipeline
name = None if sys.argv[1] == "DEFAULT" else sys.argv[1]
wav = sys.argv[2]
token = open("/home/guan/.cache/huggingface/token").read().strip()
print(f"[probe] model={name or 'DEFAULT'}", flush=True)
p = DiarizationPipeline(model_name=name, token=token, device="cuda")
t = time.time()
df = p(wav)
spk = sorted(df.speaker.unique().tolist()) if len(df) else []
print(f"[probe] DONE in {time.time()-t:.1f}s rows={len(df)} speakers={spk}", flush=True)
PY
echo "DONE=$?" >> "$LOG"
