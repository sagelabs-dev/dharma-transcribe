"""Probe: diarize a chunk with the StatsPool guard applied."""
import sys, time
sys.path.insert(0, "/home/guan/src/dharma-transcribe/src")

from dharma_transcribe.pyannote_compat import apply_patch
from whisperx.diarize import DiarizationPipeline

wav = sys.argv[1]
print(f"[probe] applying patch...", flush=True)
print(f"[probe] apply_patch() -> {apply_patch()}", flush=True)

token = open("/home/guan/.cache/huggingface/token").read().strip()
p = DiarizationPipeline(token=token, device="cuda")
t = time.time()
df = p(wav)
spk = sorted(df.speaker.unique().tolist()) if len(df) else []
print(f"[probe] DONE in {time.time()-t:.1f}s rows={len(df)} speakers={spk}", flush=True)
