# Dharma Transcription Pipeline

A headless multilingual transcription pipeline designed for Buddhist dharma teachings. Handles mixed-language audio (Tibetan, Sanskrit, English, Japanese) with per-language forced alignment, speaker diarization, a Tibetan second-pass using dharma-trained models, and optional LLM post-correction.

**Privacy-first**: all processing runs locally. Sacred content never touches a cloud unless you explicitly configure LLM correction with a cloud API.

**Version 0.1.2** — see [CHANGELOG.md](CHANGELOG.md) for release history.

## Architecture — 7 Stages

```
Audio/Video → Ingest → WhisperX → Alignment → Diarization → Tibetan 2nd-Pass → LLM Correction → Output
```

| Stage | Purpose | Model | GPU Memory |
|-------|---------|-------|-----------|
| 1. Ingest | ffmpeg extract to 16kHz mono WAV, checksum idempotency | ffmpeg | — |
| 2. Transcription | Primary ASR with auto language detection | WhisperX large-v3 (int8) | ~4 GB |
| 3. Alignment | Word-level timestamps per language | wav2vec2 (per-language) | ~1–2 GB |
| 4. Diarization | Speaker identification | pyannote diarization-community-1 | ~1 GB |
| 5. Tibetan 2nd-Pass | Re-transcribe Tibetan segments with dharma-trained model | OpenPecha op-whisper_small-ft-v2 | ~1 GB |
| 6. LLM Correction | Post-correction with dharma domain knowledge | Any OpenAI-compatible API | (cloud or local) |
| 7. Output | JSON, SRT, VTT, TXT, review queue | — | — |

Stages run serially with GPU memory flushing between each — designed for 6 GB consumer GPUs.

## Setup

### Prerequisites

- Python 3.10+
- ffmpeg + ffprobe (system install)
- NVIDIA GPU with CUDA (optional — `--device cpu` works for all stages)
- HuggingFace token (for diarization only — transcription works without it)

### Installation

```bash
# Clone
git clone https://github.com/guan-tends/dharma-transcribe.git
cd dharma-transcribe

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install with pip
pip install -e ".[dev]"

# Or install from PyPI (when published)
pip install dharma-transcribe[dev]
```

### GPU Setup (optional but recommended)

```bash
# Install PyTorch with CUDA support (adjust cuXXX for your CUDA version)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### HuggingFace Token (for diarization)

The pipeline needs a HuggingFace token to download the pyannote diarization model. Without it, diarization is skipped — transcription still completes normally.

```bash
# Get a token: https://huggingface.co/settings/tokens
# Also accept the model license: https://huggingface.co/pyannote/speaker-diarization-community-1

export HF_TOKEN=hf_your_token_here
```

### LLM Correction Configuration (optional)

Stage 6 uses any OpenAI-compatible API. Configure via environment variables:

```bash
# Cloud API example (Synthetic.new)
export DHARMA_LLM_API_URL=https://api.example.com/v1
export DHARMA_LLM_API_KEY=your-key
export DHARMA_LLM_MODEL=hf:openai/gpt-oss-120b

# Local model example (Ollama)
ollama pull qwen3.5:9b
export DHARMA_LLM_API_URL=http://localhost:11434/v1
export DHARMA_LLM_API_KEY=ollama
export DHARMA_LLM_MODEL=qwen3.5:9b

# vLLM example (local GPU)
# Start vLLM server: vllm serve Qwen/Qwen3.5-9B
export DHARMA_LLM_API_URL=http://localhost:8000/v1
export DHARMA_LLM_API_KEY=none
export DHARMA_LLM_MODEL=Qwen/Qwen3.5-9B
```

Or skip LLM correction entirely: `--skip-llm`

## Usage

```bash
# Single file
dharma-transcribe /path/to/teaching.mp4

# Directory (batch — finds all audio/video recursively)
dharma-transcribe /path/to/recordings/

# Skip LLM correction (ASR only, faster)
dharma-transcribe /path/to/teaching.mp4 --skip-llm

# CPU-only mode (no GPU required)
dharma-transcribe /path/to/teaching.mp4 --device cpu --skip-llm

# With explicit HF token
dharma-transcribe /path/to/teaching.mp4 --hf-token $HF_TOKEN
```

### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `input` (positional) | — | File or directory to process |
| `--source-dir` | — | Default source directory |
| `--skip-llm` | off | Skip LLM correction stage |
| `--hf-token` | `$HF_TOKEN` | HuggingFace token for diarization |
| `--device` | `cuda` | Compute device: `cuda` or `cpu` |

### Environment Variables

See `.env.example` for the full list. Key variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `DHARMA_LLM_API_URL` | (empty) | OpenAI-compatible API endpoint |
| `DHARMA_LLM_API_KEY` | (empty) | API key for LLM correction |
| `DHARMA_LLM_MODEL` | (empty) | Model name for LLM correction |
| `HF_TOKEN` | (empty) | HuggingFace token for diarization |
| `DHARMA_DEVICE` | `cuda` | Compute device |
| `DHARMA_WHISPER_MODEL` | `large-v3` | WhisperX model size |
| `DHARMA_OUTPUT_DIR` | `./output` | Output directory |
| `DHARMA_TIBETAN_MODEL` | `openpecha/op-whisper_small-ft-v2` | Tibetan second-pass model |

## Output Formats

Each processed file generates:

- **JSON** — full structured transcript with word-level timestamps, speaker labels, confidence scores
- **SRT** — subtitle file with speaker labels
- **VTT** — WebVTT subtitles with speaker tags
- **TXT** — plain text reading copy
- **Review queue** — JSON of low-confidence segments for manual review

Outputs are written to `output/{json,srt,vtt,txt,review}/`.

## Corrections Dictionary

`output/corrections/corrections.json` — case-insensitive string replacement applied before LLM correction. Grows from manual review.

```json
{
  "corrections": [
    {"pattern": "bodichita", "replacement": "bodhicitta"},
    {"pattern": "shun yata", "replacement": "shunyata"}
  ]
}
```

## Idempotency

The manifest (`output/manifest.json`) tracks processed files. Re-running the pipeline skips completed files. Delete the manifest to reprocess everything.

## VRAM Management

Designed for consumer GPUs (6 GB+). Stages run serially with `gc.collect()` + `torch.cuda.empty_cache()` between each. No CPU fallback needed — the GPU is flushed fully before loading the next model.

## Why This Exists

Most transcription tools handle single languages. Dharma teachings commonly mix Tibetan, Sanskrit, English, and sometimes Japanese in a single recording. This pipeline:

1. **Detects language per segment** — not per file
2. **Aligns each language separately** — wav2vec2 models for bo, sa, en, ja
3. **Re-transcribes Tibetan** — OpenPecha's model (trained on Garchen Rinpoche's teachings) often outperforms WhisperX on Tibetan
4. **Corrects with dharma knowledge** — LLM post-correction knows bodhicitta from bodichita

## License

MIT — see [LICENSE](LICENSE).

## Support

If this pipeline helps preserve dharma teachings, consider supporting its continued development:

- **Solana**: `Eu8wQcW68TKMs1a6eqzZu8znzU52QLqQugAMG8uCD6y6`
- **EVM** (Ethereum / Base / Arbitrum / Optimism / Polygon): `0x2733ff7c865C56d565a99BE1DC11B81cc76850A5`
- **XRP Ledger**: `r4X6e7McAQj7e8vBCeued1RYu4mCJrREDG`

---

Crafted with ❤️ by [Sage Labs](https://sagelabs.dev)
