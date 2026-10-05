# Contributing

## Development Setup

```bash
git clone https://github.com/sagelabs-dev/dharma-transcribe.git
cd dharma-transcribe
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Workflow

1. Create a branch: `git checkout -b feature/your-feature`
2. Make changes. Commit early, commit often.
3. Run checks before pushing:
   ```bash
   ruff check src tests
   ruff format --check src tests
   mypy src/dharma_transcribe
   pytest
   ```
4. Push and open a PR.

## Testing

```bash
# Run all tests
pytest

# Run without GPU tests
pytest -m "not gpu"

# Run with coverage
pytest --cov=dharma_transcribe --cov-report=term-missing

# Run a single module
pytest tests/test_config.py
```

### Adding Test Fixtures

Place small audio samples in `tests/fixtures/`. A fixture should be:
- Short (5–30 seconds)
- Royalty-free or your own recording
- Named descriptively: `english_sample.wav`, `tibetan_sample.wav`

Generate a test WAV with ffmpeg:
```bash
ffmpeg -f lavfi -i "sine=frequency=440:duration=5" -ar 16000 -ac 1 tests/fixtures/tone_5s.wav
```

## Code Style

- **Linter**: ruff (replaces flake8 + isort + black)
- **Formatter**: ruff format
- **Type checking**: mypy (non-strict, `ignore_missing_imports = true`)
- **Line length**: 100 characters
- **Python**: 3.10+ (uses `str | None` union syntax)

## Project Structure

```
src/dharma_transcribe/     # Package source
  __init__.py              # Package metadata
  cli.py                  # CLI entry point (argparse)
  pipeline.py             # Stage orchestration
  config.py               # Environment-based configuration
  ingest.py               # Stage 1: audio extraction
  transcribe.py           # Stage 2: WhisperX ASR
  align.py                # Stage 3: forced alignment
  diarize.py              # Stage 4: speaker diarization
  tibetan_second_pass.py  # Stage 5: Tibetan re-transcription
  llm_correct.py          # Stage 6: LLM post-correction
  output.py               # Stage 7: multi-format output
  gpu.py                  # GPU memory utilities
tests/                    # Test suite
  conftest.py             # Shared fixtures
  test_*.py               # Unit and integration tests
  fixtures/               # Test audio samples
pyproject.toml            # Project metadata, tool config
.env.example              # Environment variable template
```
