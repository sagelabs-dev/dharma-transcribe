"""Integration tests for pipeline.py — manifest and stage orchestration."""

from unittest.mock import patch

from dharma_transcribe import pipeline


def test_load_manifest_nonexistent(tmp_path, monkeypatch):
    """load_manifest should create empty manifest when file doesn't exist."""
    monkeypatch.setattr("dharma_transcribe.config.MANIFEST_FILE", tmp_path / "nonexistent.json")
    monkeypatch.setattr("dharma_transcribe.pipeline.MANIFEST_FILE", tmp_path / "nonexistent.json")

    result = pipeline.load_manifest()
    assert result == {"files": {}, "last_updated": None}


def test_save_and_load_manifest(tmp_path, monkeypatch):
    """save_manifest then load_manifest should round-trip correctly."""
    manifest_path = tmp_path / "manifest.json"
    monkeypatch.setattr("dharma_transcribe.config.MANIFEST_FILE", manifest_path)
    monkeypatch.setattr("dharma_transcribe.pipeline.MANIFEST_FILE", manifest_path)

    manifest = {"files": {"test.mp4": {"status": "completed"}}, "last_updated": None}
    pipeline.save_manifest(manifest)

    loaded = pipeline.load_manifest()
    assert loaded["files"]["test.mp4"]["status"] == "completed"
    assert loaded["last_updated"] is not None


def test_process_file_ingest_failure(tmp_path, monkeypatch):
    """process_file should return failed status when ingest fails."""
    # Point all paths to tmp
    monkeypatch.setattr("dharma_transcribe.config.MANIFEST_FILE", tmp_path / "manifest.json")
    monkeypatch.setattr("dharma_transcribe.pipeline.MANIFEST_FILE", tmp_path / "manifest.json")

    with patch("dharma_transcribe.pipeline.ingest_file", return_value=None):
        result = pipeline.process_file(tmp_path / "fake.mp4")

    assert result["status"] == "failed"
    assert result["stage"] == "ingest"


def test_process_file_transcribe_failure(tmp_path, monkeypatch):
    """process_file should return failed status when transcription fails."""
    monkeypatch.setattr("dharma_transcribe.config.MANIFEST_FILE", tmp_path / "manifest.json")
    monkeypatch.setattr("dharma_transcribe.pipeline.MANIFEST_FILE", tmp_path / "manifest.json")

    mock_metadata = {"checksum": "abc", "wav_path": "/tmp/test.wav", "duration": 10.0}

    with (
        patch("dharma_transcribe.pipeline.ingest_file", return_value=mock_metadata),
        patch("dharma_transcribe.pipeline.transcribe_with_metadata", return_value=None),
    ):
        result = pipeline.process_file(tmp_path / "fake.mp4")

    assert result["status"] == "failed"
    assert result["stage"] == "transcribe"


def test_process_file_skip_llm(tmp_path, monkeypatch):
    """process_file with skip_llm=True should skip stage 6."""
    monkeypatch.setattr("dharma_transcribe.config.MANIFEST_FILE", tmp_path / "manifest.json")
    monkeypatch.setattr("dharma_transcribe.pipeline.MANIFEST_FILE", tmp_path / "manifest.json")

    mock_metadata = {"checksum": "abc", "wav_path": "/tmp/test.wav", "duration": 10.0}
    mock_transcript = {
        "segments": [{"text": "test", "start": 0, "end": 1}],
        "source_file": "test.mp4",
    }

    with (
        patch("dharma_transcribe.pipeline.ingest_file", return_value=mock_metadata),
        patch("dharma_transcribe.pipeline.transcribe_with_metadata", return_value=mock_transcript),
        patch("dharma_transcribe.pipeline.align_transcript", return_value=mock_transcript),
        patch("dharma_transcribe.pipeline.diarize_transcript", return_value=mock_transcript),
        patch("dharma_transcribe.pipeline.tibetan_second_pass", return_value=mock_transcript),
        patch("dharma_transcribe.pipeline.llm_correct_transcript") as mock_llm,
        patch(
            "dharma_transcribe.pipeline.generate_all_outputs",
            return_value={"json": "/tmp/out.json"},
        ),
    ):
        result = pipeline.process_file(tmp_path / "fake.mp4", skip_llm=True)

    assert result["status"] == "completed"
    assert "llm_correct" not in result["stages"]
    mock_llm.assert_not_called()


def test_process_file_full_pipeline(tmp_path, monkeypatch):
    """Full pipeline should call all stages in order and return completed."""
    monkeypatch.setattr("dharma_transcribe.config.MANIFEST_FILE", tmp_path / "manifest.json")
    monkeypatch.setattr("dharma_transcribe.pipeline.MANIFEST_FILE", tmp_path / "manifest.json")

    mock_metadata = {"checksum": "abc", "wav_path": "/tmp/test.wav", "duration": 10.0}
    mock_transcript = {
        "segments": [{"text": "test", "start": 0, "end": 1}],
        "source_file": "test.mp4",
    }

    stage_order = []

    def track_stage(name, fn):
        def wrapper(*args, **kwargs):
            stage_order.append(name)
            return mock_transcript

        return wrapper

    with (
        patch("dharma_transcribe.pipeline.ingest_file", return_value=mock_metadata),
        patch(
            "dharma_transcribe.pipeline.transcribe_with_metadata",
            side_effect=track_stage("transcribe", None),
        ) as mock_t,
        patch(
            "dharma_transcribe.pipeline.align_transcript", side_effect=track_stage("align", None)
        ),
        patch(
            "dharma_transcribe.pipeline.diarize_transcript",
            side_effect=track_stage("diarize", None),
        ),
        patch(
            "dharma_transcribe.pipeline.tibetan_second_pass",
            side_effect=track_stage("tibetan", None),
        ),
        patch(
            "dharma_transcribe.pipeline.llm_correct_transcript",
            side_effect=track_stage("llm", None),
        ),
        patch(
            "dharma_transcribe.pipeline.generate_all_outputs",
            return_value={"json": "/tmp/out.json"},
        ),
    ):
        # Fix transcribe_with_metadata to return the transcript
        mock_t.side_effect = lambda *a, **kw: mock_transcript

        result = pipeline.process_file(tmp_path / "fake.mp4", skip_llm=False)

    assert result["status"] == "completed"
    assert result["stages"] == [
        "ingest",
        "transcribe",
        "align",
        "diarize",
        "tibetan_second_pass",
        "llm_correct",
        "output",
    ]
