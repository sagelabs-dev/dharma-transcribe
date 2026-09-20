"""Tests for LLM correction configuration gating.

The bug this guards against
---------------------------
Stripping the DHARMA_LLM_* environment variables (or running without a
configured LLM endpoint) used to leave the Stage-6 gate open: the pipeline
entered LLM correction, and every segment then hit the unconfigured API
client, logging one "API error on segment" warning per chunk — 6,868
warnings in one production run, with zero corrections applied.

The correct behavior: an unconfigured LLM is a DISABLED stage, skipped in
one line, transcript untouched.
"""

import dharma_transcribe.llm_correct as lc


class TestUnconfiguredSkip:
    """Missing LLM env config means the stage is disabled, not failed."""

    def test_unconfigured_llm_skips_cleanly(self, monkeypatch):
        monkeypatch.setattr(lc, "LLM_API_URL", "")
        monkeypatch.setattr(lc, "LLM_API_KEY", "")

        transcript = {"segments": [{"text": "untouched", "start": 0.0, "end": 1.0}]}
        out = lc.llm_correct_transcript(transcript)

        assert out is transcript  # same object, untouched
        assert out["segments"][0]["text"] == "untouched"
        assert "llm_corrected" not in out["segments"][0]
        assert "text_pre_llm" not in out["segments"][0]

    def test_url_without_key_still_skips(self, monkeypatch):
        """Partial config is still unconfigured — do not half-start."""
        monkeypatch.setattr(lc, "LLM_API_URL", "https://example.invalid/v1")
        monkeypatch.setattr(lc, "LLM_API_KEY", "")

        transcript = {"segments": [{"text": "untouched"}]}
        out = lc.llm_correct_transcript(transcript)

        assert out is transcript
        assert "llm_corrected" not in out["segments"][0]

    def test_empty_segments_short_circuits(self, monkeypatch):
        monkeypatch.setattr(lc, "LLM_API_URL", "")
        monkeypatch.setattr(lc, "LLM_API_KEY", "")

        out = lc.llm_correct_transcript({"segments": []})
        assert out == {"segments": []}
