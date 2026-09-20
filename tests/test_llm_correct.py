"""Unit tests for llm_correct.py — JSON parsing, token estimation, correction logic."""

from unittest.mock import MagicMock

from dharma_transcribe import llm_correct


def test_parse_json_response_plain():
    """Parser should extract JSON from a plain response."""
    content = '{"corrected": "test", "confidence": "high", "changes": []}'
    result = llm_correct._parse_json_response(content)
    assert result == {"corrected": "test", "confidence": "high", "changes": []}


def test_parse_json_response_with_fences():
    """Parser should handle markdown code fences."""
    content = '```json\n{"corrected": "test", "confidence": "low"}\n```'
    result = llm_correct._parse_json_response(content)
    assert result is not None
    assert result["corrected"] == "test"
    assert result["confidence"] == "low"


def test_parse_json_response_with_surrounding_text():
    """Parser should extract JSON from surrounding text."""
    content = 'Here is the result:\n{"corrected": "test", "confidence": "none"}\nDone.'
    result = llm_correct._parse_json_response(content)
    assert result is not None
    assert result["corrected"] == "test"


def test_parse_json_response_empty():
    """Parser should return None for empty content."""
    assert llm_correct._parse_json_response("") is None
    assert llm_correct._parse_json_response(None) is None


def test_parse_json_response_invalid():
    """Parser should return None for invalid JSON."""
    assert llm_correct._parse_json_response("not json at all") is None


def test_estimate_tokens():
    """Token estimation should return positive integers."""
    tokens = llm_correct._estimate_tokens("Hello world")
    assert tokens > 0
    assert isinstance(tokens, int)


def test_compute_max_tokens_short_input():
    """Short inputs should get the default max_tokens cap."""
    result = llm_correct._compute_max_tokens("short text")
    assert result == llm_correct.DEFAULT_MAX_TOKENS


def test_compute_max_tokens_long_input():
    """Very long inputs should scale down max_tokens."""
    long_text = "word " * 200_000
    result = llm_correct._compute_max_tokens(long_text)
    assert result < llm_correct.DEFAULT_MAX_TOKENS
    assert result >= 512  # never below minimum


def test_correct_segment_short_text():
    """Segments shorter than 3 chars should return without API call."""
    seg = {"text": "hi"}
    result = llm_correct.correct_segment(seg)
    assert result["confidence"] == "none"
    assert result["changes"] == []


def test_correct_segment_api_call(mock_openai_response, monkeypatch):
    """Correct segment should call the API and return parsed result."""
    monkeypatch.setattr(llm_correct, "LLM_API_URL", "https://api.test.com/v1")
    monkeypatch.setattr(llm_correct, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(llm_correct, "LLM_MODEL", "test-model")

    # Reset the client singleton
    llm_correct._client = None

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_openai_response
    monkeypatch.setattr(llm_correct, "_get_client", lambda: mock_client)

    seg = {"text": "Welcome to the teaching on bodichitta."}
    result = llm_correct.correct_segment(seg)

    assert result["confidence"] == "high"
    assert "bodhicitta" in result["corrected"]
    assert "bodichitta -> bodhicitta" in result["changes"]


def test_correct_segment_api_error(monkeypatch):
    """API errors should be caught and returned, not raised."""
    monkeypatch.setattr(llm_correct, "LLM_API_URL", "https://api.test.com/v1")
    monkeypatch.setattr(llm_correct, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(llm_correct, "LLM_MODEL", "test-model")

    llm_correct._client = None

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("API timeout")
    monkeypatch.setattr(llm_correct, "_get_client", lambda: mock_client)

    seg = {"text": "Some dharma teaching text here."}
    result = llm_correct.correct_segment(seg)

    assert result["confidence"] == "none"
    assert "error" in result


def test_api_label_from_url(monkeypatch):
    """_api_label should extract hostname from URL."""
    monkeypatch.setattr(llm_correct, "LLM_API_URL", "https://api.example.com/v1")
    assert llm_correct._api_label() == "api.example.com"

    monkeypatch.setattr(llm_correct, "LLM_API_URL", "http://localhost:11434/v1")
    assert llm_correct._api_label() == "localhost"

    monkeypatch.setattr(llm_correct, "LLM_API_URL", "")
    assert llm_correct._api_label() == "none"


def test_llm_correct_transcript_no_config(monkeypatch):
    """Unconfigured LLM = stage DISABLED: clean skip, transcript untouched.

    Historical behavior (the bug this asserts against): the stage gate only
    checked the --skip-llm CLI flag, so an unconfigured env sailed into the
    stage and every segment logged an 'API error on segment' warning
    (6,868 warnings in one production run, zero corrections). The stage now
    treats missing config as disabled and returns the transcript untouched.
    """
    monkeypatch.setattr(llm_correct, "LLM_API_URL", "")
    monkeypatch.setattr(llm_correct, "LLM_API_KEY", "")
    monkeypatch.setattr(llm_correct, "LLM_MODEL", "")

    llm_correct._client = None

    transcript = {"segments": [{"text": "Some teaching text here."}]}
    result = llm_correct.llm_correct_transcript(transcript)
    assert result is transcript  # untouched — stage skipped, not failed
    assert "llm_correction" not in result
    assert "llm_corrected" not in result["segments"][0]


def test_llm_correct_transcript_empty_segments():
    """Empty segments should return transcript unchanged."""
    transcript = {"segments": []}
    result = llm_correct.llm_correct_transcript(transcript)
    assert result == transcript
