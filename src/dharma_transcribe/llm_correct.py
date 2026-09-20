"""Stage 6: LLM post-correction via OpenAI-compatible API.

Works with any OpenAI-compatible endpoint: Synthetic.new, OpenAI, Ollama,
vLLM, LM Studio, etc. Configure via DHARMA_LLM_API_URL, DHARMA_LLM_API_KEY,
and DHARMA_LLM_MODEL environment variables.

Uses dynamic token budgeting via tiktoken to stay within model context
limits regardless of segment length.

Token budgeting:
    - Estimate input tokens with tiktoken (cl100k_base approximation)
    - max_tokens = min(generous_cap, context_window - estimated_input - margin)
"""

import json
import logging
import re
from urllib.parse import urlparse

import tiktoken
from openai import OpenAI

from .config import LLM_API_KEY, LLM_API_URL, LLM_MODEL

logger = logging.getLogger(__name__)

# --- Constants -----------------------------------------------------------------

MODEL_CONTEXT_WINDOW = 128_000
SAFETY_MARGIN = 2_048
DEFAULT_MAX_TOKENS = 4_096
MAX_MAX_TOKENS = 8_192

_ENCODER = tiktoken.get_encoding("cl100k_base")

# --- System prompt -------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a correction engine for Buddhist dharma teaching transcripts.
The transcripts contain English, Tibetan (bo), Sanskrit (sa), and occasionally Japanese (ja).

Correct transcription errors: misheard terms, garbled mantras, phonetic renderings.
Do NOT rephrase, add information, or restructure. Only fix specific errors.

Return ONLY a JSON object:
{"corrected": "...", "confidence": "high|low|none", "changes": ["was -> now", ...]}
"""

DHARMA_REFERENCE = """\
Dharma vocabulary for reference:
Teachers: Garchen Rinpoche, Chogyal Namkhai Norbu, Lama Fede Andino, Tenga Rinpoche, Karmapa
Terms: bodhicitta, shunyata, vajra, mantra, mandala, empowerments, lungta, damaru, Sowa Rigpa
Sutras: Prajnaparamita, Vajracchedika, Heart Sutra
Lineage: Padmasambhava, Yeshe Tsogyal, Naropa, Tilopa, Marpa, Milarepa, Gampopa, Atisha, Tsongkhapa
Teachings: Dzogchen, Mahamudra, Three Jewels, Four Noble Truths, Eightfold Path
Regions: Amdo, Kham, Utsang
"""


# --- Token estimation ---------------------------------------------------------


def _estimate_tokens(text: str) -> int:
    """Estimate token count using cl100k_base encoding."""
    return len(_ENCODER.encode(text))


def _compute_max_tokens(*texts: str) -> int:
    """Compute safe max_tokens given input texts.

    Returns the token budget for model output, leaving room for input
    within the model's context window. For short segments this returns
    the default cap; for very long segments it scales down.
    """
    input_estimate = sum(_estimate_tokens(t) for t in texts)
    available = MODEL_CONTEXT_WINDOW - input_estimate - SAFETY_MARGIN
    return max(min(available, DEFAULT_MAX_TOKENS), 512)


# --- Client -------------------------------------------------------------------

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """Lazy singleton — avoids constructing client at import time.

    Raises:
        RuntimeError: If LLM_API_URL or LLM_API_KEY is not configured.
    """
    global _client
    if _client is None:
        if not LLM_API_URL or not LLM_API_KEY:
            raise RuntimeError(
                "LLM correction requires DHARMA_LLM_API_URL and "
                "DHARMA_LLM_API_KEY environment variables. "
                "Use --skip-llm to disable this stage."
            )
        _client = OpenAI(base_url=LLM_API_URL, api_key=LLM_API_KEY)
    return _client


def _api_label() -> str:
    """Extract a human-readable label from the API URL for metadata."""
    if not LLM_API_URL:
        return "none"
    parsed = urlparse(LLM_API_URL)
    return parsed.hostname or LLM_API_URL


# --- Core ---------------------------------------------------------------------


def _parse_json_response(content: str) -> dict | None:
    """Extract a JSON object from an LLM response.

    Handles markdown code fences and extraneous text around the JSON.
    """
    if not content:
        return None
    clean = content.strip()
    # Strip markdown fences
    if clean.startswith("```"):
        parts = clean.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                clean = part
                break
    if clean.endswith("```"):
        clean = clean[:-3].strip()
    # Greedy match for outermost JSON object
    match = re.search(r"\{.*\}", clean, re.DOTALL)
    if match:
        clean = match.group(0)
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        return None


def correct_segment(
    segment: dict,
    prev_text: str = "",
    next_text: str = "",
) -> dict:
    """Send a single segment to the LLM for correction.

    Returns a dict with keys: corrected, confidence, changes, and
    optionally error. Never raises — errors are returned in the dict.
    """
    seg_text = segment.get("text", "")
    if not seg_text or len(seg_text.strip()) < 3:
        return {"corrected": seg_text, "confidence": "none", "changes": []}

    user_prompt = (
        f"{DHARMA_REFERENCE}\n"
        f"Previous: {prev_text[-200:] if prev_text else '(none)'}\n"
        f"Next: {next_text[:200] if next_text else '(none)'}\n\n"
        f"Segment:\n{seg_text}\n\n"
        f"Return JSON only."
    )

    max_tokens = _compute_max_tokens(SYSTEM_PROMPT, user_prompt)

    try:
        resp = _get_client().chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=max_tokens,
        )
    except Exception as e:
        logger.warning("API error on segment: %s", e)
        return {"corrected": seg_text, "confidence": "none", "changes": [], "error": str(e)}

    msg = resp.choices[0].message
    content = msg.content

    if not content:
        finish = resp.choices[0].finish_reason
        return {
            "corrected": seg_text,
            "confidence": "none",
            "changes": [],
            "error": f"Empty content (finish={finish})",
        }

    result = _parse_json_response(content)
    if result is None:
        return {
            "corrected": seg_text,
            "confidence": "none",
            "changes": [],
            "error": f"JSON parse failed: {content[:200]}",
        }

    return result


def llm_correct_transcript(transcript: dict) -> dict:
    """Run LLM correction across all segments in a transcript.

    Skips cleanly (returns the transcript untouched) when LLM correction is
    not configured — missing DHARMA_LLM_API_URL / DHARMA_LLM_API_KEY means
    the stage is disabled, not failed-per-segment.

    Dictionary corrections are applied BEFORE this stage (in output.py).
    High-confidence corrections are applied directly; low-confidence
    suggestions are stored for the review queue.
    """
    segments = transcript.get("segments", [])
    if not segments:
        return transcript

    # Config-level skip: if LLM correction is not configured, this stage is
    # DISABLED — return the transcript untouched instead of attempting and
    # failing on every segment. Running 6,868 guaranteed failures (one noisy
    # warning per chunk) is not a degradation mode, it is a bug.
    if not LLM_API_URL or not LLM_API_KEY:
        print(
            "  [stage6] LLM correction not configured (DHARMA_LLM_API_URL / "
            "DHARMA_LLM_API_KEY missing) — SKIPPED.",
            flush=True,
        )
        return transcript

    logger.info("Starting LLM correction with %s (%d segments)", LLM_MODEL, len(segments))
    print(f"  [stage6] LLM correction with {LLM_MODEL} ({len(segments)} segments)...", flush=True)
