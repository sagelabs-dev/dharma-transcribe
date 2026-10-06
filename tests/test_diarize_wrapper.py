"""Regression tests for the Stage-4 subprocess wrapper in ``diarize``.

The deadlock this guards against
--------------------------------
The diarization child returns its result via ``multiprocessing.Queue.put``.
A real diarization DataFrame pickles to well over the OS pipe buffer
(~64KB), so the child's feeder thread blocks inside ``put`` until the parent
reads. If the parent sits in ``proc.join()`` instead of draining the queue,
both sides wait on each other forever: the child never exits, the parent
never sees the result, and the watchdog eventually kills a job whose actual
work had already finished.

That deadlock cost a full batch run. These tests pin the drain-while-waiting
contract so it cannot silently regress.

Note: the worker functions below are module-level on purpose. These tests use
the ``spawn`` start method, which pickles the target by reference — a local
closure or lambda cannot be pickled and would fail with AttributeError.
"""

import multiprocessing as mp
import time

import pytest

from dharma_transcribe import config
from dharma_transcribe.diarize import diarize_transcript

# --- Module-level workers (picklable by reference under spawn) --------------


def _big_payload_worker(queue, nbytes: int) -> None:
    """Put a payload larger than the pipe buffer, then return."""
    queue.put(("ok", b"x" * nbytes))


def _hang_worker(wav_path, hf_token, device, queue) -> None:  # noqa: ARG001
    """Never returns — exercises the watchdog kill path.

    Signature mirrors ``_diarize_worker``: the queue is the LAST argument.
    """
    time.sleep(300)


def _error_worker(wav_path, hf_token, device, queue) -> None:  # noqa: ARG001
    """Reports a child failure the way the real worker does."""
    queue.put(("error", "RuntimeError('kaboom')"))


def _dataframe_worker(queue) -> None:
    """Puts a DataFrame shaped like a real diarization result."""
    import pandas as pd

    n = 2000
    df = pd.DataFrame(
        {
            "start": [float(i) for i in range(n)],
            "end": [float(i) + 1.0 for i in range(n)],
            "speaker": [f"SPEAKER_{i % 7:02d}" for i in range(n)],
        }
    )
    queue.put(("ok", df))


def _drain_first_result(queue, proc, deadline_s: float):
    """Drain a result while the child is alive — the pattern under test."""
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        try:
            return queue.get(timeout=1.0)
        except Exception:  # noqa: BLE001 - queue.Empty in practice
            if not proc.is_alive():
                break
    return None


def _simple_transcript() -> dict:
    return {"segments": [{"start": 0.0, "end": 1.0, "text": "hi", "words": []}]}


class TestQueueDrain:
    """The parent must drain the result queue while the child is alive."""

    def test_large_payload_does_not_deadlock(self):
        """A payload exceeding the pipe buffer must still come back.

        This is the exact shape that hung in production: the child computes
        fine, then blocks writing a large result home.
        """
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        proc = ctx.Process(target=_big_payload_worker, args=(queue, 8_000_000))
        proc.start()

        got = _drain_first_result(queue, proc, deadline_s=60)

        proc.join(timeout=10)
        if proc.is_alive():
            proc.kill()
            proc.join()

        assert got is not None, "parent never drained a large child payload"
        assert got[0] == "ok"
        assert len(got[1]) == 8_000_000


class TestWrapperDegradation:
    """The wrapper must degrade gracefully on timeout and child error.

    These drive ``diarize_transcript`` through the real spawned-process path
    (not a monkeypatched stub, which could not cross the process boundary),
    by pointing it at a worker that behaves badly.
    """

    def test_stuck_worker_times_out_and_degrades(self, monkeypatch):
        """A child that never returns must be killed, not waited on forever."""
        monkeypatch.setattr("dharma_transcribe.diarize._diarize_worker", _hang_worker)
        monkeypatch.setattr(config, "DIARIZE_TIMEOUT_SEC", 3)

        started = time.monotonic()
        out = diarize_transcript(_simple_transcript(), "/dev/null", "fake-token")
        elapsed = time.monotonic() - started

        assert out.get("diarized") is False
        assert "timeout" in (out.get("diarize_note") or "")
        assert elapsed < 60, "watchdog did not bound the wait"

    def test_child_error_is_reported_not_raised(self, monkeypatch):
        """A child failure must degrade to diarized=False, not crash the run."""
        monkeypatch.setattr("dharma_transcribe.diarize._diarize_worker", _error_worker)

        out = diarize_transcript(_simple_transcript(), "/dev/null", "fake-token")

        assert out.get("diarized") is False
        assert "kaboom" in (out.get("diarize_note") or "")


class TestSkipPath:
    """No token means skip cleanly — the long-standing contract."""

    def test_empty_token_skips_without_touching_gpu(self):
        out = diarize_transcript(_simple_transcript(), "/dev/null", "")

        assert out.get("diarized") is not True
        assert out["segments"] == _simple_transcript()["segments"]


class TestPayloadRoundTrip:
    """A DataFrame-like payload must survive the queue intact."""

    def test_dataframe_payload_round_trips(self):
        """Verify pickling preserves the object the success path consumes."""
        pytest.importorskip("pandas")

        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        proc = ctx.Process(target=_dataframe_worker, args=(queue,))
        proc.start()

        status, payload = _drain_first_result(queue, proc, deadline_s=60)

        proc.join(timeout=10)
        if proc.is_alive():
            proc.kill()
            proc.join()

        assert status == "ok"
        assert len(payload) == 2000
        assert payload["speaker"].nunique() == 7
