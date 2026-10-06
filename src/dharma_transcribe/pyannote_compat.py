"""Compatibility shim for pyannote-audio's ``StatsPool`` hang.

Upstream defect
---------------
pyannote-audio's ``StatsPool.forward`` computes an *unbiased* standard
deviation over the frame dimension::

    std = sequences.std(dim=-1, correction=1)

When an audio segment yields a frame dimension of size 1 (observed tensor
shape ``(1, 2560, 1)``), the unbiased estimator divides by ``n - 1 == 0``.
PyTorch does not raise — it returns ``NaN`` and emits::

    UserWarning: std(): degrees of freedom is <= 0.

Those NaNs propagate into the diarization pipeline, which then stalls
indefinitely (no crash, no error, process alive). The weighted ``_pool``
path in the same module is guarded by ``1e-8`` epsilons; only the
``weights is None`` branch is exposed.

A 2.5h recording hung for 21+ hours on this before the cause was isolated.

Upstream status
---------------
Issue: pyannote/pyannote-audio#1861 (open since 2025-04-20).
Fix PR: pyannote/pyannote-audio#2047 — still OPEN / unmerged as of this
writing. The guard below is the community patch from that thread, kept
faithful to upstream so behaviour is unchanged everywhere except the
degenerate case.

Design notes
------------
* We delegate to the ORIGINAL ``forward`` for every non-degenerate input,
  so the weighted / speaker-dimension logic keeps working exactly as
  upstream intended. Only the crashing case is intercepted.
* ``apply_patch()`` is idempotent and self-verifying: it is a no-op if
  upstream has already shipped the fix.
"""

from __future__ import annotations

import torch

_PATCH_MARKER = "_dharma_stats_pool_guard"

_ORIGINAL_FORWARD = None


def _guarded_forward(self, sequences, weights=None):
    """Drop-in ``StatsPool.forward`` with the degenerate frame case guarded.

    Mirrors upstream ``StatsPool.forward``, delegating all non-degenerate
    inputs to the original implementation. Only a frame dimension of size 1
    — where the unbiased std is undefined — is intercepted, and there the
    standard deviation is reported as zero (matching the upstream proposal:
    a single frame has no spread).

    Args:
        self: The ``StatsPool`` module (bound by monkeypatch).
        sequences: ``(batch, features, frames)`` feature tensor.
        weights: Optional ``(batch, frames)`` or ``(batch, speakers, frames)``
            weights, forwarded verbatim to the original implementation.

    Returns:
        Concatenated mean/std tensor, as upstream.
    """
    # Degenerate case: a single frame has no spread, and the unbiased std
    # divides by (n - 1) == 0. Intercept only the unweighted branch — the
    # weighted ``_pool`` path is epsilon-guarded upstream and handles n == 1
    # correctly (var collapses to 0), so we delegate there untouched.
    if weights is None and sequences.size(-1) == 1:
        mean = sequences.mean(dim=-1)
        return torch.cat([mean, torch.zeros_like(mean)], dim=-1)

    return _ORIGINAL_FORWARD(self, sequences, weights)


def apply_patch() -> bool:
    """Install the guard on ``pyannote.audio.models.blocks.pooling.StatsPool``.

    Idempotent and self-verifying: returns ``False`` without side effects if
    the patch is already installed or if upstream has shipped a fixed version
    (detected by probing the current implementation for the unguarded call).

    Returns:
        ``True`` if this call installed the guard, ``False`` if it was a no-op.
    """
    global _ORIGINAL_FORWARD

    from pyannote.audio.models.blocks.pooling import StatsPool

    if getattr(StatsPool.forward, _PATCH_MARKER, False):
        return False

    # Probe: if the installed implementation already guards the degenerate
    # case, leave it alone rather than shadowing a fixed upstream.
    try:
        probe = StatsPool()
        probe(torch.zeros(1, 8, 1))
    except Exception:
        # Probe failed for an unrelated reason (or is expensive); proceed with
        # the guard, which preserves original behaviour for all other inputs.
        pass
    else:
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            probe(torch.zeros(1, 8, 1))
        if not any("degrees of freedom" in str(w.message) for w in caught):
            # Upstream already handles it — do not shadow.
            return False

    _ORIGINAL_FORWARD = StatsPool.forward
    setattr(_guarded_forward, _PATCH_MARKER, True)
    # Monkey-patching the method IS this module's purpose (a compat shim);
    # mypy's method-assign check does not apply here.
    StatsPool.forward = _guarded_forward  # type: ignore[method-assign]
    return True
