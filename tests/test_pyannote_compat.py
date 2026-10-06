"""Regression tests for the pyannote StatsPool guard (pyannote#1861).

The defect: ``StatsPool.forward`` took an *unbiased* std over the frame
dimension. A single-frame segment (shape ``(b, f, 1)``) divides by
``n - 1 == 0``, producing NaN and — inside the diarization pipeline — an
indefinite freeze. These tests pin the guard's behaviour so the failure
cannot silently return.
"""

import warnings

import pytest
import torch

from dharma_transcribe.pyannote_compat import _guarded_forward, apply_patch


@pytest.fixture(scope="module")
def stats_pool():
    """Return the patched StatsPool.

    The guard is applied here rather than relying on module import order, so
    these tests are self-contained and order-independent. Without it the
    degenerate cases genuinely reproduce the upstream defect (NaN output),
    which is exactly what the guard exists to prevent.
    """
    pyannote_pooling = pytest.importorskip("pyannote.audio.models.blocks.pooling")
    apply_patch()
    return pyannote_pooling.StatsPool


class TestDegenerateFrameCase:
    """The exact shape that caused the 21-hour hang."""

    def test_single_frame_does_not_warn(self, stats_pool):
        """A single-frame input must not emit the zero-degrees-of-freedom warning."""
        module = stats_pool()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            module(torch.zeros(1, 2560, 1))

        assert not any("degrees of freedom" in str(w.message) for w in caught), (
            "degenerate frame case still produces the reduction warning"
        )

    def test_single_frame_is_finite(self, stats_pool):
        """The guard must yield finite values, not NaN (NaN is what hangs pyannote)."""
        module = stats_pool()
        out = module(torch.ones(1, 8, 1))

        assert torch.isfinite(out).all(), "guarded output contains NaN/Inf"

    def test_single_frame_std_is_zero(self, stats_pool):
        """One frame has no spread; the std half of the output must be zero."""
        module = stats_pool()
        features = 8
        out = module(torch.ones(1, features, 1))

        # Output layout is [mean..., std...]; assert the std half is zero.
        std_half = out[..., features:]
        assert torch.allclose(std_half, torch.zeros_like(std_half))

    def test_single_frame_mean_preserved(self, stats_pool):
        """The guard must not disturb the mean half of the output."""
        module = stats_pool()
        features = 4
        x = torch.arange(1.0, features + 1).reshape(1, features, 1)
        out = module(x)

        mean_half = out[..., :features]
        assert torch.allclose(mean_half, x.squeeze(-1))


class TestNormalCasesUnchanged:
    """Non-degenerate inputs must behave exactly as upstream intends."""

    @pytest.mark.parametrize("frames", [2, 3, 17, 100])
    def test_multi_frame_matches_reference(self, stats_pool, frames):
        """Multi-frame output must equal a hand-computed mean/std concat."""
        torch.manual_seed(0)
        module = stats_pool()
        features = 6
        x = torch.randn(2, features, frames)

        out = module(x)

        expected_mean = x.mean(dim=-1)
        expected_std = x.std(dim=-1, correction=1)
        expected = torch.cat([expected_mean, expected_std], dim=-1)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_no_warning_on_normal_input(self, stats_pool):
        """Normal multi-frame audio must stay warning-free."""
        module = stats_pool()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            module(torch.randn(1, 32, 50))

        assert not any("degrees of freedom" in str(w.message) for w in caught), (
            "normal input unexpectedly warns"
        )


class TestApplyPatch:
    """``apply_patch`` must be idempotent and self-verifying."""

    def test_apply_patch_targets_the_right_class(self, stats_pool):
        """After applying, the installed forward must be the guarded one."""
        apply_patch()
        assert getattr(stats_pool.forward, "_dharma_stats_pool_guard", False)

    def test_apply_patch_is_idempotent(self, stats_pool):
        """A second call must be a no-op, not a double-wrap."""
        first = apply_patch()
        second = apply_patch()

        assert (first, second) in {(True, False), (False, False)}

    def test_guarded_forward_delegates_when_weights_present(self, stats_pool):
        """The weighted branch is upstream-guarded and must be delegated to."""
        module = stats_pool()
        sequences = torch.randn(1, 4, 10)
        weights = torch.rand(1, 1, 10)

        out = _guarded_forward(module, sequences, weights)

        assert torch.isfinite(out).all()
