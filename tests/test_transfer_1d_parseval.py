"""Parseval-style checks for ``transfer_spectrum_T1`` (time mean vs ∫ T_1 dk)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from flux.transfer_1d import audit_parseval_transfer_t1_column, tukey_window_symmetric  # noqa: E402


def test_parseval_no_hann_identity_random() -> None:
    rng = np.random.default_rng(42)
    n = 512
    dx = 2.5
    ue = rng.standard_normal(n)
    un = rng.standard_normal(n)
    ae = rng.standard_normal(n)
    an = rng.standard_normal(n)
    out = audit_parseval_transfer_t1_column(
        ue,
        un,
        ae,
        an,
        dx,
        detrend="linear",
        apply_hann=False,
        apply_window_power_correction=True,
        rtol=1e-6,
    )
    assert out["no_hann_identity_ok"] is True
    assert abs(float(out["residual_no_hann_sum"])) < 1e-5 * max(
        abs(float(out["time_mean_dot_no_window"])),
        1.0,
    )


def test_parseval_hann_wpc_differs_from_nowindow_time_mean() -> None:
    """Hann + ⟨w²⟩ 补偿不保证 ∫T₁ dk 与无窗 ⟨u·A⟩ 相等（泄漏）。"""
    rng = np.random.default_rng(7)
    n = 512
    dx = 2.5
    ue = rng.standard_normal(n)
    un = rng.standard_normal(n)
    ae = rng.standard_normal(n)
    an = rng.standard_normal(n)
    out = audit_parseval_transfer_t1_column(
        ue,
        un,
        ae,
        an,
        dx,
        detrend="linear",
        apply_hann=True,
        apply_window_power_correction=True,
        rtol=1e-9,
    )
    assert bool(out["hann_identity_ok"]) is False
    assert abs(float(out["residual_hann_wpc_sum"])) > 1e-4 * max(
        abs(float(out["time_mean_dot_no_window"])),
        1e-6,
    )
    assert abs(float(out["hann_vs_nowindow_abs_ratio"]) - 1.0) > 0.01


def test_tukey_window_matches_scipy() -> None:
    scipy = pytest.importorskip("scipy")
    from scipy.signal.windows import tukey as scipy_tukey

    for m in (8, 16, 32, 64, 101):
        a = 0.2
        w = tukey_window_symmetric(m, a)
        ref = scipy_tukey(m, a, sym=True)
        assert w.shape == ref.shape
        np.testing.assert_allclose(w, ref, rtol=0.0, atol=1e-12)


def test_parseval_tukey_wpc_differs_from_nowindow_time_mean() -> None:
    rng = np.random.default_rng(11)
    n = 512
    dx = 2.5
    ue = rng.standard_normal(n)
    un = rng.standard_normal(n)
    ae = rng.standard_normal(n)
    an = rng.standard_normal(n)
    out = audit_parseval_transfer_t1_column(
        ue,
        un,
        ae,
        an,
        dx,
        detrend="linear",
        apply_hann=True,
        apply_window_power_correction=True,
        along_track_taper="tukey",
        tukey_alpha=0.2,
        rtol=1e-9,
    )
    assert bool(out["hann_identity_ok"]) is False


def test_parseval_detrend_none_consistency() -> None:
    rng = np.random.default_rng(1)
    n = 256
    dx = 1.0
    ue = rng.standard_normal(n)
    un = rng.standard_normal(n)
    ae = rng.standard_normal(n)
    an = rng.standard_normal(n)
    out = audit_parseval_transfer_t1_column(
        ue,
        un,
        ae,
        an,
        dx,
        detrend=None,
        apply_hann=False,
        rtol=1e-6,
    )
    assert out["no_hann_identity_ok"] is True
