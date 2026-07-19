from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sfs import compute_pi_field_1d_along, compute_pi_field_2d
from transfer import parseval_residual, tail_integral_from_tk, transfer_spectrum_T1


def test_transfer_parseval_identity() -> None:
    rng = np.random.default_rng(42)
    fields = [rng.standard_normal(512) for _ in range(4)]
    residual = parseval_residual(*fields, spacing_m=2.5)
    assert abs(residual) < 1e-10


def test_tail_integral_ends_at_zero() -> None:
    k, t = transfer_spectrum_T1(np.arange(32), np.arange(32), np.arange(32), np.arange(32), 2.0)
    assert tail_integral_from_tk(k, t)[-1] == 0.0


def test_constant_field_has_zero_sfs_flux() -> None:
    u = np.full((96, 25), 0.2)
    v = np.full((96, 25), -0.1)
    for compute in (compute_pi_field_1d_along, compute_pi_field_2d):
        pi, mask, _ = compute(u, v, ell_km=20.0, dx_km=2.0, dy_km=2.0)
        assert np.isfinite(pi[mask]).any()
        assert np.nanmax(np.abs(pi[mask])) < 1e-24
