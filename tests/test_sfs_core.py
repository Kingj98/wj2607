"""Minimal regression tests for the code-only SFS archive."""

from __future__ import annotations

import numpy as np

from sfs.sfs_1d.compute import compute_sfs_1d_with_field
from sfs.sfs_2d.compute import compute_sfs_2d_with_field


def test_constant_velocity_has_zero_sfs_flux() -> None:
    u = np.full((96, 25), 0.2, dtype=float)
    v = np.full((96, 25), -0.1, dtype=float)

    summary_1d, pi_1d, mask_1d = compute_sfs_1d_with_field(
        u, v, ell_km=20.0, dx_km=2.0, dy_km=2.0
    )
    summary_2d, pi_2d, mask_2d = compute_sfs_2d_with_field(
        u, v, ell_km=20.0, dx_km=2.0, dy_km=2.0
    )

    assert int(summary_1d["n_valid_inner"]) > 0
    assert int(summary_2d["n_valid_inner"]) > 0
    assert float(np.nanmax(np.abs(pi_1d[mask_1d]))) < 1.0e-24
    assert float(np.nanmax(np.abs(pi_2d[mask_2d]))) < 1.0e-24
