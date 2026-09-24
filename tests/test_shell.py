from __future__ import annotations

import numpy as np

from shell import (
    antisymmetric_transfer,
    bandpass_shell_components,
    build_shell_grid,
    cumulative_shell_flux,
    make_log_shells,
    shell_frequency_masks,
    shell_to_shell_matrix,
)


def test_default_geometric_grid_has_26_shells() -> None:
    edges = make_log_shells(10, 1000, 1.2)
    assert edges[0] == 10
    assert edges[-1] == 1000
    assert build_shell_grid(edges).n_shells == 26


def test_masks_do_not_overlap() -> None:
    grid = build_shell_grid([10, 20, 40, 80])
    k, masks = shell_frequency_masks(512, 1000, grid)
    coverage = np.sum(np.stack(masks), axis=0)
    assert coverage.shape == k.shape
    assert coverage.max() <= 1


def test_shell_components_reconstruct_covered_signal() -> None:
    n = 512
    coordinate = np.arange(n)
    signal = np.sin(2 * np.pi * 32 * coordinate / n) + 0.5 * np.cos(2 * np.pi * 16 * coordinate / n)
    field = np.column_stack((signal, 2 * signal))
    grid = build_shell_grid([10, 20, 40, 80])
    _, masks = shell_frequency_masks(n, 1000, grid)
    reconstructed = np.sum(bandpass_shell_components(field, masks), axis=0)
    np.testing.assert_allclose(reconstructed, field - field.mean(axis=0), atol=1e-12, rtol=1e-12)


def test_constant_velocity_has_zero_shell_transfer() -> None:
    grid = build_shell_grid([10, 20, 40, 80, 160, 320])
    result = shell_to_shell_matrix(np.full((256, 5), 0.2), np.full((256, 5), -0.1), 2000, 2000, grid)
    assert np.nanmax(np.abs(result["matrix"])) < 1e-24


def test_antisymmetric_and_cumulative_identities() -> None:
    rng = np.random.default_rng(2)
    grid = build_shell_grid([10, 20, 40, 80, 160])
    skew = antisymmetric_transfer(rng.standard_normal((grid.n_shells, grid.n_shells)))
    np.testing.assert_allclose(skew + skew.T, 0, atol=1e-12, rtol=1e-12)
    flux = cumulative_shell_flux(skew, grid)
    valid = np.isfinite(flux["pi_total"])
    np.testing.assert_allclose(
        flux["pi_total"][valid],
        flux["pi_local"][valid] + flux["pi_nonlocal"][valid],
        atol=1e-12,
        rtol=1e-12,
    )
