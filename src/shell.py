"""Core shell-to-shell transfer diagnostics used by the manuscript."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


LOCAL_RATIO_MIN = 0.5
LOCAL_RATIO_MAX = 2.0


@dataclass(frozen=True)
class ShellGrid:
    """Wavelength-shell geometry and its equivalent radial-wavenumber bounds."""

    wavelength_edges_km: np.ndarray
    wavelength_min_km: np.ndarray
    wavelength_max_km: np.ndarray
    wavelength_center_km: np.ndarray
    k_min_rad_per_m: np.ndarray
    k_max_rad_per_m: np.ndarray

    @property
    def n_shells(self) -> int:
        return int(self.wavelength_center_km.size)


def make_log_shells(lam_min: float, lam_max: float, scale_factor: float) -> np.ndarray:
    """Build increasing wavelength-shell edges using a geometric progression."""
    if not all(np.isfinite(x) for x in (lam_min, lam_max, scale_factor)):
        raise ValueError("lam_min, lam_max, and scale_factor must be finite")
    if lam_min <= 0 or lam_max <= lam_min:
        raise ValueError("require 0 < lam_min < lam_max")
    if scale_factor <= 1:
        raise ValueError("scale_factor must be greater than one")
    edges = [float(lam_min)]
    while edges[-1] * scale_factor < lam_max:
        edges.append(float(edges[-1] * scale_factor))
    edges.append(float(lam_max))
    out = np.asarray(edges, dtype=float)
    out[:-1] = np.round(out[:-1], 6)
    if np.any(np.diff(out) <= 0):
        raise ValueError("shell edges must be strictly increasing")
    return out


def build_shell_grid(wavelength_edges_km: np.ndarray | list[float]) -> ShellGrid:
    """Convert wavelength edges in kilometres to a shell grid."""
    edges = np.asarray(wavelength_edges_km, dtype=float).ravel()
    if edges.size < 2 or np.any(~np.isfinite(edges)) or np.any(edges <= 0):
        raise ValueError("at least two positive finite shell edges are required")
    if np.any(np.diff(edges) <= 0):
        raise ValueError("shell edges must be strictly increasing")
    lower, upper = edges[:-1].copy(), edges[1:].copy()
    return ShellGrid(
        wavelength_edges_km=edges,
        wavelength_min_km=lower,
        wavelength_max_km=upper,
        wavelength_center_km=np.sqrt(lower * upper),
        k_min_rad_per_m=2 * np.pi / (upper * 1000),
        k_max_rad_per_m=2 * np.pi / (lower * 1000),
    )


def shell_frequency_masks(
    n_points: int, spacing_m: float, shell_grid: ShellGrid
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return the along-track wavenumber axis and non-overlapping shell masks."""
    if n_points < 8:
        raise ValueError("at least eight along-track points are required")
    if not np.isfinite(spacing_m) or spacing_m <= 0:
        raise ValueError("spacing_m must be positive")
    k = np.fft.rfftfreq(int(n_points), d=float(spacing_m)) * (2 * np.pi)
    masks = [
        (k > 0) & (k >= lower) & (k < upper)
        for lower, upper in zip(shell_grid.k_min_rad_per_m, shell_grid.k_max_rad_per_m)
    ]
    return k, masks


def bandpass_shell_components(field: np.ndarray, masks: list[np.ndarray]) -> list[np.ndarray]:
    """Decompose a finite along-track by cross-track field into wavelength shells."""
    values = np.asarray(field, dtype=float)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("field must be a finite two-dimensional array")
    coefficients = np.fft.rfft(values - values.mean(axis=0, keepdims=True), axis=0)
    components = []
    for mask in masks:
        mask = np.asarray(mask, dtype=bool)
        if mask.size != coefficients.shape[0]:
            raise ValueError("shell mask length must match the rfft axis")
        selected = np.zeros_like(coefficients)
        selected[mask, :] = coefficients[mask, :]
        components.append(np.fft.irfft(selected, n=values.shape[0], axis=0).real)
    return components


def _gradient(field: np.ndarray, dx_m: float, dy_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Centered derivatives along cross-track (x) and along-track (y)."""
    gx = np.full_like(field, np.nan, dtype=float)
    gy = np.full_like(field, np.nan, dtype=float)
    gx[:, 1:-1] = (field[:, 2:] - field[:, :-2]) / (2 * dx_m)
    gy[1:-1, :] = (field[2:, :] - field[:-2, :]) / (2 * dy_m)
    return gx, gy


def shell_to_shell_matrix(
    u_cross: np.ndarray,
    u_along: np.ndarray,
    dx_m: float,
    dy_m: float,
    shell_grid: ShellGrid,
) -> dict[str, np.ndarray]:
    """Compute ``T(k,q) = -mean[u_k dot ((u_full dot grad) u_q)]``.

    Rows are receiver shells ``k`` and columns are donor shells ``q``.
    Positive values denote positive energy input to the receiver shell.
    Filtering is along track; the nonlinear advection uses both resolved
    horizontal derivatives on the finite swath patch.
    """
    uc, ua = np.asarray(u_cross, dtype=float), np.asarray(u_along, dtype=float)
    if uc.shape != ua.shape or uc.ndim != 2 or not np.all(np.isfinite(uc)) or not np.all(np.isfinite(ua)):
        raise ValueError("velocity components must be same-shape finite two-dimensional arrays")
    if min(dx_m, dy_m) <= 0 or not np.isfinite(dx_m + dy_m):
        raise ValueError("dx_m and dy_m must be positive and finite")
    k, masks = shell_frequency_masks(uc.shape[0], dy_m, shell_grid)
    uc_shells = bandpass_shell_components(uc, masks)
    ua_shells = bandpass_shell_components(ua, masks)
    mode_count = np.asarray([mask.sum() for mask in masks], dtype=int)
    matrix = np.full((shell_grid.n_shells, shell_grid.n_shells), np.nan)
    for donor in range(shell_grid.n_shells):
        if mode_count[donor] == 0:
            continue
        dc_dx, dc_dy = _gradient(uc_shells[donor], dx_m, dy_m)
        da_dx, da_dy = _gradient(ua_shells[donor], dx_m, dy_m)
        adv_cross = uc * dc_dx + ua * dc_dy
        adv_along = uc * da_dx + ua * da_dy
        for receiver in range(shell_grid.n_shells):
            if mode_count[receiver] == 0:
                continue
            work = uc_shells[receiver] * adv_cross + ua_shells[receiver] * adv_along
            matrix[receiver, donor] = -float(np.nanmean(work))
    return {"matrix": matrix, "mode_count": mode_count, "k_rad_per_m": k}


def antisymmetric_transfer(transfer_mean: np.ndarray) -> np.ndarray:
    """Return ``T_skew = (T - T.T) / 2`` after zeroing unsupported entries."""
    matrix = np.asarray(transfer_mean, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("transfer_mean must be square")
    finite = np.where(np.isfinite(matrix), matrix, 0.0)
    return (finite - finite.T) / 2


def cumulative_shell_flux(matrix: np.ndarray, shell_grid: ShellGrid) -> dict[str, np.ndarray]:
    """Split cumulative shell flux into factor-of-two local and nonlocal parts."""
    values = np.asarray(matrix, dtype=float)
    if values.shape != (shell_grid.n_shells, shell_grid.n_shells):
        raise ValueError("matrix shape must match shell_grid")
    values = np.where(np.isfinite(values), values, 0.0)
    wavelength = shell_grid.wavelength_center_km
    ratio = wavelength[None, :] / wavelength[:, None]
    local = (ratio >= LOCAL_RATIO_MIN) & (ratio <= LOCAL_RATIO_MAX)
    total_flux = np.full(wavelength.size, np.nan)
    local_flux = np.full(wavelength.size, np.nan)
    nonlocal_flux = np.full(wavelength.size, np.nan)
    for index, threshold in enumerate(wavelength):
        receiver = wavelength < threshold
        if not receiver.any():
            continue
        block = values[receiver, :]
        local_block = local[receiver, :]
        total_flux[index] = block.sum()
        local_flux[index] = (block * local_block).sum()
        nonlocal_flux[index] = (block * ~local_block).sum()
    return {
        "lambda_star_km": wavelength.copy(),
        "pi_total": total_flux,
        "pi_local": local_flux,
        "pi_nonlocal": nonlocal_flux,
    }
