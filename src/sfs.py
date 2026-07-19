"""One- and two-dimensional subfilter-scale fluxes."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import ndimage


def _central_diff(field: np.ndarray, step_m: float, axis: int) -> np.ndarray:
    out = np.full(field.shape, np.nan, dtype=float)
    if axis == 0:
        center, before, after = field[1:-1], field[:-2], field[2:]
        valid = np.isfinite(center) & np.isfinite(before) & np.isfinite(after)
        out[1:-1][valid] = (after[valid] - before[valid]) / (2 * step_m)
    elif axis == 1:
        center, before, after = field[:, 1:-1], field[:, :-2], field[:, 2:]
        valid = np.isfinite(center) & np.isfinite(before) & np.isfinite(after)
        out[:, 1:-1][valid] = (after[valid] - before[valid]) / (2 * step_m)
    else:
        raise ValueError("axis must be 0 or 1")
    return out


def _strain(u: np.ndarray, v: np.ndarray, dx_m: float, dy_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sxx = _central_diff(u, dx_m, 1)
    syy = _central_diff(v, dy_m, 0)
    sxy = 0.5 * (_central_diff(u, dy_m, 0) + _central_diff(v, dx_m, 1))
    return sxx, syy, sxy


def _strict_filter(field: np.ndarray, kernel: np.ndarray, axis: int | None = None) -> np.ndarray:
    values = np.asarray(field, dtype=float)
    valid = np.isfinite(values)
    if axis is None:
        filtered = ndimage.convolve(np.where(valid, values, 0.0), kernel, mode="constant", cval=0.0)
        support = ndimage.convolve(valid.astype(float), kernel, mode="constant", cval=0.0)
    else:
        filtered = ndimage.convolve1d(np.where(valid, values, 0.0), kernel, axis=axis, mode="constant", cval=0.0)
        support = ndimage.convolve1d(valid.astype(float), kernel, axis=axis, mode="constant", cval=0.0)
    return np.where(np.isclose(support, 1.0, atol=1e-12), filtered, np.nan)


@lru_cache(maxsize=64)
def top_hat_along(ell_km: float, dy_km: float) -> np.ndarray:
    if ell_km <= 0 or dy_km <= 0:
        raise ValueError("ell_km and dy_km must be positive")
    half = ell_km / 2
    coordinate = np.arange(-half, half + dy_km * 0.5, dy_km)
    kernel = (np.abs(coordinate) <= half + 1e-12).astype(float)
    return kernel / kernel.sum()


@lru_cache(maxsize=64)
def top_hat_disk(ell_km: float, dx_km: float, dy_km: float) -> np.ndarray:
    if ell_km <= 0 or dx_km <= 0 or dy_km <= 0:
        raise ValueError("ell_km, dx_km, and dy_km must be positive")
    radius = ell_km / 2
    x = np.arange(-radius, radius + dx_km * 0.5, dx_km)
    y = np.arange(-radius, radius + dy_km * 0.5, dy_km)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    kernel = (xx * xx + yy * yy <= radius * radius + 1e-12).astype(float)
    return kernel / kernel.sum()


def _compute(u: np.ndarray, v: np.ndarray, kernel: np.ndarray, dx_km: float, dy_km: float, axis: int | None) -> tuple[np.ndarray, np.ndarray]:
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    if u.shape != v.shape or u.ndim != 2:
        raise ValueError("u and v must be matching two-dimensional arrays")
    filt = lambda field: _strict_filter(field, kernel, axis)
    ubar, vbar = filt(u), filt(v)
    txx = filt(u * u) - ubar * ubar
    txy = filt(u * v) - ubar * vbar
    tyy = filt(v * v) - vbar * vbar
    sxx, syy, sxy = _strain(ubar, vbar, dx_km * 1000, dy_km * 1000)
    return -(txx * sxx + 2 * txy * sxy + tyy * syy), np.stack((txx, txy, tyy))


def _mask_1d(ny: int, nx: int, ell_km: float, dy_km: float) -> np.ndarray:
    i, j = np.arange(ny)[:, None], np.arange(nx)[None, :]
    along = np.minimum(i * dy_km, (ny - 1 - i) * dy_km) >= ell_km / 2 + dy_km - 1e-12
    return along & (np.minimum(j, nx - 1 - j) >= 1)


def _mask_2d(ny: int, nx: int, ell_km: float, dx_km: float, dy_km: float) -> np.ndarray:
    i, j = np.arange(ny)[:, None], np.arange(nx)[None, :]
    y_distance = np.minimum(i * dy_km, (ny - 1 - i) * dy_km)
    x_distance = np.minimum(j * dx_km, (nx - 1 - j) * dx_km)
    distance = np.minimum(y_distance, x_distance)
    return distance >= ell_km / 2 + max(dx_km, dy_km) - 1e-12


def compute_pi_field_1d_along(u: np.ndarray, v: np.ndarray, ell_km: float, dx_km: float, dy_km: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return `Pi_l`, valid mask, and stress components for 1D along-track filtering."""
    pi, tau = _compute(u, v, top_hat_along(ell_km, dy_km), dx_km, dy_km, axis=0)
    return pi, _mask_1d(u.shape[0], u.shape[1], ell_km, dy_km), tau


def compute_pi_field_2d(u: np.ndarray, v: np.ndarray, ell_km: float, dx_km: float, dy_km: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return `Pi_l`, valid mask, and stress components for 2D disk filtering."""
    pi, tau = _compute(u, v, top_hat_disk(ell_km, dx_km, dy_km), dx_km, dy_km, axis=None)
    return pi, _mask_2d(u.shape[0], u.shape[1], ell_km, dx_km, dy_km), tau


def summarize(pi: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(pi, dtype=float)[np.asarray(mask, dtype=bool)]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n_valid": 0, "mean": np.nan, "median": np.nan, "fraction_positive": np.nan}
    return {
        "n_valid": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "fraction_positive": float(np.mean(values > 0)),
    }
