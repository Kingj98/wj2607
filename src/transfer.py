"""Direct along-track transfer spectra."""

from __future__ import annotations

import numpy as np


def tukey_window_symmetric(n: int, alpha: float) -> np.ndarray:
    n = int(n)
    if n <= 0:
        return np.ones(0, dtype=float)
    if alpha <= 0:
        return np.ones(n, dtype=float)
    if alpha >= 1:
        return np.hanning(n)
    if n == 1:
        return np.ones(1, dtype=float)
    i = np.arange(n, dtype=float)
    width = int(np.floor(alpha * (n - 1) / 2))
    left = 0.5 * (1 + np.cos(np.pi * (-1 + 2 * i[: width + 1] / alpha / (n - 1))))
    middle = np.ones(n - 2 * width - 2, dtype=float)
    right_i = i[n - width - 1 :]
    right = 0.5 * (1 + np.cos(np.pi * (-2 / alpha + 1 + 2 * right_i / alpha / (n - 1))))
    return np.concatenate((left, middle, right))


def _detrend(y: np.ndarray) -> np.ndarray:
    x = np.arange(y.size, dtype=float)
    return y - np.polyval(np.polyfit(x, y, 1), x)


def _window(n: int, taper: str | None, correct_power: bool) -> tuple[np.ndarray | None, float]:
    name = "hann" if taper is None else str(taper).lower()
    if name in {"none", "rect", "rectangular", "off"}:
        return None, 1.0
    if name == "hann":
        w = np.hanning(n)
    elif name == "tukey":
        w = tukey_window_symmetric(n, 0.2)
    else:
        raise ValueError("taper must be one of: hann, tukey, none")
    power = float(np.mean(w * w)) if correct_power else 1.0
    if power <= 0:
        raise ValueError("window power must be positive")
    return w, power


def transfer_spectrum_T1(
    u_east: np.ndarray,
    u_north: np.ndarray,
    a_east: np.ndarray,
    a_north: np.ndarray,
    spacing_m: float,
    *,
    detrend: bool = True,
    taper: str | None = "hann",
    correct_window_power: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return positive wavenumbers and the direct one-dimensional transfer spectrum."""
    fields = [np.asarray(v, dtype=float).ravel() for v in (u_east, u_north, a_east, a_north)]
    if len({v.shape for v in fields}) != 1:
        raise ValueError("all velocity and acceleration series must have the same shape")
    if not np.isfinite(spacing_m) or spacing_m <= 0:
        raise ValueError("spacing_m must be positive")
    n = fields[0].size
    if n < 8:
        raise ValueError("at least eight samples are required")
    if detrend:
        fields = [_detrend(v) for v in fields]
    w, power = _window(n, taper, correct_window_power)
    if w is not None:
        fields = [v * w for v in fields]
    ue, un, ae, an = [np.fft.rfft(v) for v in fields]
    scale = spacing_m / (np.pi * n * power)
    t1 = -scale * np.real(np.conj(ue) * ae + np.conj(un) * an)
    k = np.fft.rfftfreq(n, d=spacing_m) * (2 * np.pi)
    return k, t1


def kinetic_energy_spectrum_1d(
    u_east: np.ndarray,
    u_north: np.ndarray,
    spacing_m: float,
    *,
    detrend: bool = True,
    taper: str | None = "hann",
    correct_window_power: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return positive wavenumbers and one-dimensional kinetic-energy spectrum."""
    u, v = [np.asarray(x, dtype=float).ravel() for x in (u_east, u_north)]
    if u.shape != v.shape or u.size < 8:
        raise ValueError("matching velocity series with at least eight samples are required")
    if not np.isfinite(spacing_m) or spacing_m <= 0:
        raise ValueError("spacing_m must be positive")
    if detrend:
        u, v = _detrend(u), _detrend(v)
    w, power = _window(u.size, taper, correct_window_power)
    if w is not None:
        u, v = u * w, v * w
    uh, vh = np.fft.rfft(u), np.fft.rfft(v)
    scale = spacing_m / (2 * np.pi * u.size * power)
    e1 = scale * (np.abs(uh) ** 2 + np.abs(vh) ** 2)
    k = np.fft.rfftfreq(u.size, d=spacing_m) * (2 * np.pi)
    return k, e1


def tail_integral_from_tk(k_rad_per_m: np.ndarray, transfer: np.ndarray) -> np.ndarray:
    """Return `Pi_K(k) = integral_k^kmax T(q)dq`."""
    k = np.asarray(k_rad_per_m, dtype=float).ravel()
    t = np.asarray(transfer, dtype=float).ravel()
    if k.shape != t.shape:
        raise ValueError("k and transfer must have the same shape")
    out = np.full_like(t, np.nan)
    valid = np.isfinite(k) & np.isfinite(t)
    if valid.sum() < 2:
        return out
    index = np.flatnonzero(valid)
    order = index[np.argsort(k[index])]
    ks, ts = k[order], t[order]
    if np.any(np.diff(ks) <= 0):
        raise ValueError("wavenumbers must be strictly increasing")
    segment = 0.5 * (ts[:-1] + ts[1:]) * np.diff(ks)
    out[order] = np.concatenate((np.cumsum(segment[::-1])[::-1], [0.0]))
    return out


def parseval_residual(
    u_east: np.ndarray,
    u_north: np.ndarray,
    a_east: np.ndarray,
    a_north: np.ndarray,
    spacing_m: float,
) -> float:
    """Return `mean(u dot a) + integral(T1 dk)` for the rectangular-window audit."""
    values = [_detrend(np.asarray(v, dtype=float).ravel()) for v in (u_east, u_north, a_east, a_north)]
    k, t1 = transfer_spectrum_T1(*values, spacing_m, detrend=False, taper="none")
    integrate = getattr(np, "trapezoid", np.trapz)
    return float(np.mean(values[0] * values[2] + values[1] * values[3]) + integrate(t1, k))
