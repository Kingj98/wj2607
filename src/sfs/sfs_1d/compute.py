"""沿轨（`axis=0`）一维 top-hat 粗粒化 + 二维应变率；`Pi_l = - tau_ij S_ij`。"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import ndimage

from sfs.common import finite_stats, sfs_mask_along
from sfs.sfs_2d.compute import strain_rate_tensor_bar


@lru_cache(maxsize=64)
def build_top_hat_kernel_1d_along(ell_km: float, dy_km: float) -> np.ndarray:
    """沿轨一维 top-hat，支撑总长约 `ell_km`（km），步长 `dy_km`。"""
    half = float(ell_km) / 2.0
    if half <= 0.0:
        raise ValueError(f"invalid ell_km: {ell_km}")
    r = np.arange(-half, half + dy_km * 0.5, dy_km, dtype=float)
    mask = np.abs(r) <= half + 1.0e-12
    k = mask.astype(float)
    s = float(np.sum(k))
    if s <= 0.0:
        raise ValueError("1d kernel support is empty")
    return k / s


def strict_filter1d(field: np.ndarray, kernel_1d: np.ndarray, axis: int) -> np.ndarray:
    values = np.asarray(field, dtype=float)
    valid = np.isfinite(values)
    filled = np.where(valid, values, 0.0)
    k = np.asarray(kernel_1d, dtype=float).ravel()
    filtered = ndimage.convolve1d(filled, k, axis=axis, mode="constant", cval=0.0)
    support = ndimage.convolve1d(valid.astype(float), k, axis=axis, mode="constant", cval=0.0)
    out = np.full(values.shape, np.nan, dtype=float)
    full_support = np.isclose(support, 1.0, atol=1.0e-12)
    out[full_support] = filtered[full_support]
    return out


def compute_pi_field_1d_along(
    u: np.ndarray,
    v: np.ndarray,
    ell_km: float,
    dx_km: float,
    dy_km: float,
    *,
    include_rho0: bool = False,
    rho0: float = 1025.0,
    along_axis: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if along_axis != 0:
        raise ValueError("sfs_1d 当前仅实现 along_axis=0（数组第一维为沿轨）")
    kernel_1d = build_top_hat_kernel_1d_along(float(ell_km), float(dy_km))
    ubar = strict_filter1d(u, kernel_1d, axis=along_axis)
    vbar = strict_filter1d(v, kernel_1d, axis=along_axis)
    uu_bar = strict_filter1d(u**2, kernel_1d, axis=along_axis)
    uv_bar = strict_filter1d(u * v, kernel_1d, axis=along_axis)
    vv_bar = strict_filter1d(v**2, kernel_1d, axis=along_axis)

    tau_xx = uu_bar - (ubar**2)
    tau_xy = uv_bar - (ubar * vbar)
    tau_yy = vv_bar - (vbar**2)

    dx_m = dx_km * 1000.0
    dy_m = dy_km * 1000.0
    s_xx, s_yy, s_xy = strain_rate_tensor_bar(ubar, vbar, dx_m, dy_m)
    pi_field = -(tau_xx * s_xx + 2.0 * tau_xy * s_xy + tau_yy * s_yy)
    if include_rho0:
        pi_field = pi_field * float(rho0)

    # sfs_1d: 沿轨主约束，横轨仅要求最小像元边距，避免窄刈幅被二维边界全清空
    mask_sfs = sfs_mask_along(u.shape[0], u.shape[1], float(ell_km), float(dy_km), cross_margin_cells=1)
    return pi_field, mask_sfs, np.stack([tau_xx, tau_xy, tau_yy], axis=0)


def summarize_pi_on_mask(pi_field: np.ndarray, mask_sfs: np.ndarray) -> dict[str, float | int]:
    inner_valid = mask_sfs & np.isfinite(pi_field)
    n_inner_total = int(np.sum(mask_sfs))
    stats = finite_stats(pi_field[inner_valid])
    n_valid_inner = int(stats["n_valid"])
    coverage_inner = float(n_valid_inner / n_inner_total) if n_inner_total > 0 else 0.0
    return {
        "pi_mean": float(stats["mean"]),
        "pi_median": float(stats["median"]),
        "pi_iqr": float(stats["iqr"]),
        "frac_pos": float(stats["frac_pos"]),
        "frac_neg": float(stats["frac_neg"]),
        "abs_pi_mean": float(stats["abs_mean"]),
        "n_valid_inner": n_valid_inner,
        "n_inner_total": n_inner_total,
        "coverage_inner": coverage_inner,
    }


def compute_sfs_1d_with_field(
    u: np.ndarray,
    v: np.ndarray,
    ell_km: float,
    dx_km: float,
    dy_km: float,
    *,
    include_rho0: bool = False,
    rho0: float = 1025.0,
    along_axis: int = 0,
) -> tuple[dict[str, float | int], np.ndarray, np.ndarray]:
    pi_field, mask_sfs, _ = compute_pi_field_1d_along(
        u,
        v,
        ell_km,
        dx_km,
        dy_km,
        include_rho0=include_rho0,
        rho0=rho0,
        along_axis=along_axis,
    )
    summary = summarize_pi_on_mask(pi_field, mask_sfs)
    return summary, pi_field, mask_sfs
