"""二维圆盘核 SFS：`Pi_l = - tau_ij S_ij`（默认单位质量，见 `include_rho0`）。"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import ndimage

from sfs.common import central_diff_strict, finite_stats, sfs_mask


@lru_cache(maxsize=64)
def build_top_hat_kernel_disk(ell_km: float, dx_km: float, dy_km: float) -> np.ndarray:
    radius_km = float(ell_km) / 2.0
    if radius_km <= 0.0:
        raise ValueError(f"invalid ell_km: {ell_km}")
    x = np.arange(-radius_km, radius_km + dx_km * 0.5, dx_km, dtype=float)
    y = np.arange(-radius_km, radius_km + dy_km * 0.5, dy_km, dtype=float)
    xg, yg = np.meshgrid(x, y, indexing="ij")
    mask = (xg**2 + yg**2) <= (radius_km**2 + 1.0e-12)
    kernel = mask.astype(float)
    kernel_sum = float(np.sum(kernel))
    if kernel_sum <= 0.0:
        raise ValueError("kernel support is empty")
    return kernel / kernel_sum


def strict_filter2d(field: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    values = np.asarray(field, dtype=float)
    valid = np.isfinite(values)
    filled = np.where(valid, values, 0.0)
    filtered = ndimage.convolve(filled, kernel, mode="constant", cval=0.0)
    support = ndimage.convolve(valid.astype(float), kernel, mode="constant", cval=0.0)
    out = np.full(values.shape, np.nan, dtype=float)
    full_support = np.isclose(support, 1.0, atol=1.0e-12)
    out[full_support] = filtered[full_support]
    return out


def strain_rate_tensor_bar(
    ubar: np.ndarray, vbar: np.ndarray, dx_m: float, dy_m: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    数组布局 `(y, x)`：`axis=0` 为 y、`axis=1` 为 x。
    速度分量：`u` 对应 x 向、`v` 对应 y 向（与输入快照 `u,v` 约定一致）。
    S_xx = ∂u/∂x, S_yy = ∂v/∂y, S_xy = 0.5*(∂u/∂y + ∂v/∂x)。
    """
    s_xx = central_diff_strict(ubar, dx_m, axis=1)
    s_yy = central_diff_strict(vbar, dy_m, axis=0)
    dudy = central_diff_strict(ubar, dy_m, axis=0)
    dvdx = central_diff_strict(vbar, dx_m, axis=1)
    s_xy = 0.5 * (dudy + dvdx)
    return s_xx, s_yy, s_xy


def compute_pi_field_2d(
    u: np.ndarray,
    v: np.ndarray,
    ell_km: float,
    dx_km: float,
    dy_km: float,
    *,
    include_rho0: bool = False,
    rho0: float = 1025.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kernel = build_top_hat_kernel_disk(float(ell_km), float(dx_km), float(dy_km))
    ubar = strict_filter2d(u, kernel)
    vbar = strict_filter2d(v, kernel)
    uu_bar = strict_filter2d(u**2, kernel)
    uv_bar = strict_filter2d(u * v, kernel)
    vv_bar = strict_filter2d(v**2, kernel)

    tau_xx = uu_bar - (ubar**2)
    tau_xy = uv_bar - (ubar * vbar)
    tau_yy = vv_bar - (vbar**2)

    dx_m = dx_km * 1000.0
    dy_m = dy_km * 1000.0
    s_xx, s_yy, s_xy = strain_rate_tensor_bar(ubar, vbar, dx_m, dy_m)
    pi_field = -(tau_xx * s_xx + 2.0 * tau_xy * s_xy + tau_yy * s_yy)
    if include_rho0:
        pi_field = pi_field * float(rho0)

    mask_sfs = sfs_mask(u.shape[0], u.shape[1], float(ell_km), float(dx_km), float(dy_km))
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


def compute_sfs_2d(
    u: np.ndarray,
    v: np.ndarray,
    ell_km: float,
    dx_km: float,
    dy_km: float,
    *,
    include_rho0: bool = False,
    rho0: float = 1025.0,
) -> dict[str, float | int]:
    pi_field, mask_sfs, _ = compute_pi_field_2d(
        u, v, ell_km, dx_km, dy_km, include_rho0=include_rho0, rho0=rho0
    )
    return summarize_pi_on_mask(pi_field, mask_sfs)


def compute_sfs_2d_with_field(
    u: np.ndarray,
    v: np.ndarray,
    ell_km: float,
    dx_km: float,
    dy_km: float,
    *,
    include_rho0: bool = False,
    rho0: float = 1025.0,
) -> tuple[dict[str, float | int], np.ndarray, np.ndarray]:
    pi_field, mask_sfs, _ = compute_pi_field_2d(
        u, v, ell_km, dx_km, dy_km, include_rho0=include_rho0, rho0=rho0
    )
    summary = summarize_pi_on_mask(pi_field, mask_sfs)
    return summary, pi_field, mask_sfs
