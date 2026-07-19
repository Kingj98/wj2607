"""SFS 共享工具：差分、掩膜、统计、science profile 与输出契约。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
import yaml


def load_config(config_path: Path) -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_science_contract(cfg: dict[str, Any]) -> dict[str, Any]:
    route_name = str(cfg.get("route_name", ""))
    sfs_cfg = dict(cfg.get("sfs", {}))
    science_cfg = dict(cfg.get("science_outputs", {}))
    is_wang_1d = route_name == "sfs_1d"
    science_profile = str(
        cfg.get("science_profile")
        or science_cfg.get("science_profile")
        or ("wang2025_ke_v1" if is_wang_1d else "generic_2d_compare_v1")
    )
    velocity_source = str(
        science_cfg.get("velocity_source")
        or sfs_cfg.get("velocity_source")
        or ("product_uv_native_cross_along" if is_wang_1d else "eta_only")
    )
    comparison_family = str(
        science_cfg.get("comparison_family")
        or ("wang2025_swot_alongtrack" if is_wang_1d else "future_2d_fft" if route_name == "sfs_2d" else "generic_compare")
    )
    return {
        "route_name": route_name,
        "science_profile": science_profile,
        "velocity_source": velocity_source,
        "comparison_family": comparison_family,
        "domain_type": str(
            science_cfg.get("domain_type")
            or ("swot_native_swath" if is_wang_1d else "surface_snapshot_2d")
        ),
        "split_swath_sides": bool(science_cfg.get("split_swath_sides", is_wang_1d)),
        "emit_science_bundle": bool(science_cfg.get("emit_science_bundle", route_name in {"sfs_1d", "sfs_2d"})),
        "emit_map_bundle": bool(science_cfg.get("emit_map_bundle", route_name == "sfs_2d")),
        "wing": str(science_cfg.get("wing", "both")),
        "route_description": str(
            science_cfg.get("route_description")
            or (
                "Wang 2025-style along-track SFS route on SWOT swath grid (KE adaptation)"
                if is_wang_1d
                else "2D compare-route SFS (for future 2D FFT comparison)"
            )
        ),
        "reference_paper": str(
            science_cfg.get("reference_paper")
            or ("Wang et al. 2025, JPO, doi:10.1175/JPO-D-24-0134.1" if is_wang_1d else "")
        ),
        "filter_geometry": str(
            science_cfg.get("filter_geometry")
            or ("1D top-hat along axis0 (along-track)" if is_wang_1d else "2D disk top-hat on snapshot grid")
        ),
        "gradient_geometry": str(
            science_cfg.get("gradient_geometry")
            or ("2D strain-rate tensor on native swath grid" if is_wang_1d else "2D strain-rate tensor on snapshot grid")
        ),
        "approximation_scope": str(
            science_cfg.get("approximation_scope")
            or (
                "1D along-track coarse-graining on SWOT swath grid; retains 2D velocity gradients on the native swath patch"
                if is_wang_1d
                else "Full 2D coarse-graining on 2D surface snapshots"
            )
        ),
        "sign_convention": "Pi_l > 0 => downscale; Pi_l < 0 => upscale",
    }


def central_diff_strict(field: np.ndarray, step_m: float, axis: int) -> np.ndarray:
    """中心差分；三邻域均有限才写值。"""
    values = np.asarray(field, dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)
    if axis == 0:
        center = values[1:-1, :]
        prev_vals = values[:-2, :]
        next_vals = values[2:, :]
        valid = np.isfinite(center) & np.isfinite(prev_vals) & np.isfinite(next_vals)
        out[1:-1, :][valid] = (next_vals[valid] - prev_vals[valid]) / (2.0 * step_m)
        return out
    if axis == 1:
        center = values[:, 1:-1]
        prev_vals = values[:, :-2]
        next_vals = values[:, 2:]
        valid = np.isfinite(center) & np.isfinite(prev_vals) & np.isfinite(next_vals)
        out[:, 1:-1][valid] = (next_vals[valid] - prev_vals[valid]) / (2.0 * step_m)
        return out
    raise ValueError(f"invalid axis: {axis}")


def sfs_mask(ny: int, nx: int, ell_km: float, dx_km: float, dy_km: float) -> np.ndarray:
    """内域掩膜：距边界至少 ell/2 + max(dx,dy)（km）。"""
    i = np.arange(ny, dtype=float)[:, None]
    j = np.arange(nx, dtype=float)[None, :]
    dist_top = i * dy_km
    dist_bottom = (ny - 1 - i) * dy_km
    dist_left = j * dx_km
    dist_right = (nx - 1 - j) * dx_km
    boundary_dist = np.minimum(np.minimum(dist_top, dist_bottom), np.minimum(dist_left, dist_right))
    margin_km = ell_km / 2.0 + max(dx_km, dy_km)
    return boundary_dist >= (margin_km - 1.0e-12)


def sfs_mask_along(ny: int, nx: int, ell_km: float, dy_km: float, *, cross_margin_cells: int = 1) -> np.ndarray:
    """沿轨主约束掩膜：沿轨使用 ell 边距，横轨仅保留最小像元边距。"""
    i = np.arange(ny, dtype=float)[:, None]
    j = np.arange(nx, dtype=float)[None, :]
    along_margin_km = ell_km / 2.0 + dy_km
    dist_top = i * dy_km
    dist_bottom = (ny - 1 - i) * dy_km
    along_ok = np.minimum(dist_top, dist_bottom) >= (along_margin_km - 1.0e-12)

    cross_margin_cells = max(int(cross_margin_cells), 0)
    cross_left = j
    cross_right = nx - 1 - j
    cross_ok = np.minimum(cross_left, cross_right) >= cross_margin_cells
    return along_ok & cross_ok


def finite_stats(values: np.ndarray) -> dict[str, float | int]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            "n_valid": 0,
            "mean": np.nan,
            "median": np.nan,
            "iqr": np.nan,
            "frac_pos": np.nan,
            "frac_neg": np.nan,
            "abs_mean": np.nan,
        }
    q25, q75 = np.nanpercentile(finite, [25.0, 75.0])
    return {
        "n_valid": int(finite.size),
        "mean": float(np.nanmean(finite)),
        "median": float(np.nanmedian(finite)),
        "iqr": float(q75 - q25),
        "frac_pos": float(np.mean(finite > 0)),
        "frac_neg": float(np.mean(finite < 0)),
        "abs_mean": float(np.nanmean(np.abs(finite))),
    }


def _safe_sem(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    if arr.size == 1:
        return 0.0
    return float(np.nanstd(arr, ddof=1) / np.sqrt(arr.size))


def aggregate_curve_frame(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    contract: dict[str, Any],
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    grouped = (
        df.groupby(group_cols, dropna=False)
        .agg(
            n_samples=("pi_mean", "count"),
            mean_pi_mean=("pi_mean", "mean"),
            mean_pi_median=("pi_median", "mean"),
            mean_abs_pi=("abs_pi_mean", "mean"),
            mean_frac_pos=("frac_pos", "mean"),
            mean_frac_neg=("frac_neg", "mean"),
            mean_coverage_inner=("coverage_inner", "mean"),
        )
        .reset_index()
    )
    sem = df.groupby(group_cols, dropna=False)["pi_mean"].apply(lambda s: _safe_sem(s.to_numpy(dtype=float)))
    grouped["pi_std_error"] = sem.to_numpy(dtype=float)
    grouped["route_name"] = str(contract.get("route_name", ""))
    grouped["science_profile"] = str(contract.get("science_profile", ""))
    grouped["velocity_source"] = str(contract.get("velocity_source", ""))
    grouped["comparison_family"] = str(contract.get("comparison_family", ""))
    grouped["domain_type"] = str(contract.get("domain_type", ""))
    grouped["sign_convention"] = str(contract.get("sign_convention", ""))

    front_cols = [c for c in ["route_name", "science_profile", "velocity_source", "comparison_family", "domain_type"] if c in grouped.columns]
    id_cols = [c for c in group_cols if c in grouped.columns]
    metric_cols = [c for c in grouped.columns if c not in front_cols + id_cols]
    return grouped[front_cols + id_cols + metric_cols]


def detect_transition_frame(df: pd.DataFrame, *, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[*group_cols, "transition_status", "transition_ell_km"])

    rows: list[dict[str, Any]] = []
    for keys, sub in df.groupby(group_cols, dropna=False):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        row = {name: value for name, value in zip(group_cols, key_tuple)}
        ordered = sub.sort_values("ell_km")
        ell = np.asarray(ordered["ell_km"], dtype=float)
        pi = np.asarray(ordered["mean_pi_mean"], dtype=float)
        finite = np.isfinite(ell) & np.isfinite(pi)
        ell = ell[finite]
        pi = pi[finite]
        if ell.size == 0:
            row["transition_status"] = "no_finite_curve"
            row["transition_ell_km"] = np.nan
            rows.append(row)
            continue
        exact_zero = np.where(np.isclose(pi, 0.0, atol=1.0e-15))[0]
        if exact_zero.size:
            row["transition_status"] = "exact_zero"
            row["transition_ell_km"] = float(ell[int(exact_zero[0])])
            rows.append(row)
            continue
        signs = np.sign(pi)
        hit = None
        for i in range(signs.size - 1):
            if signs[i] == 0 or signs[i + 1] == 0:
                continue
            if signs[i] != signs[i + 1]:
                hit = i
                break
        if hit is None:
            row["transition_status"] = "no_sign_change"
            row["transition_ell_km"] = np.nan
            rows.append(row)
            continue
        x0, x1 = float(ell[hit]), float(ell[hit + 1])
        y0, y1 = float(pi[hit]), float(pi[hit + 1])
        frac = 0.5 if np.isclose(y1, y0) else (-y0 / (y1 - y0))
        frac = float(np.clip(frac, 0.0, 1.0))
        row["transition_status"] = "interpolated_sign_change"
        row["transition_ell_km"] = float(x0 + frac * (x1 - x0))
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_acceptance(
    annual_df: pd.DataFrame,
    *,
    acceptance: dict[str, Any],
    ell_list_km: list[float],
) -> tuple[str, list[str], dict[str, Any]]:
    expected = sorted({round(float(v), 8) for v in ell_list_km})
    details = {
        "expected_ell_km": expected,
        "coverage_mean_min": acceptance.get("coverage_mean_min"),
        "require_all_ell": bool(acceptance.get("require_all_ell", False)),
        "fail_on_missing_ell": bool(acceptance.get("fail_on_missing_ell", False)),
    }
    if annual_df.empty:
        return "FAIL", ["annual curve is empty"], details

    rows = annual_df.copy()
    if "group_name" in rows.columns:
        merged_rows = rows[rows["group_name"].astype(str) == "merged"]
        if not merged_rows.empty:
            rows = merged_rows
    rows["ell_key"] = rows["ell_km"].astype(float).round(8)
    present = sorted(set(rows["ell_key"].tolist()))

    missing = [v for v in expected if v not in present]
    low_coverage: list[float] = []
    threshold = acceptance.get("coverage_mean_min")
    if threshold is not None:
        threshold = float(threshold)
        for _, r in rows.iterrows():
            cov = float(r.get("mean_coverage_inner", np.nan))
            if np.isfinite(cov) and cov < threshold:
                low_coverage.append(float(r["ell_km"]))

    issues: list[str] = []
    if missing:
        issues.append("missing ell: " + ", ".join(f"{v:g}" for v in missing))
    if low_coverage:
        issues.append(
            "coverage below threshold: "
            + ", ".join(f"{v:g}" for v in sorted(set(low_coverage)))
            + f" (threshold={float(threshold):.3f})"
        )

    if not issues:
        return "PASS", [], details
    if missing and bool(acceptance.get("fail_on_missing_ell", False)):
        return "FAIL", issues, details
    if low_coverage:
        return "FAIL", issues, details
    return "BORDERLINE", issues, details


def annual_mean_eta(raw_surface_dir: Path, source_files: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eta_sum: np.ndarray | None = None
    eta_count: np.ndarray | None = None
    xc_ref: np.ndarray | None = None
    yc_ref: np.ndarray | None = None
    for name in source_files:
        raw_path = raw_surface_dir / name
        if not raw_path.exists():
            continue
        ds = xr.open_dataset(raw_path)
        eta = np.asarray(ds["Eta"].mean(dim="time", skipna=True).values, dtype=float)
        xc = np.asarray(ds["XC"].values, dtype=float)
        yc = np.asarray(ds["YC"].values, dtype=float)
        ds.close()
        if eta_sum is None:
            eta_sum = np.zeros_like(eta, dtype=float)
            eta_count = np.zeros_like(eta, dtype=float)
            xc_ref = xc
            yc_ref = yc
        valid = np.isfinite(eta)
        eta_sum[valid] += eta[valid]
        eta_count[valid] += 1.0
    if eta_sum is None or eta_count is None or xc_ref is None or yc_ref is None:
        raise SystemExit(f"[ERROR] failed to build annual mean Eta from {raw_surface_dir}")
    eta_mean = np.full_like(eta_sum, np.nan, dtype=float)
    good = eta_count > 0
    eta_mean[good] = eta_sum[good] / eta_count[good]
    return eta_mean, xc_ref, yc_ref


def resolve_run_paths(paths: dict[str, Any]) -> tuple[Path, Path, Path, Path]:
    """返回 (run_root, sfs_inputs_dir, figures_dir, summary_dir)。"""
    if paths.get("run_root"):
        run_root = Path(paths["run_root"])
        sfs_inputs_dir = run_root / "artifacts" / "sfs_inputs"
        figures_dir = run_root / "artifacts" / "figures"
        summary_dir = run_root / "summary"
        return run_root, sfs_inputs_dir, figures_dir, summary_dir
    sfs_inputs_dir = Path(paths["sfs_output"])
    figures_dir = Path(paths.get("figures_output", sfs_inputs_dir.parent / "artifacts" / "figures"))
    summary_dir = Path(paths.get("summary_output", sfs_inputs_dir.parent / "summary"))
    run_root = summary_dir.parent
    return run_root, sfs_inputs_dir, figures_dir, summary_dir


def write_run_manifest(
    path: Path,
    *,
    route_name: str,
    filter_mode: str,
    config_path: Path,
    snapshot_base: Path,
    sfs_inputs_dir: Path,
    n_files: int,
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "route_name": route_name,
        "filter_mode": filter_mode,
        "config_path": str(config_path.resolve()),
        "snapshot_base": str(snapshot_base.resolve()),
        "sfs_inputs_output": str(sfs_inputs_dir.resolve()),
        "n_snapshot_files": n_files,
    }
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_conclusion_report_sfs(
    path: Path,
    *,
    route_name: str,
    filter_mode: str,
    science_profile: str,
    route_description: str,
    velocity_source: str,
    domain_type: str,
    comparison_family: str,
    reference_paper: str,
    filter_geometry: str,
    gradient_geometry: str,
    approximation_scope: str,
    n_snapshots: int,
    rows_csv: Path,
    annual_csv: Path,
    map_nc: Path | None,
    verdict: str = "PASS",
    map_status: str = "available",
    issues: list[str] | None = None,
    science_outputs: dict[str, Path] | None = None,
) -> None:
    lines = [
        f"# SFS 多尺度（{route_name}，{filter_mode}）",
        "",
        f"- 路线：`{route_name}`；滤波：`{filter_mode}`。",
        f"- scientific profile：`{science_profile}`。",
        f"- 路线说明：{route_description}",
        f"- 速度来源：`{velocity_source}`；domain_type：`{domain_type}`。",
        f"- comparison family：`{comparison_family}`。",
        f"- reference：{reference_paper or 'N/A'}",
        f"- filter geometry：{filter_geometry}",
        f"- gradient geometry：{gradient_geometry}",
        f"- approximation scope：{approximation_scope}",
        f"- 处理快照数：`{n_snapshots}`。",
        f"- 最终裁决：`{verdict}`。",
        f"- annual map 状态：`{map_status}`。",
        "",
    ]
    if issues:
        lines.extend(["## 裁决说明", ""])
        for item in issues:
            lines.append(f"- {item}")
        lines.append("")

    lines.extend([
        "## 关键覆盖统计（按尺度 annual 聚合）",
        "",
    ])

    try:
        adf = pd.read_csv(annual_csv)
        if not adf.empty and all(c in adf.columns for c in ["ell_km", "mean_coverage_inner"]):
            for _, r in adf.sort_values("ell_km").iterrows():
                lines.append(f"- ell={float(r['ell_km']):g} km: mean_coverage_inner={float(r['mean_coverage_inner']):.3f}")
        else:
            lines.append("- annual.csv 缺少覆盖统计字段，需检查上游输出。")
    except Exception as exc:  # pragma: no cover
        lines.append(f"- 读取 annual.csv 失败：`{exc}`")

    lines.extend([
        "",
        "## 主要产物",
        "",
        f"- 逐快照行表：`{rows_csv.resolve()}`",
        f"- 按尺度 annual：`{annual_csv.resolve()}`",
    ])
    if map_nc is not None:
        lines.append(f"- annual 空间场 NetCDF：`{map_nc.resolve()}`")
    if science_outputs:
        lines.extend(["", "## Science Bundle", ""])
        for name, file_path in science_outputs.items():
            lines.append(f"- {name}：`{file_path.resolve()}`")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
