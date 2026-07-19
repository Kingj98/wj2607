"""Along-track direct 1D transfer spectrum T_1(k)."""

from __future__ import annotations

import numpy as np

_trapz = getattr(np, "trapezoid", np.trapz)


def tukey_window_symmetric(M: int, alpha: float) -> np.ndarray:
    """
    Symmetric Tukey (tapered cosine) window, matching ``scipy.signal.windows.tukey(..., sym=True)``.

    ``alpha`` is the fraction of the (M-1) span occupied by each cosine taper region.
    Edge cases follow SciPy: ``alpha<=0`` -> rectangular ones; ``alpha>=1`` -> Hann.
    """
    M = int(M)
    if M <= 0:
        return np.ones(0, dtype=float)
    a = float(alpha)
    if a <= 0.0:
        return np.ones(M, dtype=float)
    if a >= 1.0:
        return np.asarray(np.hanning(M), dtype=float)
    if M == 1:
        return np.ones(1, dtype=float)
    n = np.arange(0, M, dtype=float)
    width = int(np.floor(a * (M - 1) / 2.0))
    n1 = n[0 : width + 1]
    n2 = n[width + 1 : M - width - 1]
    n3 = n[M - width - 1 :]
    w1 = 0.5 * (1.0 + np.cos(np.pi * (-1.0 + 2.0 * n1 / a / (M - 1))))
    w2 = np.ones(n2.shape, dtype=float)
    w3 = 0.5 * (1.0 + np.cos(np.pi * (-2.0 / a + 1.0 + 2.0 * n3 / a / (M - 1))))
    return np.concatenate((w1, w2, w3)).astype(float, copy=False)


def build_along_track_fft_window(
    n: int,
    *,
    apply_hann: bool = True,
    apply_window_power_correction: bool = True,
    along_track_taper: str | None = None,
    tukey_alpha: float = 0.2,
) -> tuple[np.ndarray | None, float]:
    """
    Weights ``w`` for along-track FFT (``None`` = rectangular / no taper) and divisor ``<w^2>`` when
    ``apply_window_power_correction`` is enabled (otherwise divisor is ``1.0``).
    """
    if along_track_taper is not None:
        t = str(along_track_taper).strip().lower()
        if t in ("none", "off", "rect", "rectangular"):
            eff = "none"
        elif t == "hann":
            eff = "hann"
        elif t == "tukey":
            eff = "tukey"
        else:
            raise ValueError(
                "along_track_taper must be None or one of {'none','hann','tukey'} "
                f"(aliases: off, rect, rectangular); got {along_track_taper!r}"
            )
    else:
        eff = "hann" if apply_hann else "none"

    if eff == "none":
        return None, 1.0
    if eff == "hann":
        w = np.asarray(np.hanning(int(n)), dtype=float)
    else:
        w = tukey_window_symmetric(int(n), float(tukey_alpha))

    window_power = 1.0
    if apply_window_power_correction:
        window_power = float(np.mean(w * w))
        if window_power <= 0.0:
            raise ValueError("window power must stay positive")
    return w, window_power


def _detrend_linear(y: np.ndarray) -> np.ndarray:
    x = np.arange(len(y), dtype=float)
    p = np.polyfit(x, y, 1)
    return y - np.polyval(p, x)


def transfer_spectrum_T1(
    u_east: np.ndarray,
    u_north: np.ndarray,
    A_east: np.ndarray,
    A_north: np.ndarray,
    spacing_m: float,
    *,
    detrend: str | None = "linear",
    apply_hann: bool = True,
    one_sided_factor: float = 2.0,
    apply_window_power_correction: bool = True,
    along_track_taper: str | None = None,
    tukey_alpha: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate the along-swath transfer spectrum ``T_1(k)`` on positive ``k``.

    The authoritative direct discrete form keeps the *vector* dot product in
    spectral space while taking the FFT along the swath direction:

    ``T_1(k) = -C * (dx / (2*pi*N)) * Re[u_hat*(k) · A_hat(k)] / <w^2>``

    where ``u_hat = (û_e, û_n)`` and ``A_hat = (Â_e, Â_n)`` are the Fourier
    transforms of the east/north component sequences sampled along one swath
    column. This intentionally avoids the spurious cross terms that appear if
    the vector fields are projected to a local tangent before taking the
    spectrum.

    ``C`` is the one-sided factor; the authoritative mainline uses ``C=2``.
    ``<w^2>`` is the mean taper power when window-power correction is enabled
    (Hann or Tukey).
    The ``dx / (2*pi*N)`` term maps the raw
    FFT product back to a spectral density *per unit rad/m* (the same unit as
    the returned ``k`` axis), and the window-power correction removes the
    amplitude loss from the taper.

    When ``along_track_taper`` is ``None`` (default), ``apply_hann`` selects Hann vs rectangular.
    If ``along_track_taper`` is set to ``'hann'``, ``'tukey'``, or ``'none'``, it overrides ``apply_hann``.

    Returns
    -------
    k_rad_per_m
        Positive along-track wavenumber bins in rad/m, including ``k=0``.
    T1
        Real-valued transfer spectrum on the same grid.
    """
    ue = np.asarray(u_east, dtype=float).ravel()
    un = np.asarray(u_north, dtype=float).ravel()
    ae = np.asarray(A_east, dtype=float).ravel()
    an = np.asarray(A_north, dtype=float).ravel()
    if not (ue.shape == un.shape == ae.shape == an.shape):
        raise ValueError("u/A east-north component series must all have the same shape")
    if not np.isfinite(spacing_m) or spacing_m <= 0.0:
        raise ValueError("spacing_m must be a positive finite scalar")
    if not np.isfinite(one_sided_factor) or float(one_sided_factor) <= 0.0:
        raise ValueError("one_sided_factor must be a positive finite scalar")

    n = ue.size
    if n < 8:
        raise ValueError("need at least 8 samples for a minimal spectrum")

    if detrend == "linear":
        ue = _detrend_linear(ue)
        un = _detrend_linear(un)
        ae = _detrend_linear(ae)
        an = _detrend_linear(an)
    elif detrend is not None:
        raise ValueError("detrend must be 'linear' or None")

    w, window_power = build_along_track_fft_window(
        n,
        apply_hann=apply_hann,
        apply_window_power_correction=apply_window_power_correction,
        along_track_taper=along_track_taper,
        tukey_alpha=float(tukey_alpha),
    )
    if w is not None:
        ue = ue * w
        un = un * w
        ae = ae * w
        an = an * w

    ue_hat = np.fft.rfft(ue)
    un_hat = np.fft.rfft(un)
    ae_hat = np.fft.rfft(ae)
    an_hat = np.fft.rfft(an)

    norm = float(spacing_m) / (2.0 * np.pi * float(n))
    dot_hat = np.conj(ue_hat) * ae_hat + np.conj(un_hat) * an_hat
    t1 = -float(one_sided_factor) * norm * np.real(dot_hat) / window_power

    k = np.fft.rfftfreq(n, d=spacing_m) * (2.0 * np.pi)
    return k, t1


def audit_parseval_transfer_t1_column(
    u_east: np.ndarray,
    u_north: np.ndarray,
    A_east: np.ndarray,
    A_north: np.ndarray,
    spacing_m: float,
    *,
    detrend: str | None = "linear",
    one_sided_factor: float = 2.0,
    apply_hann: bool = True,
    apply_window_power_correction: bool = True,
    along_track_taper: str | None = None,
    tukey_alpha: float = 0.2,
    rtol: float = 1e-7,
    atol: float = 1e-12,
) -> dict[str, float | bool | int]:
    """沿轨一列：无窗时域 ``⟨u·A⟩``（与 ``transfer_spectrum_T1`` 相同 detrend）对比 ``∫ T_1(k)\\,dk``。

    离散实现下（``numpy.trapz``，含 ``k=0``），当 ``apply_hann=False`` 时数值上满足

    ``mean(u·A) + ∫ T_1(k)\\,dk ≈ 0``（等价于 ``-∫ T_1\\,dk ≈ mean(u·A)``），与 ``T_1`` 定义中的
    ``-C·(dx/(2πN))·Re(û*·Â)/⟨w²⟩`` 及 ``C=2`` 的 one-sided 约定一致。

    当 ``apply_hann=True`` 且 ``apply_window_power_correction=True`` 时，``⟨w²⟩`` 只对 taper 后 FFT
    做**逐模态幅度**补偿；**一般不能保证** ``∫ T_1\\,dk`` 与无窗时域 ``⟨u·A⟩`` 仍相等（泄漏与
    有效样本改变）。Tukey 等非矩形窗在 ``apply_window_power_correction=True`` 时同理。
    本函数同时返回 ``hann`` 与 ``no_hann`` 的积分及相对残差，便于强制审计。
    """
    ue = np.asarray(u_east, dtype=float).ravel()
    un = np.asarray(u_north, dtype=float).ravel()
    ae = np.asarray(A_east, dtype=float).ravel()
    an = np.asarray(A_north, dtype=float).ravel()
    if not (ue.shape == un.shape == ae.shape == an.shape):
        raise ValueError("u/A east-north component series must all have the same shape")
    if detrend == "linear":
        ue_m = _detrend_linear(ue.copy())
        un_m = _detrend_linear(un.copy())
        ae_m = _detrend_linear(ae.copy())
        an_m = _detrend_linear(an.copy())
    elif detrend is None:
        ue_m, un_m, ae_m, an_m = ue.copy(), un.copy(), ae.copy(), an.copy()
    else:
        raise ValueError("detrend must be 'linear' or None")

    time_mean_dot_no_window = float(np.mean(ue_m * ae_m + un_m * an_m))

    k_nh, t1_nh = transfer_spectrum_T1(
        ue,
        un,
        ae,
        an,
        float(spacing_m),
        detrend=detrend,
        apply_hann=False,
        one_sided_factor=float(one_sided_factor),
        apply_window_power_correction=True,
    )
    int_trapz_no_hann = float(_trapz(t1_nh, k_nh))

    k_hw, t1_hw = transfer_spectrum_T1(
        ue,
        un,
        ae,
        an,
        float(spacing_m),
        detrend=detrend,
        apply_hann=bool(apply_hann),
        one_sided_factor=float(one_sided_factor),
        apply_window_power_correction=bool(apply_window_power_correction),
        along_track_taper=along_track_taper,
        tukey_alpha=float(tukey_alpha),
    )
    int_trapz_hann_wpc = float(_trapz(t1_hw, k_hw))

    res_nh = float(time_mean_dot_no_window + int_trapz_no_hann)
    scale = max(abs(time_mean_dot_no_window), abs(int_trapz_no_hann), 1e-30)
    no_hann_identity_ok = bool(abs(res_nh) <= float(atol) + float(rtol) * scale)

    res_hw = float(time_mean_dot_no_window + int_trapz_hann_wpc)
    scale_hw = max(abs(time_mean_dot_no_window), abs(int_trapz_hann_wpc), 1e-30)
    hann_identity_ok = bool(abs(res_hw) <= float(atol) + float(rtol) * scale_hw)

    return {
        "n_samples": int(ue.size),
        "time_mean_dot_no_window": time_mean_dot_no_window,
        "trapz_t1_no_hann": int_trapz_no_hann,
        "residual_no_hann_sum": res_nh,
        "no_hann_identity_ok": no_hann_identity_ok,
        "trapz_t1_hann_wpc": int_trapz_hann_wpc,
        "residual_hann_wpc_sum": res_hw,
        "hann_vs_nowindow_abs_ratio": float(abs(int_trapz_hann_wpc) / max(abs(time_mean_dot_no_window), 1e-30)),
        "hann_identity_ok": hann_identity_ok,
    }


def kinetic_energy_spectrum_1d(
    u_east: np.ndarray,
    u_north: np.ndarray,
    spacing_m: float,
    *,
    detrend: str | None = "linear",
    apply_hann: bool = True,
    one_sided_factor: float = 2.0,
    apply_window_power_correction: bool = True,
    along_track_taper: str | None = None,
    tukey_alpha: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate the along-swath 1D kinetic-energy spectrum ``E_K(k)`` on positive ``k``.

    This intentionally shares the same discrete mouthpiece as ``transfer_spectrum_T1``:

    ``E_K(k) = 0.5 * C * (dx / (2*pi*N)) * (|u_hat|^2 + |v_hat|^2) / <w^2>``

    where ``C`` is the explicit one-sided factor and ``<w^2>`` is the mean taper
    power when window-power correction is enabled. The returned ``k`` axis is in
    rad/m, so the spectrum integrates to kinetic energy per unit mass.
    """
    ue = np.asarray(u_east, dtype=float).ravel()
    un = np.asarray(u_north, dtype=float).ravel()
    if ue.shape != un.shape:
        raise ValueError("u east/north component series must have the same shape")
    if not np.isfinite(spacing_m) or spacing_m <= 0.0:
        raise ValueError("spacing_m must be a positive finite scalar")
    if not np.isfinite(one_sided_factor) or float(one_sided_factor) <= 0.0:
        raise ValueError("one_sided_factor must be a positive finite scalar")

    n = ue.size
    if n < 8:
        raise ValueError("need at least 8 samples for a minimal spectrum")

    if detrend == "linear":
        ue = _detrend_linear(ue)
        un = _detrend_linear(un)
    elif detrend is not None:
        raise ValueError("detrend must be 'linear' or None")

    w, window_power = build_along_track_fft_window(
        n,
        apply_hann=apply_hann,
        apply_window_power_correction=apply_window_power_correction,
        along_track_taper=along_track_taper,
        tukey_alpha=float(tukey_alpha),
    )
    if w is not None:
        ue = ue * w
        un = un * w

    ue_hat = np.fft.rfft(ue)
    un_hat = np.fft.rfft(un)

    norm = 0.5 * float(one_sided_factor) * float(spacing_m) / (2.0 * np.pi * float(n))
    ek = norm * (np.abs(ue_hat) ** 2 + np.abs(un_hat) ** 2) / window_power
    k = np.fft.rfftfreq(n, d=spacing_m) * (2.0 * np.pi)
    return k, ek


def variance_spectrum_1d(
    eta: np.ndarray,
    spacing_m: float,
    *,
    detrend: str | None = "linear",
    apply_hann: bool = True,
    one_sided_factor: float = 2.0,
    apply_window_power_correction: bool = True,
    along_track_taper: str | None = None,
    tukey_alpha: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Along-swath one-sided variance spectrum ``Phi_eta(k)`` for a real scalar ``eta``.

    Uses the same discrete FFT mouthpiece as ``kinetic_energy_spectrum_1d`` (linear
    detrend, Hann/Tukey taper, window-power correction, ``k`` in rad/m) but without
    the kinetic-energy ``0.5`` prefactor:

    ``Phi_eta(k) = C * (dx / (2*pi*N)) * |eta_hat|^2 / <w^2>``

    so ``Phi_eta`` has SI units ``m^2 / (rad m^{-1})`` when ``eta`` is in metres.
    """
    y = np.asarray(eta, dtype=float).ravel()
    if not np.isfinite(spacing_m) or spacing_m <= 0.0:
        raise ValueError("spacing_m must be a positive finite scalar")
    if not np.isfinite(one_sided_factor) or float(one_sided_factor) <= 0.0:
        raise ValueError("one_sided_factor must be a positive finite scalar")

    n = y.size
    if n < 8:
        raise ValueError("need at least 8 samples for a minimal spectrum")

    if detrend == "linear":
        y = _detrend_linear(y)
    elif detrend is not None:
        raise ValueError("detrend must be 'linear' or None")

    w, window_power = build_along_track_fft_window(
        n,
        apply_hann=apply_hann,
        apply_window_power_correction=apply_window_power_correction,
        along_track_taper=along_track_taper,
        tukey_alpha=float(tukey_alpha),
    )
    if w is not None:
        y = y * w

    y_hat = np.fft.rfft(y)
    norm = float(one_sided_factor) * float(spacing_m) / (2.0 * np.pi * float(n))
    phi = norm * (np.abs(y_hat) ** 2) / window_power
    k = np.fft.rfftfreq(n, d=spacing_m) * (2.0 * np.pi)
    return k, phi
