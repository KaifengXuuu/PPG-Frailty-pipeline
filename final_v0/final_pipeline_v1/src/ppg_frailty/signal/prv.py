"""保留真实时间和邻接关系的 HR/PPI/PRV / Time- and adjacency-preserving PRV."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import interpolate, signal

from ..contracts import PulseResult, SignalRoute


TIME_METRICS = (
    "accepted_interval_count", "accepted_duration_s", "coverage",
    "ppi_mean_s", "ppi_median_s", "ppi_sd_s", "ppi_iqr_s", "ppi_mad_s", "ppi_cv",
    "hr_mean_bpm", "hr_median_bpm", "hr_sd_bpm", "sdnn_s", "rmssd_s", "sdsd_s",
    "nn50_count", "pnn50", "sd1_s", "sd2_s", "sd1_sd2_ratio", "sample_entropy",
)
SPECTRAL_METRICS = (
    "vlf_power_s2", "lf_power_s2", "hf_power_s2", "lf_hf_ratio",
    "lf_normalized", "hf_normalized",
)


@dataclass(frozen=True)
class PrvResult:
    """值、逐字段 validity 与资格理由 / Values, field validity, and eligibility."""

    values: dict[str, float]
    validity: dict[str, bool]
    reasons: tuple[str, ...]
    time_domain_eligible: bool
    frequency_domain_eligible: bool
    sample_entropy_eligible: bool
    interval_timestamps_s: np.ndarray


def _nan_payload() -> tuple[dict[str, float], dict[str, bool]]:
    """创建不可用占位；零不代表缺失 / Build NaN/false unavailable placeholders."""

    names = TIME_METRICS + SPECTRAL_METRICS
    return ({name: float("nan") for name in names}, {name: False for name in names})


def _sample_entropy(values: np.ndarray, m: int = 2, tolerance: float | None = None) -> float:
    """确定性 SampEn(m=2,r=.2 SD) / Deterministic sample entropy implementation."""

    x = np.asarray(values, dtype=np.float64).ravel()
    if x.size < m + 2:
        return float("nan")
    r = float(tolerance if tolerance is not None else 0.2 * np.std(x, ddof=1))
    if not np.isfinite(r) or r <= 0.0:
        return float("nan")

    def matches(length: int) -> int:
        """计数无自匹配的 Chebyshev 邻居 / Count non-self Chebyshev matches."""

        count = 0
        for left in range(x.size - length):
            template = x[left : left + length]
            for right in range(left + 1, x.size - length + 1):
                if np.max(np.abs(template - x[right : right + length])) <= r:
                    count += 1
        return count

    count_m = matches(m)
    count_m1 = matches(m + 1)
    # English: SampEn is the negative log ratio of *match probabilities*.
    # The m and m+1 template sets contain different numbers of unordered pairs,
    # so a raw-count ratio is biased and is not the declared SampEn estimator.
    # 中文：SampEn 必须使用“匹配概率”之比。m 与 m+1 的模板对总数不同，
    # 直接用匹配计数之比会产生系统偏差，不符合声明的 SampEn 定义。
    template_count_m = x.size - m + 1
    template_count_m1 = x.size - (m + 1) + 1
    possible_pairs_m = template_count_m * (template_count_m - 1) / 2.0
    possible_pairs_m1 = template_count_m1 * (template_count_m1 - 1) / 2.0
    probability_m = count_m / possible_pairs_m if possible_pairs_m > 0.0 else 0.0
    probability_m1 = count_m1 / possible_pairs_m1 if possible_pairs_m1 > 0.0 else 0.0
    if probability_m <= 0.0 or probability_m1 <= 0.0:
        return float("nan")
    return float(-np.log(probability_m1 / probability_m))


def _band_integral(frequencies: np.ndarray, power: np.ndarray, low: float, high: float) -> float:
    """积分右端含高边界的频带 / Integrate a frequency band including its upper edge."""

    mask = (frequencies >= low) & (frequencies <= high)
    if np.count_nonzero(mask) < 2:
        return float("nan")
    return float(np.trapz(power[mask], frequencies[mask]))


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """连续合格 interval 的右开区间 / Contiguous valid interval runs."""

    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    changes = np.diff(padded.astype(np.int8))
    return list(
        zip(
            np.flatnonzero(changes == 1).astype(int).tolist(),
            np.flatnonzero(changes == -1).astype(int).tolist(),
        )
    )


def compute_prv(
    pulse: PulseResult,
    *,
    observation_duration_s: float,
    role: str,
    route: SignalRoute = SignalRoute.DIRECT,
    q_rate_qualified: bool,
) -> PrvResult:
    """计算完整 rate/time/spectral PRV / Compute complete rate/time/spectral PRV.

    RMSSD/SDSD/NN50 只使用原始相邻且两侧均有效的 interval pair，绝不跨缺失。
    RMSSD/SDSD/NN50 use only originally adjacent pairs with both intervals valid.
    """

    values, validity = _nan_payload()
    ppi_all = np.asarray(pulse.ppi_s, dtype=np.float64)
    valid = np.asarray(pulse.valid_interval_mask, dtype=bool).copy()
    adjacency = np.asarray(pulse.adjacency_mask, dtype=bool)
    starts = np.asarray(pulse.interval_start_peak_indices, dtype=np.int64)
    stops = np.asarray(pulse.interval_stop_peak_indices, dtype=np.int64)
    times = np.asarray(pulse.peak_timestamps_s, dtype=np.float64)
    if not (ppi_all.shape == valid.shape == adjacency.shape == starts.shape == stops.shape):
        raise ValueError("PulseResult interval arrays must share shape")
    if np.any(starts < 0) or np.any(stops >= times.size):
        raise ValueError("PulseResult interval endpoints exceed peak timestamp array")
    interval_times = times[stops] if stops.size else np.empty(0, dtype=np.float64)
    valid &= adjacency & np.isfinite(ppi_all) & (ppi_all > 0.0)
    accepted = ppi_all[valid]
    accepted_count = int(accepted.size)
    accepted_duration = float(np.sum(accepted))
    time_span = float(times[-1] - times[0]) if times.size >= 2 else 0.0
    coverage = float(accepted_duration / time_span) if time_span > 0.0 else 0.0
    values.update({
        "accepted_interval_count": float(accepted_count),
        "accepted_duration_s": accepted_duration,
        "coverage": min(1.0, coverage),
    })
    validity.update({"accepted_interval_count": True, "accepted_duration_s": True, "coverage": True})
    reasons: list[str] = []
    rate_eligible = observation_duration_s >= 8.0 and times.size >= 5 and accepted_count >= 4
    if not rate_eligible:
        reasons.append("rate_requires_8s_5peaks")
    elif accepted_count:
        hr = 60.0 / accepted
        basic = {
            "ppi_mean_s": float(np.mean(accepted)),
            "ppi_median_s": float(np.median(accepted)),
            "ppi_sd_s": float(np.std(accepted, ddof=1)) if accepted_count > 1 else 0.0,
            "ppi_iqr_s": float(np.percentile(accepted, 75) - np.percentile(accepted, 25)),
            "ppi_mad_s": float(np.median(np.abs(accepted - np.median(accepted)))),
            "ppi_cv": float(np.std(accepted, ddof=1) / np.mean(accepted)) if accepted_count > 1 else 0.0,
            "hr_mean_bpm": float(np.mean(hr)),
            "hr_median_bpm": float(np.median(hr)),
            "hr_sd_bpm": float(np.std(hr, ddof=1)) if accepted_count > 1 else 0.0,
        }
        values.update(basic)
        validity.update({name: True for name in basic})

    time_eligible = bool(observation_duration_s >= 60.0 and coverage >= 0.80 and accepted_count >= 30)
    if not time_eligible:
        reasons.append("time_prv_requires_60s_and_0p80_coverage")
    else:
        values["sdnn_s"] = float(np.std(accepted, ddof=1))
        validity["sdnn_s"] = True
        # 中文：pair mask 对应 ppi[i] 与 ppi[i+1]，必须两者有效且原始邻接。
        # English: Each pair requires two valid intervals and preserved original adjacency.
        endpoint_contiguous = stops[:-1] == starts[1:]
        pair_mask = (
            valid[:-1]
            & valid[1:]
            & adjacency[:-1]
            & adjacency[1:]
            & endpoint_contiguous
        )
        differences = np.diff(ppi_all)[pair_mask]
        if differences.size:
            rmssd = float(np.sqrt(np.mean(np.square(differences))))
            sdsd = float(np.std(differences, ddof=1)) if differences.size > 1 else 0.0
            nn50 = int(np.count_nonzero(np.abs(differences) > 0.050))
            sd1 = float(np.sqrt(0.5) * sdsd)
            variance = float(np.var(accepted, ddof=1))
            sd2 = float(np.sqrt(max(2.0 * variance - 0.5 * sdsd * sdsd, 0.0)))
            pair_values = {
                "rmssd_s": rmssd, "sdsd_s": sdsd, "nn50_count": float(nn50),
                "pnn50": float(nn50 / differences.size), "sd1_s": sd1, "sd2_s": sd2,
                "sd1_sd2_ratio": float(sd1 / sd2) if sd2 > 0.0 else float("nan"),
            }
            values.update(pair_values)
            validity.update({name: np.isfinite(value) for name, value in pair_values.items()})
        else:
            reasons.append("no_adjacent_valid_interval_pairs")

    # 中文：不把 gap 两侧拼成相邻模板；SampEn 只用最长连续 interval run。
    # English: SampEn never concatenates templates across a rejected interval.
    runs = _true_runs(valid & adjacency)
    longest_run = max(
        runs,
        key=lambda bounds: float(
            np.sum(ppi_all[bounds[0] : bounds[1]])
        ),
        default=(0, 0),
    )
    run_start, run_stop = longest_run
    contiguous_ppi = ppi_all[run_start:run_stop]
    entropy_eligible = contiguous_ppi.size >= 200
    if entropy_eligible:
        entropy = _sample_entropy(contiguous_ppi)
        values["sample_entropy"] = entropy
        validity["sample_entropy"] = bool(np.isfinite(entropy))
        if not np.isfinite(entropy):
            reasons.append("sample_entropy_degenerate")
        if contiguous_ppi.size != accepted_count:
            reasons.append("sample_entropy_uses_longest_contiguous_run")
    else:
        reasons.append("sample_entropy_requires_200_intervals")

    # English: R1-R4 are the frozen long recovery roles in the formal protocol.
    # 中文：R1-R4 是正式协议冻结的长时恢复角色，必须具备频域 PRV 资格。
    allowed_roles = {
        "static", "reference", "baseline", "relax", "recovery", "b", "r",
        "r1", "r2", "r3", "r4",
    }
    route_eligible = route in {
        SignalRoute.DIRECT,
        SignalRoute.IDENTITY,
        SignalRoute.ARTIFACT_RATE_ONLY,
    }
    contiguous_duration = float(np.sum(contiguous_ppi))
    frequency_eligible = bool(
        route_eligible
        and q_rate_qualified
        and role.strip().lower() in allowed_roles
        and observation_duration_s >= 300.0
        and contiguous_duration >= 300.0
        and contiguous_ppi.size >= 200
        and coverage >= 0.80
    )
    if not frequency_eligible:
        reasons.append(
            "frequency_prv_requires_qrate_static_contiguous300s_200intervals"
        )
    else:
        valid_times = interval_times[run_start:run_stop]
        # 中文：使用真实 endpoint time 插值 4 Hz tachogram，再线性去趋势。
        # English: Interpolate true endpoint time to 4 Hz, then linearly detrend.
        grid = np.arange(valid_times[0], valid_times[-1], 0.25, dtype=np.float64)
        if grid.size >= 256 and np.all(np.diff(valid_times) > 0.0):
            tachogram = interpolate.interp1d(
                valid_times,
                contiguous_ppi,
                kind="linear",
                bounds_error=True,
            )(grid)
            tachogram = signal.detrend(tachogram, type="linear")
            frequencies, power = signal.welch(
                tachogram, fs=4.0, nperseg=min(1024, tachogram.size), detrend=False
            )
            vlf = _band_integral(frequencies, power, 0.003, 0.04)
            lf = _band_integral(frequencies, power, 0.04, 0.15)
            hf = _band_integral(frequencies, power, 0.15, 0.40)
            denominator = lf + hf
            spectral = {
                "vlf_power_s2": vlf, "lf_power_s2": lf, "hf_power_s2": hf,
                "lf_hf_ratio": float(lf / hf) if hf > 0.0 else float("nan"),
                "lf_normalized": float(lf / denominator) if denominator > 0.0 else float("nan"),
                "hf_normalized": float(hf / denominator) if denominator > 0.0 else float("nan"),
            }
            values.update(spectral)
            validity.update({name: np.isfinite(value) for name, value in spectral.items()})
        else:
            frequency_eligible = False
            reasons.append("frequency_prv_tachogram_insufficient")
    return PrvResult(
        values=values,
        validity=validity,
        reasons=tuple(dict.fromkeys(reasons)),
        time_domain_eligible=time_eligible,
        frequency_domain_eligible=frequency_eligible,
        sample_entropy_eligible=entropy_eligible,
        interval_timestamps_s=interval_times,
    )
