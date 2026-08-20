"""保留真实时间和邻接关系的 HR/PPI/PRV / Time- and adjacency-preserving PRV."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from scipy import interpolate, signal

from ..contracts import PulseResult, SignalRoute
from .peaks import MIN_BASIC_RATE_PEAKS


MIN_TIME_DOMAIN_PRV_INTERVALS = 30


@dataclass(frozen=True)
class PrvConfig:
    """Runtime PRV thresholds and spectral parameters.

    Defaults reproduce the pre-configuration implementation.  ``from_mapping``
    reads the public ``features`` section and supplies an explicit default for
    every omitted field, so persisted configuration is executed rather than
    merely reported as provenance.
    """

    rate_prv_min_duration_s: float = 8.0
    rate_prv_min_peaks: int = MIN_BASIC_RATE_PEAKS
    time_prv_min_duration_s: float = 60.0
    time_prv_min_coverage: float = 0.80
    time_prv_min_intervals: int = MIN_TIME_DOMAIN_PRV_INTERVALS
    spectral_prv_min_duration_s: float = 300.0
    spectral_prv_min_coverage: float = 0.80
    spectral_prv_min_intervals: int = 200
    tachogram_fs_hz: float = 4.0
    vlf_band_hz: tuple[float, float] = (0.003, 0.04)
    lf_band_hz: tuple[float, float] = (0.04, 0.15)
    hf_band_hz: tuple[float, float] = (0.15, 0.40)
    sample_entropy_m: int = 2
    sample_entropy_r_sd_fraction: float = 0.20
    sample_entropy_min_intervals: int = 200

    @classmethod
    def from_mapping(cls, features: Mapping[str, Any] | None) -> "PrvConfig":
        """Resolve supported feature fields without requiring boilerplate."""

        if features is None:
            return cls().validated()
        if not isinstance(features, Mapping):
            raise ValueError("features must be a mapping when resolving PRV config")
        defaults = cls()
        if (
            "rate_prv_min_peaks" in features
            and "time_prv_min_accepted_peaks" in features
            and features["rate_prv_min_peaks"]
            != features["time_prv_min_accepted_peaks"]
        ):
            raise ValueError(
                "features.rate_prv_min_peaks conflicts with its deprecated "
                "time_prv_min_accepted_peaks alias"
            )
        bands_raw = features.get("spectral_bands_hz", {})
        if bands_raw is None:
            bands_raw = {}
        if not isinstance(bands_raw, Mapping):
            raise ValueError("features.spectral_bands_hz must be a mapping")
        unknown_bands = set(bands_raw) - {"vlf", "lf", "hf"}
        if unknown_bands:
            raise ValueError(
                "features.spectral_bands_hz contains unknown bands: "
                f"{sorted(unknown_bands)}"
            )
        entropy_raw = features.get("sample_entropy", {})
        if entropy_raw is None:
            entropy_raw = {}
        if not isinstance(entropy_raw, Mapping):
            raise ValueError("features.sample_entropy must be a mapping")
        unknown_entropy = set(entropy_raw) - {
            "m",
            "r_sd_fraction",
            "min_intervals",
        }
        if unknown_entropy:
            raise ValueError(
                "features.sample_entropy contains unknown fields: "
                f"{sorted(unknown_entropy)}"
            )

        def band(name: str, default: tuple[float, float]) -> tuple[float, float]:
            raw = bands_raw.get(name, default)
            if (
                isinstance(raw, (str, bytes))
                or not isinstance(raw, (list, tuple))
                or len(raw) != 2
            ):
                raise ValueError(f"features.spectral_bands_hz.{name} must have two edges")
            return (float(raw[0]), float(raw[1]))

        # ``time_prv_min_accepted_peaks`` was the old public name for the
        # basic-rate peak threshold.  Execute it as that compatibility alias;
        # the time-domain interval threshold has its own unambiguous field.
        rate_min_peaks = features.get(
            "rate_prv_min_peaks",
            features.get(
                "time_prv_min_accepted_peaks",
                defaults.rate_prv_min_peaks,
            ),
        )
        time_coverage = features.get(
            "time_prv_min_coverage",
            defaults.time_prv_min_coverage,
        )
        result = cls(
            rate_prv_min_duration_s=float(
                features.get(
                    "rate_prv_min_duration_s",
                    defaults.rate_prv_min_duration_s,
                )
            ),
            rate_prv_min_peaks=rate_min_peaks,
            time_prv_min_duration_s=float(
                features.get(
                    "time_prv_min_duration_s",
                    defaults.time_prv_min_duration_s,
                )
            ),
            time_prv_min_coverage=float(time_coverage),
            time_prv_min_intervals=features.get(
                "time_prv_min_intervals",
                defaults.time_prv_min_intervals,
            ),
            spectral_prv_min_duration_s=float(
                features.get(
                    "spectral_prv_min_duration_s",
                    defaults.spectral_prv_min_duration_s,
                )
            ),
            spectral_prv_min_coverage=float(
                features.get(
                    "spectral_prv_min_coverage",
                    defaults.spectral_prv_min_coverage,
                )
            ),
            spectral_prv_min_intervals=features.get(
                "spectral_prv_min_intervals",
                defaults.spectral_prv_min_intervals,
            ),
            tachogram_fs_hz=float(
                features.get("tachogram_fs_hz", defaults.tachogram_fs_hz)
            ),
            vlf_band_hz=band("vlf", defaults.vlf_band_hz),
            lf_band_hz=band("lf", defaults.lf_band_hz),
            hf_band_hz=band("hf", defaults.hf_band_hz),
            sample_entropy_m=entropy_raw.get("m", defaults.sample_entropy_m),
            sample_entropy_r_sd_fraction=float(
                entropy_raw.get(
                    "r_sd_fraction",
                    defaults.sample_entropy_r_sd_fraction,
                )
            ),
            sample_entropy_min_intervals=entropy_raw.get(
                "min_intervals",
                defaults.sample_entropy_min_intervals,
            ),
        )
        return result.validated()

    def validated(self) -> "PrvConfig":
        """Reject non-finite or structurally invalid numerical settings."""

        positive_values = {
            "rate_prv_min_duration_s": self.rate_prv_min_duration_s,
            "time_prv_min_duration_s": self.time_prv_min_duration_s,
            "spectral_prv_min_duration_s": self.spectral_prv_min_duration_s,
            "tachogram_fs_hz": self.tachogram_fs_hz,
            "sample_entropy_r_sd_fraction": self.sample_entropy_r_sd_fraction,
        }
        for name, raw in positive_values.items():
            value = float(raw)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name, raw, minimum in (
            ("rate_prv_min_peaks", self.rate_prv_min_peaks, 2),
            ("time_prv_min_intervals", self.time_prv_min_intervals, 2),
            ("spectral_prv_min_intervals", self.spectral_prv_min_intervals, 2),
            ("sample_entropy_m", self.sample_entropy_m, 1),
            ("sample_entropy_min_intervals", self.sample_entropy_min_intervals, 1),
        ):
            if isinstance(raw, bool) or not isinstance(raw, (int, np.integer)):
                raise ValueError(f"{name} must be an integer")
            if int(raw) < minimum:
                raise ValueError(f"{name} must be at least {minimum}")
        if self.sample_entropy_min_intervals < self.sample_entropy_m + 2:
            raise ValueError(
                "sample_entropy_min_intervals must be at least sample_entropy_m + 2"
            )
        for name, raw in (
            ("time_prv_min_coverage", self.time_prv_min_coverage),
            ("spectral_prv_min_coverage", self.spectral_prv_min_coverage),
        ):
            value = float(raw)
            if not np.isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be finite in (0,1]")
        nyquist = self.tachogram_fs_hz / 2.0
        named_bands = (
            ("vlf", self.vlf_band_hz),
            ("lf", self.lf_band_hz),
            ("hf", self.hf_band_hz),
        )
        for name, band_edges in named_bands:
            low, high = map(float, band_edges)
            if (
                not np.isfinite([low, high]).all()
                or low < 0.0
                or low >= high
                or high > nyquist
            ):
                raise ValueError(
                    f"{name} spectral band must be finite, ordered, and within tachogram Nyquist"
                )
        if self.vlf_band_hz[1] > self.lf_band_hz[0] or self.lf_band_hz[1] > self.hf_band_hz[0]:
            raise ValueError("VLF/LF/HF bands must be ordered and non-overlapping")
        return self

    def to_dict(self) -> dict[str, Any]:
        """Return the exact runtime payload used by ``compute_prv``."""

        return {
            "rate_prv_min_duration_s": float(self.rate_prv_min_duration_s),
            "rate_prv_min_peaks": int(self.rate_prv_min_peaks),
            "time_prv_min_duration_s": float(self.time_prv_min_duration_s),
            "time_prv_min_coverage": float(self.time_prv_min_coverage),
            "time_prv_min_intervals": int(self.time_prv_min_intervals),
            "spectral_prv_min_duration_s": float(self.spectral_prv_min_duration_s),
            "spectral_prv_min_coverage": float(self.spectral_prv_min_coverage),
            "spectral_prv_min_intervals": int(self.spectral_prv_min_intervals),
            "tachogram_fs_hz": float(self.tachogram_fs_hz),
            "spectral_bands_hz": {
                "vlf": list(map(float, self.vlf_band_hz)),
                "lf": list(map(float, self.lf_band_hz)),
                "hf": list(map(float, self.hf_band_hz)),
            },
            "sample_entropy": {
                "m": int(self.sample_entropy_m),
                "r_sd_fraction": float(self.sample_entropy_r_sd_fraction),
                "min_intervals": int(self.sample_entropy_min_intervals),
            },
        }


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
    source_route: SignalRoute
    detection_run_id: str
    configuration: dict[str, Any]


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
    return float(np.trapezoid(power[mask], frequencies[mask]))


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
    config: PrvConfig | None = None,
) -> PrvResult:
    """计算完整 rate/time/spectral PRV / Compute complete rate/time/spectral PRV.

    RMSSD/SDSD/NN50 只使用原始相邻且两侧均有效的 interval pair，绝不跨缺失。
    RMSSD/SDSD/NN50 use only originally adjacent pairs with both intervals valid.
    """

    resolved = (config or PrvConfig()).validated()
    pulse.validate_identity()
    pulse_route = (
        pulse.source_route
        if isinstance(pulse.source_route, SignalRoute)
        else SignalRoute(str(pulse.source_route))
    )
    resolved_route = route if isinstance(route, SignalRoute) else SignalRoute(route)
    if pulse_route is not resolved_route:
        raise ValueError("PRV route must match the Pulse/PPI source route")
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
    rate_eligible = (
        observation_duration_s >= resolved.rate_prv_min_duration_s
        and times.size >= resolved.rate_prv_min_peaks
        and accepted_count >= resolved.rate_prv_min_peaks - 1
    )
    if not rate_eligible:
        reasons.append("rate_prv_configured_duration_or_peak_requirement_not_met")
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

    time_eligible = bool(
        observation_duration_s >= resolved.time_prv_min_duration_s
        and coverage >= resolved.time_prv_min_coverage
        and accepted_count >= resolved.time_prv_min_intervals
    )
    if not time_eligible:
        reasons.append("time_prv_configured_requirements_not_met")
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
    entropy_eligible = (
        contiguous_ppi.size >= resolved.sample_entropy_min_intervals
    )
    if entropy_eligible:
        entropy = _sample_entropy(
            contiguous_ppi,
            m=resolved.sample_entropy_m,
            tolerance=(
                resolved.sample_entropy_r_sd_fraction
                * np.std(contiguous_ppi, ddof=1)
            ),
        )
        values["sample_entropy"] = entropy
        validity["sample_entropy"] = bool(np.isfinite(entropy))
        if not np.isfinite(entropy):
            reasons.append("sample_entropy_degenerate")
        if contiguous_ppi.size != accepted_count:
            reasons.append("sample_entropy_uses_longest_contiguous_run")
    else:
        reasons.append("sample_entropy_configured_min_intervals_not_met")

    # English: V2 callers must pass the canonical role family. B/R can support long
    # static PRV; numeric file suffixes and free-text aliases are not roles here.
    # 中文：V2 调用方必须传入规范 role family；B/R 可支持长时静态 PRV，数字文件
    # 后缀和自由文本别名在本层不再被视为 role。
    allowed_roles = {"B", "R"}
    route_eligible = resolved_route in {
        SignalRoute.DIRECT,
        SignalRoute.IDENTITY,
        SignalRoute.ARTIFACT_RATE_ONLY,
    }
    contiguous_duration = float(np.sum(contiguous_ppi))
    frequency_eligible = bool(
        route_eligible
        and q_rate_qualified
        and role.strip().upper() in allowed_roles
        and observation_duration_s >= resolved.spectral_prv_min_duration_s
        and contiguous_duration >= resolved.spectral_prv_min_duration_s
        and contiguous_ppi.size >= resolved.spectral_prv_min_intervals
        and coverage >= resolved.spectral_prv_min_coverage
    )
    if not frequency_eligible:
        reasons.append("frequency_prv_configured_requirements_not_met")
    else:
        valid_times = interval_times[run_start:run_stop]
        # 中文：使用真实 endpoint time 按配置采样 tachogram，再线性去趋势。
        # English: Interpolate true endpoint time at the configured tachogram rate.
        grid = np.arange(
            valid_times[0],
            valid_times[-1],
            1.0 / resolved.tachogram_fs_hz,
            dtype=np.float64,
        )
        if grid.size >= 256 and np.all(np.diff(valid_times) > 0.0):
            tachogram = interpolate.interp1d(
                valid_times,
                contiguous_ppi,
                kind="linear",
                bounds_error=True,
            )(grid)
            tachogram = signal.detrend(tachogram, type="linear")
            frequencies, power = signal.welch(
                tachogram,
                fs=resolved.tachogram_fs_hz,
                nperseg=min(1024, tachogram.size),
                detrend=False,
            )
            vlf = _band_integral(frequencies, power, *resolved.vlf_band_hz)
            lf = _band_integral(frequencies, power, *resolved.lf_band_hz)
            hf = _band_integral(frequencies, power, *resolved.hf_band_hz)
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
        source_route=resolved_route,
        detection_run_id=pulse.detection_run_id,
        configuration=resolved.to_dict(),
    )
