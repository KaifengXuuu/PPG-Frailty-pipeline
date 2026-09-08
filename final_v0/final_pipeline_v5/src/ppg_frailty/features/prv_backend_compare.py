"""Fixed-PPI PRV backend comparison only / 仅固定 PPI 的 PRV 后端对照。

English: Aura and rhenan adapters receive the exact same untouched PPI vector as the
formal local implementation. Imports are lazy, optional failures stay within one backend,
and no cleaner or classifier integration exists here.
中文：Aura 与 rhenan 适配器接收和正式本地实现完全相同、未经清洗的 PPI 向量。
依赖缺失只影响对应后端；本模块不做 cleaner，也不接入分类器。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib
from importlib import metadata
from typing import Iterable, Mapping, Sequence

import numpy as np

from ..contracts import to_strict_json_value


PRV_BACKEND_COMPARISON_SCHEMA = "ppg_frailty.prv_backend_comparison.v2"
SUPPORTED_PRV_BACKENDS = ("local", "aura_hrv_analysis", "rhenan_hrv")

@dataclass(frozen=True)
class PrvBackendResult:
    """One backend row / 单个后端结果行。"""

    backend: str
    status: str
    package_name: str
    package_version: str | None
    interval_count: int
    input_sha256: str
    values: dict[str, float | None]
    cleaner_applied: bool = False
    classifier_integrated: bool = False
    error_type: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return strict-JSON-safe payload / 返回 strict JSON 可写载荷。"""

        return to_strict_json_value(asdict(self))

def _validate_ppi(ppi_ms: Sequence[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(ppi_ms, dtype=np.float64).ravel()
    if values.size < 4 or not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ValueError("fixed PPI vector requires at least four finite positive milliseconds")
    return values

def _ppi_hash(values: np.ndarray) -> str:
    payload = np.asarray(values, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(str(payload.size).encode("ascii"))
    digest.update(b":")
    digest.update(payload.tobytes(order="C"))
    return digest.hexdigest()

def fixed_ppi_fixtures() -> dict[str, np.ndarray]:
    """Deterministic vectors; no preprocessing / 确定性向量，不做预处理。"""

    index = np.arange(512, dtype=np.float64)
    return {
        "steady_75bpm": np.full(512, 800.0, dtype=np.float64),
        "alternating_75bpm": np.where(index.astype(int) % 2 == 0, 760.0, 840.0),
        "dual_modulated": 800.0
        + 35.0 * np.sin(2.0 * np.pi * index / 128.0)
        + 12.0 * np.sin(2.0 * np.pi * index / 23.0),
        "slow_trend": np.linspace(740.0, 880.0, 512, dtype=np.float64),
        "single_outlier_unmodified": np.where(index == 256, 1400.0, 800.0),
    }

def _numeric_values(payload: Mapping[object, object]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for key, raw_value in payload.items():
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        result[str(key)] = value if np.isfinite(value) else None
    return result

def _local_features(ppi_ms: np.ndarray) -> dict[str, float | None]:
    """Local formal equations evaluated directly on fixed PPI / 本地方程。"""

    differences = np.diff(ppi_ms)
    heart_rate = 60000.0 / ppi_ms
    sdnn = float(np.std(ppi_ms, ddof=1))
    sdsd = float(np.std(differences, ddof=1)) if differences.size > 1 else 0.0
    values = {
        "mean_nni": float(np.mean(ppi_ms)),
        "median_nni": float(np.median(ppi_ms)),
        "range_nni": float(np.ptp(ppi_ms)),
        "sdnn": sdnn,
        "sdsd": sdsd,
        "rmssd": float(np.sqrt(np.mean(np.square(differences)))),
        "nni_50": float(np.count_nonzero(np.abs(differences) > 50.0)),
        "pnni_50": float(100.0 * np.mean(np.abs(differences) > 50.0)),
        "cvnni": float(sdnn / np.mean(ppi_ms)),
        "cvsd": float(np.sqrt(np.mean(np.square(differences))) / np.mean(ppi_ms)),
        "mean_hr": float(np.mean(heart_rate)),
        "min_hr": float(np.min(heart_rate)),
        "max_hr": float(np.max(heart_rate)),
        "std_hr": float(np.std(heart_rate, ddof=1)),
    }
    return _numeric_values(values)

def _package_version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None

def _aura_features(ppi_ms: np.ndarray) -> dict[str, float | None]:
    module = importlib.import_module("hrvanalysis")
    vector = ppi_ms.tolist()
    combined: dict[str, float | None] = {}
    for function_name in (
        "get_time_domain_features",
        "get_geometrical_features",
        "get_frequency_domain_features",
    ):
        combined.update(_numeric_values(getattr(module, function_name)(vector)))
    combined.update({key.lower(): value for key, value in _numeric_values(module.get_csi_cvi_features(vector)).items()})
    combined.update(
        {
            key.lower().replace(" ", "_").replace("/", "_"): value
            for key, value in _numeric_values(module.get_poincare_plot_features(vector)).items()
        }
    )
    sample_entropy = module.get_sampen(vector)
    if isinstance(sample_entropy, Mapping):
        combined.update({f"sampen_{key.lower()}": value for key, value in _numeric_values(sample_entropy).items()})
    else:
        combined.update(_numeric_values({"sampen": sample_entropy}))
    return combined

def _rhenan_features(ppi_ms: np.ndarray) -> dict[str, float | None]:
    rri_module = importlib.import_module("hrv.rri")
    classical = importlib.import_module("hrv.classical")
    rri = rri_module.RRi(ppi_ms.tolist())
    combined: dict[str, float | None] = {}
    for function_name in ("time_domain", "frequency_domain", "non_linear"):
        combined.update(_numeric_values(getattr(classical, function_name)(rri)))
    return combined

def evaluate_prv_backend(
    ppi_ms: Sequence[float] | np.ndarray,
    backend: str,
) -> PrvBackendResult:
    """Evaluate one backend without cleaning / 不清洗地运行一个后端。"""

    values = _validate_ppi(ppi_ms)
    normalized = backend.strip().lower()
    if normalized not in SUPPORTED_PRV_BACKENDS:
        raise ValueError(f"unknown PRV comparison backend: {backend}")
    package = {
        "local": "ppg_frailty",
        "aura_hrv_analysis": "hrv-analysis",
        "rhenan_hrv": "hrv",
    }[normalized]
    version = None if normalized == "local" else _package_version(package)
    try:
        if normalized == "local":
            features = _local_features(values)
            version = PRV_BACKEND_COMPARISON_SCHEMA
        elif normalized == "aura_hrv_analysis":
            features = _aura_features(values)
        else:
            features = _rhenan_features(values)
        return PrvBackendResult(
            backend=normalized,
            status="success",
            package_name=package,
            package_version=version,
            interval_count=int(values.size),
            input_sha256=_ppi_hash(values),
            values=features,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        return PrvBackendResult(
            backend=normalized,
            status="unavailable_optional_dependency",
            package_name=package,
            package_version=version,
            interval_count=int(values.size),
            input_sha256=_ppi_hash(values),
            values={},
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
    except Exception as exc:
        return PrvBackendResult(
            backend=normalized,
            status="backend_failed",
            package_name=package,
            package_version=version,
            interval_count=int(values.size),
            input_sha256=_ppi_hash(values),
            values={},
            error_type=type(exc).__name__,
            error_message=str(exc),
        )

def run_prv_backend_comparison(
    backends: Iterable[str] = SUPPORTED_PRV_BACKENDS,
    fixture_ids: Iterable[str] | None = None,
) -> dict[str, object]:
    """Run only function-level fixed-vector comparisons / 仅运行函数级固定向量对照。"""

    requested_backends = tuple(str(item) for item in backends)
    unknown = sorted(set(requested_backends) - set(SUPPORTED_PRV_BACKENDS))
    if unknown or not requested_backends:
        raise ValueError("invalid PRV backends: " + ",".join(unknown))
    fixtures = fixed_ppi_fixtures()
    selected_ids = tuple(fixtures) if fixture_ids is None else tuple(str(item) for item in fixture_ids)
    missing = sorted(set(selected_ids) - set(fixtures))
    if missing or not selected_ids:
        raise ValueError("invalid fixed PPI fixture IDs: " + ",".join(missing))
    rows: list[dict[str, object]] = []
    for fixture_id in selected_ids:
        vector = fixtures[fixture_id]
        backend_rows = [evaluate_prv_backend(vector, backend).to_dict() for backend in requested_backends]
        rows.append(
            {
                "fixture_id": fixture_id,
                "interval_count": int(vector.size),
                "input_sha256": _ppi_hash(vector),
                "backends": backend_rows,
            }
        )
    result_statuses = tuple(str(backend_row["status"]) for fixture in rows for backend_row in fixture["backends"])
    diagnostic_success = all(value == "success" for value in result_statuses)
    return {
        "schema_version": PRV_BACKEND_COMPARISON_SCHEMA,
        "status": ("diagnostic_success_not_exact_profile_evidence" if diagnostic_success else "failed_closed"),
        "execution_authority": {
            "status": "diagnostic_only_unverified_runtime",
            "formal_optional_profile_evidence": False,
        },
        "comparison_scope": "fixed_ppi_function_outputs_only",
        "requested_backends": list(requested_backends),
        "cleaner_applied": False,
        "classifier_integrated": False,
        "formal_local_backend": "local",
        "optional_comparison_backends": ["aura_hrv_analysis", "rhenan_hrv"],
        "fixtures": rows,
    }


__all__ = [
    "PRV_BACKEND_COMPARISON_SCHEMA",
    "PrvBackendResult",
    "SUPPORTED_PRV_BACKENDS",
    "evaluate_prv_backend",
    "fixed_ppi_fixtures",
    "run_prv_backend_comparison",
]
