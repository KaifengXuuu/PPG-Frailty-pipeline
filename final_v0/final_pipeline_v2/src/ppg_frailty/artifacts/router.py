"""Reducer registry 与无 fallback 路由 / Reducer registry and no-fallback routing."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping, TypeVar

import numpy as np

from ..contracts import ArtifactReductionResult, SignalRoute
from ..signal.views import CANONICAL_FS_HZ, CanonicalSignalViews
from .base import ArtifactReducer, failure_result, parameters_dict, validate_result
from .bss import (
    FastIcaBssConfig,
    FastIcaBssReducer,
    NmfBssConfig,
    NmfBssReducer,
    PcaBssConfig,
    PcaBssReducer,
)
from .decomposition import SsaConfig, SsaReducer
from .identity import IdentityReducer
from .legacy import (
    CeemdLiteNlmsLegacyConfig,
    CeemdLiteNlmsLegacyReducer,
    DwtA2LegacyConfig,
    DwtA2LegacyReducer,
    EmdSiftingConfig,
    EmdSiftingRateOnlyReducer,
)
from .nlms import NlmsConfig, NlmsReducer
from .spectral import SpectralMaskConfig, SpectralMaskReducer


ConfigType = TypeVar("ConfigType")


class UnsupportedReducer(ArtifactReducer):
    """注册但不伪造 learned denoiser / Registered explicit unsupported learned route."""

    reducer_version = "unsupported_in_v1"

    def __init__(self, reducer_id: str, parameters: Mapping[str, Any] | None = None) -> None:
        self.reducer_id = reducer_id
        self.parameters = dict(parameters or {})

    def reduce(
        self,
        ppg: np.ndarray,
        imu_processed: Mapping[str, np.ndarray] | None,
        *,
        fs_hz: float = CANONICAL_FS_HZ,
    ) -> ArtifactReductionResult:
        """返回 unsupported，绝不返回 input 伪装成功 / Return no fake waveform."""

        return failure_result(
            self,
            "learned denoiser is registered but unsupported until an audited model artifact exists",
            status="unsupported",
            parameters=self.parameters,
        )


def _config(config_type: type[ConfigType], parameters: Mapping[str, Any] | None) -> ConfigType:
    """拒绝未知参数后构建 frozen config / Build config after rejecting unknown keys."""

    payload = dict(parameters or {})
    allowed = {item.name for item in fields(config_type)}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError("unknown reducer parameters: " + ",".join(unknown))
    if "delay_taps" in payload:
        payload["delay_taps"] = tuple(int(value) for value in payload["delay_taps"])
    if "preserve_band_hz" in payload:
        band = payload["preserve_band_hz"]
        if not isinstance(band, (list, tuple)):
            raise ValueError("preserve_band_hz must be a two-value sequence")
        payload["preserve_band_hz"] = tuple(float(value) for value in band)
    return config_type(**payload)  # type: ignore[arg-type]


def get_reducer(name: str, parameters: Mapping[str, Any] | None = None) -> ArtifactReducer:
    """按稳定名称返回 reducer / Return a reducer by stable registry name."""

    normalized = name.strip().lower().replace("-", "_")
    if normalized in {"identity", "none", "direct"}:
        if parameters:
            raise ValueError("identity accepts no parameters")
        return IdentityReducer()
    if normalized in {"nlms", "nlms_imu_anc"}:
        return NlmsReducer(_config(NlmsConfig, parameters))
    if normalized in {"ssa", "ssa_decomposition", "decomposition"}:
        return SsaReducer(_config(SsaConfig, parameters))
    if normalized in {"spectral_mask", "spectral", "stft", "stft_imu_mask"}:
        return SpectralMaskReducer(_config(SpectralMaskConfig, parameters))
    if normalized in {"pca", "pca_bss"}:
        return PcaBssReducer(_config(PcaBssConfig, parameters))
    if normalized in {"ica", "fastica", "fastica_bss"}:
        return FastIcaBssReducer(_config(FastIcaBssConfig, parameters))
    if normalized in {"nmf", "nmf_bss"}:
        return NmfBssReducer(_config(NmfBssConfig, parameters))
    if normalized == "emd_sifting_rate_only":
        return EmdSiftingRateOnlyReducer(_config(EmdSiftingConfig, parameters))
    if normalized == "ceemd_lite_nlms_legacy":
        return CeemdLiteNlmsLegacyReducer(_config(CeemdLiteNlmsLegacyConfig, parameters))
    if normalized == "dwt_a2_legacy":
        return DwtA2LegacyReducer(_config(DwtA2LegacyConfig, parameters))
    if normalized in {"learned", "learned_denoiser", "hybrid_denoiser", "onnx_denoiser"}:
        return UnsupportedReducer(normalized, parameters)
    raise KeyError(f"unknown artifact reducer: {name}")


def reducer_audit_metadata(
    name: str,
    parameters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return executable reducer identity, resolved parameters and short method text."""

    reducer = get_reducer(name, parameters)
    description = str(reducer.algorithm_kernel_description).strip()
    if not description or len(description) > 300:
        raise ValueError(
            f"{reducer.reducer_id} algorithm/kernel description must contain 1-300 characters"
        )
    config = getattr(reducer, "config", getattr(reducer, "parameters", {}))
    return {
        "reducer_id": reducer.reducer_id,
        "reducer_version": reducer.reducer_version,
        "resolved_parameters": parameters_dict(config),
        "algorithm_kernel_description": description,
        "description_character_count": len(description),
    }


@dataclass(frozen=True)
class ArtifactRouteOutcome:
    """路由结果及可选更新视图 / Route result and optional updated views."""

    result: ArtifactReductionResult
    route: SignalRoute
    views: CanonicalSignalViews | None


def run_artifact_route(
    signal: np.ndarray | CanonicalSignalViews,
    reducer_name: str,
    *,
    parameters: Mapping[str, Any] | None = None,
    imu_processed: Mapping[str, np.ndarray] | None = None,
    fs_hz: float = CANONICAL_FS_HZ,
) -> ArtifactRouteOutcome:
    """运行指定 reducer；失败 route=DROPPED / Run one reducer without fallback.

    中文：失败或 unsupported 时 `views=None`、`x_ar=None`；调用方只能 drop/reject，
    不能自动换回 direct。English: failure yields no views/waveform and must be dropped.
    """

    views: CanonicalSignalViews | None
    if isinstance(signal, CanonicalSignalViews):
        views = signal
        views.validate()
        if views.route is SignalRoute.ARTIFACT_RATE_ONLY:
            raise ValueError("chaining artifact reducers is outside the V1 protocol")
        source = np.asarray(views.x_filter, dtype=np.float64)
        references = views.imu_processed if imu_processed is None else imu_processed
    else:
        views = None
        source = np.asarray(signal, dtype=np.float64)
        references = imu_processed
    reducer = get_reducer(reducer_name, parameters)
    result = reducer.reduce(source, references, fs_hz=fs_hz)
    validate_result(source, result)
    if result.status != "success":
        return ArtifactRouteOutcome(result, SignalRoute.DROPPED, None)
    route = SignalRoute.IDENTITY if result.is_identity else SignalRoute.ARTIFACT_RATE_ONLY
    updated = views.with_artifact_result(result) if views is not None else None
    return ArtifactRouteOutcome(result, route, updated)
