"""V2 fixed-kernel-samples resampling ablation for CompactCNN/InceptionTimeFull.

The convolution kernel sample counts stay unchanged when DL input fs changes.
Consequently physical kernel duration changes by design; this is not a
physical-time-matched experiment. Cases are registered here but are not run
automatically.

V2 固定卷积核样本数重采样消融仅适用于 CompactCNN/InceptionTimeFull。输入采样率
变化时卷积核样本数保持不变，因此其物理时长会有意改变；本实验不属于物理时间
匹配。这里只注册“参考条件 + 单因素变化”，不会自动执行训练或评估。
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np


ABLATION_ID = "fixed_kernel_samples_resampling_ablation"
REFERENCE_FS_HZ = 400.0
REFERENCE_WINDOW_SECONDS = 5.0
INCEPTION_KERNEL_SAMPLES = (39, 19, 9)
COMPACT_KERNEL_SAMPLES = (9, 9, 7)
SUPPORTED_MODEL_NAMES = ("CompactCNN1D", "InceptionTimeFull")

@dataclass(frozen=True)
class FixedKernelResamplingCase:
    """One registered DL condition / 一个只注册、不自动执行的深度学习条件。"""

    case_id: str
    model_name: str
    dl_fs_hz: float
    raw_window_seconds: float
    sequence_length_samples: int
    kernel_samples: tuple[int, ...]
    dilation: int
    comparison_id: str = ABLATION_ID
    kernel_policy: str = "fixed_sample_counts_not_physical_time_matched"
    representation_mode: str = "raw"
    scientific_status: str = "registered_not_run"
    input_transform: str = "dl_only_polyphase_antialias_from_canonical_400hz"
    canonical_engineering_fs_hz: float = REFERENCE_FS_HZ
    engineering_features_resampled: bool = False

    def __post_init__(self) -> None:
        if self.model_name not in SUPPORTED_MODEL_NAMES:
            raise ValueError("time-scale ablation supports CompactCNN1D/InceptionTimeFull only")
        if self.dl_fs_hz <= 0.0 or self.raw_window_seconds <= 0.0 or self.dilation <= 0:
            raise ValueError("fs, window and dilation must be positive")
        expected_length = int(round(self.dl_fs_hz * self.raw_window_seconds))
        if self.sequence_length_samples != expected_length or expected_length <= 0:
            raise ValueError("sequence length must equal round(fs*window_seconds)")
        expected_kernels = COMPACT_KERNEL_SAMPLES if self.model_name == "CompactCNN1D" else INCEPTION_KERNEL_SAMPLES
        if self.kernel_samples != expected_kernels:
            raise ValueError("V2 kernel sample counts must remain fixed across fs")


_SINGLE_FACTOR_CONDITIONS: tuple[tuple[str, float, float, int], ...] = (
    ("reference", 400.0, 5.0, 1),
    ("context_10s", 400.0, 10.0, 1),
    ("fs_100", 100.0, 5.0, 1),
    ("fs_160", 160.0, 5.0, 1),
    ("fs_200", 200.0, 5.0, 1),
    ("dilation_2", 400.0, 5.0, 2),
)

def _condition_name(dl_fs_hz: float, raw_window_seconds: float, dilation: int) -> str:
    """Resolve one allowed factor change / 解析一个允许的单因素变化。"""

    key = (float(dl_fs_hz), float(raw_window_seconds), int(dilation))
    for name, fs_hz, window_s, registered_dilation in _SINGLE_FACTOR_CONDITIONS:
        if key == (fs_hz, window_s, registered_dilation):
            return name
    raise ValueError(
        "V2-019 permits only reference plus one-factor fs/window/dilation conditions; "
        "factor interactions are not registered"
    )

def build_fixed_kernel_resampling_cases() -> tuple[FixedKernelResamplingCase, ...]:
    """Materialise the fixed 12-case registry / 生成固定 12 条注册条件，不运行测试。"""

    cases: list[FixedKernelResamplingCase] = []
    for model_name in SUPPORTED_MODEL_NAMES:
        kernels = COMPACT_KERNEL_SAMPLES if model_name == "CompactCNN1D" else INCEPTION_KERNEL_SAMPLES
        for condition_name, fs_hz, window_s, dilation in _SINGLE_FACTOR_CONDITIONS:
            cases.append(
                FixedKernelResamplingCase(
                    case_id=f"{model_name.lower()}__{condition_name}",
                    model_name=model_name,
                    dl_fs_hz=fs_hz,
                    raw_window_seconds=window_s,
                    sequence_length_samples=int(round(fs_hz * window_s)),
                    kernel_samples=kernels,
                    dilation=dilation,
                )
            )
    if len({case.case_id for case in cases}) != len(cases):
        raise RuntimeError("fixed-kernel resampling case IDs must be unique")
    return tuple(cases)

def fixed_kernel_case(case_id: str) -> FixedKernelResamplingCase:
    """Resolve one exact registered identity without aliases."""

    matches = tuple(case for case in build_fixed_kernel_resampling_cases() if case.case_id == str(case_id))
    if len(matches) != 1:
        raise ValueError(f"unknown fixed-kernel case_id: {case_id}")
    return matches[0]

def materialize_fixed_kernel_case_config(
    case: FixedKernelResamplingCase | str,
) -> dict[str, object]:
    """Return the formal single-factor identity; never execute the case."""

    selected = fixed_kernel_case(case) if isinstance(case, str) else case
    model_id = "compact_cnn" if selected.model_name == "CompactCNN1D" else "inception_full"
    return {
        "comparison_id": selected.comparison_id,
        "case_id": selected.case_id,
        "model_id": model_id,
        "representation_mode": "raw",
        "catalog_role": (
            "reference_condition"
            if selected.dl_fs_hz == 400.0 and selected.raw_window_seconds == 5.0 and selected.dilation == 1
            else "single_factor_ablation"
        ),
        "dl_input_fs_hz": float(selected.dl_fs_hz),
        "canonical_signal_view_fs_hz": REFERENCE_FS_HZ,
        "canonical_feature_and_peak_fs_hz": REFERENCE_FS_HZ,
        "raw_window_seconds": float(selected.raw_window_seconds),
        "sequence_length_samples": int(selected.sequence_length_samples),
        "kernel_samples": tuple(int(value) for value in selected.kernel_samples),
        "dilation": int(selected.dilation),
        "kernel_policy": selected.kernel_policy,
        "input_transform": selected.input_transform,
        "anti_aliasing": (
            "scipy_signal_resample_poly_kaiser_beta5"
            if selected.dl_fs_hz != REFERENCE_FS_HZ
            else "not_applied_same_sampling_rate"
        ),
        "dl_input_only": True,
        "engineering_features_resampled": False,
        "automatic_execution": False,
        "scientific_status": selected.scientific_status,
    }

def prepare_fixed_kernel_dl_input(
    values: np.ndarray,
    sample_mask: np.ndarray,
    case: FixedKernelResamplingCase | str,
    *,
    source_fs_hz: float = REFERENCE_FS_HZ,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Build only the DL input view with anti-aliased polyphase resampling.

    The canonical signal/peak/feature view remains 400 Hz and is not returned
    or mutated. Padding masks must describe a contiguous valid prefix.
    """

    from scipy.signal import resample_poly

    selected = fixed_kernel_case(case) if isinstance(case, str) else case
    if float(source_fs_hz) != REFERENCE_FS_HZ:
        raise ValueError("V2-019 source signal view must remain canonical 400 Hz")
    array = np.asarray(values, dtype=np.float32)
    mask = np.asarray(sample_mask, dtype=bool)
    expected_source_length = int(round(REFERENCE_FS_HZ * selected.raw_window_seconds))
    if (
        array.ndim != 3
        or mask.shape != (array.shape[0], array.shape[2])
        or array.shape[2] != expected_source_length
        or not np.isfinite(array).all()
    ):
        raise ValueError(
            "fixed-kernel DL input requires finite [sample,channel,T] canonical "
            "400-Hz windows and a matching [sample,T] mask"
        )
    valid_lengths = mask.sum(axis=1).astype(np.int64)
    expected_mask = np.arange(mask.shape[1], dtype=np.int64)[None, :] < valid_lengths[:, None]
    if not np.array_equal(mask, expected_mask):
        raise ValueError("fixed-kernel resampling requires contiguous valid-prefix masks")
    if np.any(valid_lengths < 2):
        raise ValueError("each fixed-kernel input needs at least two valid source samples")

    target_length = int(selected.sequence_length_samples)
    output = np.zeros(
        (array.shape[0], array.shape[1], target_length),
        dtype=np.float32,
    )
    output_mask = np.zeros((array.shape[0], target_length), dtype=bool)
    ratio = Fraction(int(round(selected.dl_fs_hz)), int(round(source_fs_hz)))
    for sample_index, valid_length in enumerate(valid_lengths.tolist()):
        valid = array[sample_index, :, :valid_length]
        if ratio.numerator == ratio.denominator:
            transformed = valid
        else:
            transformed = resample_poly(
                valid,
                up=ratio.numerator,
                down=ratio.denominator,
                axis=-1,
                window=("kaiser", 5.0),
                padtype="constant",
            ).astype(np.float32, copy=False)
        expected_valid = min(
            target_length,
            int(round(valid_length * selected.dl_fs_hz / source_fs_hz)),
        )
        if abs(transformed.shape[-1] - expected_valid) > 1:
            raise RuntimeError("polyphase resampler produced an unexpected valid length")
        copied = min(expected_valid, transformed.shape[-1], target_length)
        output[sample_index, :, :copied] = transformed[:, :copied]
        output_mask[sample_index, :copied] = True
    provenance = materialize_fixed_kernel_case_config(selected)
    provenance.update(
        {
            "source_sequence_length_samples": expected_source_length,
            "output_sequence_length_samples": target_length,
            "resample_up": ratio.numerator,
            "resample_down": ratio.denominator,
            "mask_transform": "contiguous_valid_prefix_scaled_with_dl_sampling_rate",
        }
    )
    return output, output_mask, provenance

def create_fixed_kernel_resampling_model(
    model_name: str,
    *,
    n_channels: int,
    n_classes: int,
    dl_fs_hz: float,
    raw_window_seconds: float = REFERENCE_WINDOW_SECONDS,
    dilation: int = 1,
    seed: int = 42,
):
    """Construct one registered case / 构造一个已注册条件且不改变卷积核样本数。"""

    if n_channels <= 0 or n_classes <= 1:
        raise ValueError("model dimensions are invalid")
    condition_name = _condition_name(dl_fs_hz, raw_window_seconds, dilation)
    case = FixedKernelResamplingCase(
        case_id=f"{model_name.lower()}__{condition_name}",
        model_name=model_name,
        dl_fs_hz=float(dl_fs_hz),
        raw_window_seconds=float(raw_window_seconds),
        sequence_length_samples=int(round(float(dl_fs_hz) * float(raw_window_seconds))),
        kernel_samples=(COMPACT_KERNEL_SAMPLES if model_name == "CompactCNN1D" else INCEPTION_KERNEL_SAMPLES),
        dilation=int(dilation),
    )
    import torch

    torch.manual_seed(int(seed))
    if model_name == "CompactCNN1D":
        from .compact_cnn import CompactCNN1D

        model = CompactCNN1D(
            n_channels,
            n_classes,
            kernel_sizes=case.kernel_samples,
            dilations=(case.dilation,) * 3,
        )
    elif model_name == "InceptionTimeFull":
        from .inception import InceptionTimeSingleNetwork

        model = InceptionTimeSingleNetwork(
            n_channels,
            n_classes,
            variant="full",
            kernel_sizes=case.kernel_samples,
            dilation=case.dilation,
        )
    else:
        raise ValueError("ShapeFormer/small/matrix models are outside V2-019")
    model.fixed_kernel_resampling_provenance = {
        **case.__dict__,
        "reference_condition": "5_seconds_at_400_hz_dilation_1",
        "seed": int(seed),
        "automatic_execution": False,
    }
    return model


# V1 physical-time function names deliberately fail rather than change semantics silently.
def build_physical_time_cases(*args, **kwargs):
    """Reject the V1 semantic route / 拒绝把 V1 物理时间匹配路线当作 V2 证据。"""

    raise RuntimeError("V1 physical-time-matched cases are not V2 evidence; use " "build_fixed_kernel_resampling_cases")

def create_time_scaled_model(*args, **kwargs):
    """Reject the V1 constructor / 拒绝使用 V1 构造器并静默改变实验语义。"""

    raise RuntimeError(
        "V1 physical-time model construction is disabled in V2; use " "create_fixed_kernel_resampling_model"
    )


__all__ = [
    "ABLATION_ID",
    "COMPACT_KERNEL_SAMPLES",
    "FixedKernelResamplingCase",
    "INCEPTION_KERNEL_SAMPLES",
    "REFERENCE_FS_HZ",
    "REFERENCE_WINDOW_SECONDS",
    "SUPPORTED_MODEL_NAMES",
    "build_fixed_kernel_resampling_cases",
    "create_fixed_kernel_resampling_model",
    "fixed_kernel_case",
    "materialize_fixed_kernel_case_config",
    "prepare_fixed_kernel_dl_input",
]
