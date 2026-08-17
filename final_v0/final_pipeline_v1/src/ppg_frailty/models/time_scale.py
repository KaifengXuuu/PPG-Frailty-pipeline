'''Physical-time audit helpers for CNN/Inception ablations.

English: Acquisition remains 400 Hz. This module converts declared physical kernel
durations to deterministic odd sample counts for a separate DL-only sampling grid and
records the realised duration/error. It never changes the feature or audit grid.

中文：采集与特征审计网格始终保持 400 Hz。本模块只为 DL 消融把显式物理卷积核
时长转换为确定性的奇数样本数，并记录实际时长和误差，绝不静默改写采集频率。
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


REFERENCE_INCEPTION_KERNEL_DURATIONS_S = (39 / 400, 19 / 400, 9 / 400)
REFERENCE_COMPACT_KERNEL_DURATIONS_S = (9 / 400, 9 / 400, 7 / 400)


@dataclass(frozen=True)
class RealizedKernelSet:
    '''Declared seconds and realised odd samples / 声明时长与实际奇数样本。'''

    fs_hz: float
    declared_durations_s: tuple[float, ...]
    sample_counts: tuple[int, ...]
    realized_durations_s: tuple[float, ...]
    absolute_errors_s: tuple[float, ...]


@dataclass(frozen=True)
class PhysicalTimeAblationCase:
    '''One named matched time-scale condition / 一个具名配对时间尺度条件。'''

    case_id: str
    dl_fs_hz: float
    raw_window_s: float
    inception_kernels: RealizedKernelSet
    compact_kernels: RealizedKernelSet
    dilation: int
    representation_mode: str


def _nearest_odd(value: float, *, minimum: int = 3) -> int:
    '''Choose the nearest positive odd integer / 选择最近的正奇数。'''

    if value <= 0.0:
        raise ValueError('kernel duration must map to a positive sample count')
    lower = int(value)
    if lower % 2 == 0:
        lower -= 1
    lower = max(minimum if minimum % 2 else minimum + 1, lower)
    upper = lower + 2
    # English: Equal-distance ties choose the larger kernel to avoid shortening context.
    # 中文：等距离时选择较大的核，避免无意缩短物理上下文。
    return lower if abs(value - lower) < abs(upper - value) else upper


def realize_kernel_durations(
    durations_s: Iterable[float], *, fs_hz: float
) -> RealizedKernelSet:
    '''Convert seconds to auditable odd samples / 将秒制核转换为可审计奇数样本。'''

    durations = tuple(float(value) for value in durations_s)
    if fs_hz <= 0.0 or not durations or any(value <= 0.0 for value in durations):
        raise ValueError('fs_hz and every kernel duration must be positive')
    samples = tuple(_nearest_odd(value * float(fs_hz)) for value in durations)
    realized = tuple(value / float(fs_hz) for value in samples)
    errors = tuple(abs(left - right) for left, right in zip(durations, realized))
    return RealizedKernelSet(float(fs_hz), durations, samples, realized, errors)


def inception_local_receptive_field(
    kernel_samples: Iterable[int], *, depth: int, dilation: int = 1
) -> int:
    '''Return the theoretical longest-branch local field / 返回最长分支理论局部感受野。'''

    kernels = tuple(int(value) for value in kernel_samples)
    if not kernels or depth <= 0 or dilation <= 0:
        raise ValueError('kernels, depth and dilation must be positive')
    if any(value <= 0 or value % 2 == 0 for value in kernels):
        raise ValueError('kernel_samples must be positive odd integers')
    return 1 + int(depth) * int(dilation) * (max(kernels) - 1)


def build_physical_time_cases(
    *,
    dl_fs_values: Iterable[float] = (100.0, 160.0, 200.0, 400.0),
    raw_window_values_s: Iterable[float] = (5.0, 10.0),
    dilation_values: Iterable[int] = (1, 2),
    representation_modes: Iterable[str] = ('raw', 'feature_vector', 'feature_matrix', 'fusion'),
) -> tuple[PhysicalTimeAblationCase, ...]:
    '''Materialise the declared comparison grid / 物化规范声明的比较网格。

    English: A formal runner must reuse identical frozen folds/seeds and report
    participant metrics, calibration, coverage, runtime and memory.

    中文：正式运行必须复用相同 participant folds/seeds，并报告参与者指标、
    校准、覆盖率、运行时间和内存；本函数只物化候选条件。
    '''

    modes = tuple(str(value) for value in representation_modes)
    if set(modes) != {'raw', 'feature_vector', 'feature_matrix', 'fusion'}:
        raise ValueError('representation_modes must contain the four canonical modes exactly')
    cases: list[PhysicalTimeAblationCase] = []
    for fs_hz in tuple(float(value) for value in dl_fs_values):
        for window_s in tuple(float(value) for value in raw_window_values_s):
            for dilation in tuple(int(value) for value in dilation_values):
                for mode in modes:
                    case_id = f'dlfs_{fs_hz:g}_window_{window_s:g}_dilation_{dilation}_{mode}'
                    cases.append(
                        PhysicalTimeAblationCase(
                            case_id=case_id,
                            dl_fs_hz=fs_hz,
                            raw_window_s=window_s,
                            inception_kernels=realize_kernel_durations(
                                REFERENCE_INCEPTION_KERNEL_DURATIONS_S, fs_hz=fs_hz
                            ),
                            compact_kernels=realize_kernel_durations(
                                REFERENCE_COMPACT_KERNEL_DURATIONS_S, fs_hz=fs_hz
                            ),
                            dilation=dilation,
                            representation_mode=mode,
                        )
                    )
    if len({case.case_id for case in cases}) != len(cases):
        raise RuntimeError('physical-time case IDs are not unique')
    return tuple(cases)


def create_time_scaled_model(
    model_name: str,
    *,
    n_channels: int,
    n_classes: int,
    dl_fs_hz: float,
    dilation: int = 1,
    seed: int = 42,
):
    '''Build an executable CNN/Inception time-scale ablation / 构建可执行时间消融模型。

    English: This is an explicit comparison factory, separate from the reviewed
    default factory. It preserves physical kernel duration across DL sampling rates
    and records the realised kernel provenance on the returned model.

    中文：这是与冻结默认模型工厂分离的显式比较工厂；它在不同 DL 采样率下保持
    卷积核物理时长，并把实际奇数样本和误差绑定到返回模型。
    '''

    if n_channels <= 0 or n_classes <= 1 or dilation <= 0:
        raise ValueError('invalid model dimensions or dilation')
    import torch

    torch.manual_seed(int(seed))
    if model_name == 'CompactCNN1D':
        from .compact_cnn import CompactCNN1D

        realized = realize_kernel_durations(
            REFERENCE_COMPACT_KERNEL_DURATIONS_S, fs_hz=dl_fs_hz
        )
        model = CompactCNN1D(
            n_channels,
            n_classes,
            kernel_sizes=realized.sample_counts,
            dilations=(int(dilation),) * 3,
        )
    elif model_name in {'InceptionTimeFull', 'InceptionTimeSmall', 'InceptionTimeMatrix'}:
        from .inception import InceptionTimeSingleNetwork

        realized = realize_kernel_durations(
            REFERENCE_INCEPTION_KERNEL_DURATIONS_S, fs_hz=dl_fs_hz
        )
        variant = 'small' if model_name == 'InceptionTimeSmall' else 'full'
        model = InceptionTimeSingleNetwork(
            n_channels,
            n_classes,
            variant=variant,
            kernel_sizes=realized.sample_counts,
            dilation=int(dilation),
        )
    else:
        raise ValueError('time-scale factory supports only CompactCNN1D and Inception variants')
    model.physical_time_provenance = {
        'model_name': model_name,
        'dl_fs_hz': float(dl_fs_hz),
        'dilation': int(dilation),
        'declared_kernel_durations_s': realized.declared_durations_s,
        'realized_kernel_samples': realized.sample_counts,
        'realized_kernel_durations_s': realized.realized_durations_s,
        'absolute_kernel_errors_s': realized.absolute_errors_s,
        'seed': int(seed),
        'acquisition_and_feature_grid_hz': 400.0,
    }
    return model


__all__ = [
    'PhysicalTimeAblationCase',
    'RealizedKernelSet',
    'build_physical_time_cases',
    'create_time_scaled_model',
    'inception_local_receptive_field',
    'realize_kernel_durations',
]
