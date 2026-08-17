"""Prediction-only evaluation utilities / 只预测评估工具。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import t as student_t
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # pragma: no cover - covered by subprocess portability test
    torch = None
    DataLoader = None
    Dataset = Any

from .datasets import collate_samples
from .trainer import FrozenOuterSplit, dataset_participant_ids, forward_batch


@dataclass(frozen=True)
class PerClassMetrics:
    """Metrics for one declared class / 单个声明类别的指标。"""

    label: int
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class EvaluationMetrics:
    """Participant/file/window metric summary / 参与者、文件或窗口指标摘要。"""

    n_rows: int
    balanced_accuracy: float
    macro_f1: float
    multiclass_log_loss: float
    confusion_matrix: tuple[tuple[int, ...], ...]
    class_order: tuple[int, ...] = ()
    per_class: tuple[PerClassMetrics, ...] = ()
    worst_class_label: int | None = None
    worst_class_precision: float = float("nan")
    worst_class_recall: float = float("nan")
    worst_class_f1: float = float("nan")
    multiclass_brier: float = float("nan")
    expected_calibration_error: float = float("nan")
    n_total: int = 0
    n_retained: int = 0
    n_dropped: int = 0
    coverage_rate: float = float("nan")


@dataclass(frozen=True)
class RepeatMetricSummary:
    """Repeat-level mean, dispersion and Student-t interval.

    repeat 层级均值、总体/样本标准差及 Student-t 置信区间。
    """

    n_repeats: int
    mean: float
    population_sd: float
    sample_sd: float
    ci95_lower: float
    ci95_upper: float
    ci_method: str = "student_t_95_two_sided"


@dataclass(frozen=True)
class PairedDeltaSummary:
    """Paired candidate-minus-baseline fold/seed deltas / 配对 fold/seed 差值。"""

    keys: tuple[tuple[int, int, int], ...]
    deltas: tuple[float, ...]
    summary: RepeatMetricSummary


def _multiclass_brier(
    labels: np.ndarray, probability: np.ndarray, classes: np.ndarray
) -> float:
    """Mean squared probability error summed over classes.

    对类别维求和后再对样本求均值的多分类 Brier 分数。
    """

    one_hot = (labels[:, None] == classes[None, :]).astype(np.float64)
    return float(np.mean(np.sum((probability - one_hot) ** 2, axis=1)))


def _expected_calibration_error(
    labels: np.ndarray,
    predicted: np.ndarray,
    probability: np.ndarray,
    n_bins: int,
) -> float:
    """Top-label equal-width ECE with deterministic edge handling.

    使用最高概率类别、等宽分箱并固定边界规则计算 ECE。
    """

    if n_bins <= 0:
        raise ValueError("ece_bins must be positive")
    confidence = probability.max(axis=1)
    correct = (predicted == labels).astype(np.float64)
    # English: digitize places confidence=1 in the final valid bin.
    # 中文：digitize 后再截断，确保 confidence=1 落入最后一个有效箱。
    bin_index = np.minimum((confidence * n_bins).astype(np.int64), n_bins - 1)
    error = 0.0
    for index in range(n_bins):
        selected = bin_index == index
        if np.any(selected):
            error += float(selected.mean()) * abs(
                float(correct[selected].mean()) - float(confidence[selected].mean())
            )
    return float(error)


def evaluate_predictions(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    class_order: tuple[int, ...] | list[int] | np.ndarray,
    retained_mask: np.ndarray | None = None,
    n_total: int | None = None,
    ece_bins: int = 10,
) -> EvaluationMetrics:
    """Evaluate already-produced predictions; this function never fits.

    评估已经生成的预测；本函数绝不拟合任何对象。
    """

    labels = np.asarray(y_true)
    probability = np.asarray(probabilities, dtype=np.float64)
    classes = np.asarray(class_order)
    if labels.ndim != 1 or classes.ndim != 1 or len(set(classes.tolist())) != classes.size:
        raise ValueError("labels and unique class_order must be one-dimensional")
    if probability.shape != (labels.size, classes.size):
        raise ValueError("probabilities must be [row,class] matching labels and class_order")
    original_count = int(labels.size)
    if retained_mask is not None:
        retained = np.asarray(retained_mask, dtype=bool)
        if retained.shape != (original_count,):
            raise ValueError("retained_mask must be row-aligned")
        labels = labels[retained]
        probability = probability[retained]
        coverage_total = original_count
    else:
        coverage_total = original_count if n_total is None else int(n_total)
    if coverage_total < original_count or labels.size == 0:
        raise ValueError("coverage denominator is invalid or no prediction was retained")
    if not set(labels.tolist()) <= set(classes.tolist()):
        raise ValueError("labels contain a class outside class_order")
    if not np.isfinite(probability).all() or np.any(probability < 0.0):
        raise ValueError("probabilities must be finite and non-negative")
    sums = probability.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=1e-6):
        raise ValueError("each probability row must sum to one")
    # English: Re-normalise float32-derived rows exactly before sklearn metrics;
    # this avoids version-dependent warnings without changing class ordering.
    # 中文：在 sklearn 指标前精确重归一化 float32 概率行，避免版本相关警告，
    # 同时不改变类别顺序。
    probability = probability / sums[:, None]
    predicted = classes[probability.argmax(axis=1)]
    matrix = confusion_matrix(labels, predicted, labels=classes)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predicted,
        labels=classes,
        zero_division=0,
    )
    per_class = tuple(
        PerClassMetrics(
            label=int(label),
            precision=float(precision[index]),
            recall=float(recall[index]),
            f1=float(f1[index]),
            support=int(support[index]),
        )
        for index, label in enumerate(classes)
    )
    worst_index = int(np.argmin(f1))
    retained_count = int(labels.size)
    return EvaluationMetrics(
        n_rows=retained_count,
        balanced_accuracy=float(balanced_accuracy_score(labels, predicted)),
        macro_f1=float(f1_score(labels, predicted, labels=classes, average="macro", zero_division=0)),
        multiclass_log_loss=float(log_loss(labels, probability, labels=classes)),
        confusion_matrix=tuple(tuple(int(value) for value in row) for row in matrix),
        class_order=tuple(int(value) for value in classes),
        per_class=per_class,
        worst_class_label=int(classes[worst_index]),
        worst_class_precision=float(precision[worst_index]),
        worst_class_recall=float(recall[worst_index]),
        worst_class_f1=float(f1[worst_index]),
        multiclass_brier=_multiclass_brier(labels, probability, classes),
        expected_calibration_error=_expected_calibration_error(
            labels, predicted, probability, ece_bins
        ),
        n_total=coverage_total,
        n_retained=retained_count,
        n_dropped=coverage_total - retained_count,
        coverage_rate=float(retained_count / coverage_total),
    )


def summarize_repeat_metric(values: np.ndarray | list[float] | tuple[float, ...]) -> RepeatMetricSummary:
    """Summarise independent repeat values with an explicit Student-t CI.

    对独立 repeat 值计算均值、总体/样本标准差和明确的 Student-t 95% CI。
    单个 repeat 无法估计区间，因此 CI 和样本标准差返回 NaN。
    """

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("repeat metric values must be a non-empty finite vector")
    mean = float(array.mean())
    population_sd = float(array.std(ddof=0))
    if array.size == 1:
        sample_sd = lower = upper = float("nan")
    else:
        sample_sd = float(array.std(ddof=1))
        half_width = float(
            student_t.ppf(0.975, df=array.size - 1) * sample_sd / np.sqrt(array.size)
        )
        lower, upper = mean - half_width, mean + half_width
    return RepeatMetricSummary(
        n_repeats=int(array.size),
        mean=mean,
        population_sd=population_sd,
        sample_sd=sample_sd,
        ci95_lower=lower,
        ci95_upper=upper,
    )


def paired_fold_seed_deltas(
    baseline: dict[tuple[int, int, int], float],
    candidate: dict[tuple[int, int, int], float],
) -> PairedDeltaSummary:
    """Compute candidate-minus-baseline deltas on exactly matching keys.

    仅在完全一致的 (repeat, fold, seed) 键上计算 candidate-baseline 配对差值。
    """

    if not baseline or set(baseline) != set(candidate):
        raise ValueError("paired comparisons require identical non-empty fold/seed keys")
    keys = tuple(sorted(baseline))
    deltas = tuple(float(candidate[key]) - float(baseline[key]) for key in keys)
    if not np.isfinite(np.asarray(deltas, dtype=np.float64)).all():
        raise ValueError("paired comparison values must be finite")
    return PairedDeltaSummary(keys=keys, deltas=deltas, summary=summarize_repeat_metric(deltas))


def predict_torch_dataset(
    model: torch.nn.Module,
    oof_dataset: Dataset,
    frozen_split: FrozenOuterSplit,
    *,
    batch_size: int = 64,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, list[Any]]:
    """Predict one frozen OOF dataset and verify its membership.

    预测一个冻结 OOF 数据集并校验其成员关系。此入口没有 optimizer 或 fit 调用。
    """

    if torch is None or DataLoader is None:
        raise ImportError("predict_torch_dataset requires optional dependency torch")

    participant_ids = dataset_participant_ids(oof_dataset)
    if not set(participant_ids) <= set(frozen_split.oof_participant_ids):
        raise ValueError("evaluation dataset contains subjects outside the frozen OOF partition")
    if set(participant_ids) & set(frozen_split.train_participant_ids):
        raise ValueError("outer-train subject reached OOF evaluation")
    target_device = torch.device(device)
    model.to(target_device)
    model.eval()
    probabilities, labels, identities = [], [], []
    loader = DataLoader(
        oof_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_samples,
    )
    with torch.no_grad():
        for batch in loader:
            if hasattr(model, "predict_probabilities") and "window_bag" not in batch:
                x = batch["x"].to(target_device)
                mask = batch.get("mask")
                probability = model.predict_probabilities(
                    x, None if mask is None else mask.to(target_device)
                )
            else:
                probability = torch.softmax(forward_batch(model, batch, target_device), dim=-1)
            probabilities.append(probability.cpu().numpy())
            labels.append(batch["y"].numpy())
            identities.extend(batch["identities"])
    return np.concatenate(probabilities), np.concatenate(labels), identities
