"""Focused V2 participant-cluster inference and archive smoke tests."""

from __future__ import annotations

import json

import numpy as np
import pytest

from ppg_frailty.training.statistics import (
    ComparisonArchive,
    ConfigMetrics,
    ManualFinalSelection,
    ParticipantPrediction,
    build_config_metrics_from_predictions_and_fold_summaries,
    holm_adjust,
    paired_participant_permutation,
    participant_cluster_bootstrap,
    rank_top10,
    read_verified_manual_selections,
    verify_comparison_archive,
    write_comparison_archive,
)


def _predictions(*, degrade: bool = False) -> tuple[ParticipantPrediction, ...]:
    rows = []
    for repeat in range(5):
        for label in (0, 1, 2):
            for index in (0, 1):
                probabilities = [0.05, 0.05, 0.05]
                probabilities[label] = 0.90
                if degrade and label == 2 and index == 0:
                    probabilities = [0.90, 0.05, 0.05]
                rows.append(
                    ParticipantPrediction(
                        participant_id=f"p{label}{index}",
                        label=label,
                        repeat=repeat,
                        probabilities=tuple(probabilities),
                    )
                )
    return tuple(rows)


def _fold_payload(
    predictions: tuple[ParticipantPrediction, ...],
) -> tuple[dict[str, float], dict[str, tuple[tuple[float, ...], ...]], dict[str, tuple[str, ...]]]:
    balanced: dict[str, float] = {}
    matrices: dict[str, tuple[tuple[float, ...], ...]] = {}
    rosters: dict[str, tuple[str, ...]] = {}
    for repeat in range(5):
        rows = sorted(
            (item for item in predictions if item.repeat == repeat),
            key=lambda item: item.participant_id,
        )
        for fold in range(5):
            selected = rows[fold::5]
            key = f"r{repeat}f{fold}"
            matrix = np.zeros((3, 3), dtype=np.float64)
            for item in selected:
                matrix[item.label, int(np.argmax(item.probabilities))] += 1.0
            support = matrix.sum(axis=1) > 0.0
            balanced[key] = float(
                np.mean(np.diag(matrix)[support] / matrix.sum(axis=1)[support])
            )
            matrices[key] = tuple(tuple(float(value) for value in row) for row in matrix)
            rosters[key] = tuple(item.participant_id for item in selected)
    return balanced, matrices, rosters


def _metrics(
    index: int,
    *,
    role: str = "reference",
    balanced_accuracy_lcb95: float | None = None,
    macro_f1_lcb95: float | None = None,
) -> ConfigMetrics:
    score = 0.50 + index * 0.01
    return ConfigMetrics(
        config_id=f"cfg{index:02d}",
        registry_role=role,
        participant_mean_balanced_accuracy=score,
        participant_mean_macro_f1=score - 0.01,
        worst_fold_balanced_accuracy=score - 0.05,
        balanced_accuracy_lcb95=(
            score - 0.08
            if balanced_accuracy_lcb95 is None
            else balanced_accuracy_lcb95
        ),
        macro_f1_lcb95=(
            score - 0.09 if macro_f1_lcb95 is None else macro_f1_lcb95
        ),
        worst_class_recall=score - 0.04,
        worst_class_f1=score - 0.03,
        expected_calibration_error=0.10,
        variability={"repeat_standard_deviation": 0.02},
        confusion_matrices={"participant": ((2.0, 0.0), (0.0, 2.0))},
        inference_cost={"milliseconds_per_participant": 1.5},
        parameter_count=100 + index,
    )


def test_small_resample_smoke_uses_participant_clusters() -> None:
    reference = _predictions()
    candidate = _predictions(degrade=True)
    bootstrap = participant_cluster_bootstrap(reference, n_resamples=40, seed=42)
    permutation = paired_participant_permutation(
        reference, candidate, n_resamples=80, seed=42
    )
    assert bootstrap.n_participants == 6
    assert bootstrap.n_repeats == 5
    assert bootstrap.lcb95 == bootstrap.ci95_lower
    assert permutation.n_participants == 6
    assert permutation.exchange_unit == "participant_with_all_repeats"
    holm = holm_adjust(
        {"a": 0.01, "b": 0.04},
        comparison_family="shapeformer",
        metric="balanced_accuracy",
    )
    assert len(holm) == 2
    assert all(item.family_size == 2 for item in holm)


def test_top10_is_ba_sorted_review_list_without_auto_selection() -> None:
    ranked = rank_top10(_metrics(index) for index in range(11))
    assert len(ranked) == 10
    assert ranked[0].config_id == "cfg10"
    assert ranked[-1].config_id == "cfg01"


def test_metrics_builder_computes_both_lcbs_and_fails_closed_on_missing_cost() -> None:
    predictions = _predictions()
    fold_ba, fold_confusion, fold_rosters = _fold_payload(predictions)
    metrics, bootstrap = build_config_metrics_from_predictions_and_fold_summaries(
        config_id="cfg_missing_ops",
        registry_role="comparison",
        predictions=predictions,
        fold_balanced_accuracies=fold_ba,
        fold_confusion_matrices=fold_confusion,
        fold_participant_rosters=fold_rosters,
        inference_cost={"milliseconds_per_participant": None},
        parameter_count=None,
        n_bootstrap_resamples=20,
        bootstrap_seed=42,
    )
    assert {item.metric for item in bootstrap} == {"balanced_accuracy", "macro_f1"}
    assert metrics.balanced_accuracy_lcb95 == bootstrap[0].lcb95
    assert metrics.macro_f1_lcb95 == bootstrap[1].lcb95
    assert metrics.worst_fold_balanced_accuracy == 1.0
    assert metrics.parameter_count is None
    assert metrics.inference_cost["milliseconds_per_participant"] is None
    assert metrics.eligible is False
    assert metrics.exclusion_reason == "operational_measurements_not_measured"
    assert "pooled_participant_repeat" in metrics.confusion_matrices
    assert set(fold_confusion).issubset(metrics.confusion_matrices)


def test_metrics_builder_never_coerces_invalid_operational_cost_to_zero() -> None:
    predictions = _predictions()
    fold_ba, fold_confusion, fold_rosters = _fold_payload(predictions)
    metrics, _ = build_config_metrics_from_predictions_and_fold_summaries(
        config_id="cfg_invalid_ops",
        registry_role="ablation",
        predictions=predictions,
        fold_balanced_accuracies=fold_ba,
        fold_confusion_matrices=fold_confusion,
        fold_participant_rosters=fold_rosters,
        inference_cost={"milliseconds": float("nan")},
        parameter_count=-1,
        n_bootstrap_resamples=10,
    )
    assert metrics.inference_cost == {"milliseconds": None}
    assert metrics.parameter_count is None
    assert metrics.eligible is False


def test_comparison_archive_is_strict_json_and_shows_all_primary_metrics(tmp_path) -> None:
    bootstrap_ba = participant_cluster_bootstrap(
        _predictions(), n_resamples=20, seed=42
    )
    bootstrap_f1 = participant_cluster_bootstrap(
        _predictions(), metric="macro_f1", n_resamples=20, seed=42
    )
    permutation = paired_participant_permutation(
        _predictions(), _predictions(degrade=True), n_resamples=40, seed=42
    )
    configs = tuple(
        _metrics(
            index,
            role="reference" if index == 0 else "ablation",
            balanced_accuracy_lcb95=bootstrap_ba.lcb95,
            macro_f1_lcb95=bootstrap_f1.lcb95,
        )
        for index in (0, 1)
    )
    archive = ComparisonArchive(
        comparison_id="shapeformer_discovery",
        run_id="smoke",
        configs=configs,
        bootstrap_results={
            config.config_id: (bootstrap_ba, bootstrap_f1)
            for config in configs
        },
        paired_permutation_results={"cfg01_vs_cfg00": permutation},
        holm_results=holm_adjust(
            {"cfg01_vs_cfg00": permutation.two_sided_p_value},
            comparison_family="shapeformer_discovery",
            metric="balanced_accuracy",
        ),
        selections=(
            ManualFinalSelection(
                purpose="analysis",
                config_id="cfg01",
                registry_role="ablation",
                rationale="human review smoke",
            ),
        ),
        run_manifest={
            "independent_test": False,
            "fold_protocol": "subject_level_5fold_5repeat",
            "seeds": (42, 10042, 20042, 30042, 40042),
        },
    )
    target = write_comparison_archive(archive, tmp_path)
    manifest = json.loads((target / "run_manifest.json").read_text(encoding="utf-8"))
    report = (target / "comparison_report.md").read_text(encoding="utf-8")
    assert manifest["automatic_selection"] is False
    assert "Worst-fold BA" in report
    assert "ECE" in report
    assert "Inference cost" in report
    assert "绝不自动选出" in report
    assert (target / "ranking_top10.csv").is_file()
    assert verify_comparison_archive(target)["overwrite"] is False
    assert read_verified_manual_selections(target)[0]["config_id"] == "cfg01"
    (target / "unindexed.txt").write_text("tamper", encoding="utf-8")
    with pytest.raises(ValueError, match="unindexed"):
        verify_comparison_archive(target)
    (target / "unindexed.txt").unlink()
    (target / "comparison_report.md").write_text(
        report + "\ntampered\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="integrity"):
        verify_comparison_archive(target)


def test_archive_rejects_nonfinite_manifest_and_invalid_confusion_matrix() -> None:
    bootstrap_ba = participant_cluster_bootstrap(
        _predictions(), n_resamples=10, seed=42
    )
    bootstrap_f1 = participant_cluster_bootstrap(
        _predictions(), metric="macro_f1", n_resamples=10, seed=42
    )
    metric = _metrics(
        0,
        balanced_accuracy_lcb95=bootstrap_ba.lcb95,
        macro_f1_lcb95=bootstrap_f1.lcb95,
    )
    with pytest.raises(ValueError, match="NaN and infinity"):
        ComparisonArchive(
            comparison_id="c",
            run_id="r",
            configs=(metric,),
            bootstrap_results={"cfg00": (bootstrap_ba, bootstrap_f1)},
            paired_permutation_results={},
            holm_results=(),
            selections=(),
            run_manifest={
                "independent_test": False,
                "fold_protocol": "fivefold",
                "seeds": (42,),
                "bad": float("nan"),
            },
        )
    with pytest.raises(ValueError, match="confusion matrices"):
        ConfigMetrics(
            config_id="bad",
            registry_role="reference",
            participant_mean_balanced_accuracy=0.5,
            participant_mean_macro_f1=0.5,
            worst_fold_balanced_accuracy=0.4,
            balanced_accuracy_lcb95=0.3,
            macro_f1_lcb95=0.3,
            worst_class_recall=0.4,
            worst_class_f1=0.4,
            expected_calibration_error=0.1,
            variability={"sd": 0.1},
            confusion_matrices={"participant": ((1.0, float("nan")), (0.0, 1.0))},
            inference_cost={"milliseconds": 1.0},
            parameter_count=10,
        )
