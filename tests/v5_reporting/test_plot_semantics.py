"""Regression tests for the semantic V2 figure surface exposed by V5."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from ppg_frailty.reporting import plots
from ppg_frailty.v5_reporting import plots as v5_plots
from ppg_frailty.v5_reporting.registry import KNOWN_FIGURES


@pytest.fixture
def pyplot():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as value

    yield value
    value.close("all")


def test_every_v2_figure_has_a_source_and_semantic_dispatch(tmp_path, monkeypatch):
    assert set(plots.FIGURE_TABLE_SOURCES) == set(plots.STATIC_FIGURE_NAMES)
    assert all(plots.FIGURE_TABLE_SOURCES.values())
    observed: set[str] = set()

    def fake_save(directory, name, draw, pyplot, *, render_na_png=False):
        del draw, pyplot, render_na_png
        target = directory / f"{name}.png"
        target.write_bytes(b"semantic-dispatch")
        observed.add(name)
        return {
            "figure": name,
            "status": "generated",
            "path": target.relative_to(directory.parent).as_posix(),
            "reason": "",
        }

    monkeypatch.setattr(plots, "_save", fake_save)
    analysis = SimpleNamespace()
    profiles = (
        {},
        {"legacy_bridge": {}},
        {"legacy_bridge": {"design": "centered_star_v1"}},
    )
    for index, plan in enumerate(profiles):
        plots.generate_static_figures(
            SimpleNamespace(plan=plan),
            analysis,
            tmp_path / f"profile_{index}",
        )

    assert observed == set(plots.STATIC_FIGURE_NAMES)


def test_v5_dispatch_emits_exactly_all_36_registered_ids(tmp_path, monkeypatch):
    def fake_standard(collected, analysis, directory, *, modules):
        del collected, analysis
        output = []
        for name in modules:
            target = directory / f"{name}.png"
            target.write_bytes(b"standard")
            output.append(
                {
                    "figure": name,
                    "status": "generated",
                    "path": target.relative_to(directory.parent).as_posix(),
                    "reason": "",
                }
            )
        return tuple(output)

    def fake_ensemble(products, directory):
        del products
        target = directory / "ensemble_member_metrics.png"
        target.write_bytes(b"ensemble")
        return {
            "figure": "ensemble_member_metrics",
            "status": "generated",
            "path": target.relative_to(directory.parent).as_posix(),
            "reason": "",
        }

    monkeypatch.setattr(v5_plots, "generate_static_figures", fake_standard)
    monkeypatch.setattr(v5_plots, "_ensemble_member_metrics", fake_ensemble)
    names = tuple(sorted(KNOWN_FIGURES))
    statuses = v5_plots.generate_selected_figures(
        SimpleNamespace(collected=object()),
        SimpleNamespace(analysis=object(), tables={}),
        SimpleNamespace(on_missing="error"),
        names,
        tmp_path / "figures",
    )

    assert tuple(row["figure"] for row in statuses) == names
    assert {row["status"] for row in statuses} == {"generated"}
    assert {path.stem for path in (tmp_path / "figures").glob("*.png")} == set(names)


def test_heatmap_interaction_and_hierarchy_keep_matrix_semantics(pyplot):
    fold = plots._fold_heatmap(
        SimpleNamespace(
            fold_metrics=tuple(
                {
                    "case_id": case,
                    "repeat": 0,
                    "fold": fold_id,
                    "balanced_accuracy": score,
                }
                for case, fold_id, score in (
                    ("case_a", 0, 0.70),
                    ("case_a", 1, 0.75),
                    ("case_b", 0, 0.65),
                    ("case_b", 1, 0.80),
                )
            )
        ),
        pyplot,
    )
    np.testing.assert_allclose(
        fold.axes[0].images[0].get_array(),
        [[0.70, 0.75], [0.65, 0.80]],
    )
    assert "Fold × repeat" in fold.axes[0].get_title()

    cases = []
    summary = []
    for first, second, score in ((1, "a", 0.61), (1, "b", 0.66), (2, "a", 0.72), (2, "b", 0.78)):
        case_id = f"case_{first}_{second}"
        cases.append(
            {
                "case_id": case_id,
                "changed_values": {"model.depth": first, "training.optimizer": second},
            }
        )
        summary.append(
            {
                "case_id": case_id,
                "participant_mean_balanced_accuracy": score,
            }
        )
    interaction = plots._parameter_interaction(
        SimpleNamespace(
            plan={
                "axes": (
                    {"path": "model.depth"},
                    {"path": "training.optimizer"},
                )
            },
            manifest={"cases": tuple(cases)},
        ),
        SimpleNamespace(case_summary=tuple(summary)),
        pyplot,
    )
    assert interaction.axes[0].images[0].get_array().shape == (2, 2)
    assert interaction.axes[0].get_ylabel() == "model.depth"
    assert interaction.axes[0].get_xlabel() == "training.optimizer"

    coverage_rows = tuple(
        {
            "case_id": case,
            "aggregation_level": level,
            "group_label": group,
            "participant_count": count,
        }
        for level, groups in (("window", ("B", "R1")), ("file", ("B", "R1")), ("role", ("B", "R")))
        for case, count in (("case_a", 10), ("case_b", 12))
        for group in groups
    )
    hierarchy = plots._aggregation_hierarchy_coverage(
        SimpleNamespace(aggregation_hierarchy_coverage=coverage_rows),
        pyplot,
    )
    matrix_axes = [axis for axis in hierarchy.axes if axis.images]
    assert len(matrix_axes) == 3
    assert all(axis.images[0].get_array().shape[0] == 2 for axis in matrix_axes)


def test_box_roc_confusion_learning_and_calibration_semantics(pyplot):
    stability = plots._stability(
        SimpleNamespace(
            repeat_metrics=tuple(
                {
                    "case_id": case,
                    "balanced_accuracy": score,
                }
                for case, score in (
                    ("case_a", 0.70),
                    ("case_a", 0.74),
                    ("case_b", 0.62),
                    ("case_b", 0.68),
                )
            )
        ),
        pyplot,
    )
    assert stability.axes[0].get_title() == "Repeat stability"
    assert [label.get_text() for label in stability.axes[0].get_xticklabels()] == [
        "case_a",
        "case_b",
    ]
    assert len(stability.axes[0].lines) >= 2

    roc = plots._classification_roc_auc_curves(
        SimpleNamespace(
            classification_roc_curves=tuple(
                {
                    "classifier_id": "cnn",
                    "evaluation_id": "participant_outer_oof",
                    "aggregation_level": "participant",
                    "curve": "macro_average_ovr",
                    "class_label": "macro",
                    "point_index": index,
                    "false_positive_rate": x,
                    "true_positive_rate": y,
                    "roc_auc": 0.82,
                }
                for index, (x, y) in enumerate(((0.0, 0.0), (0.2, 0.7), (1.0, 1.0)))
            )
        ),
        pyplot,
    )
    assert roc.axes[0].get_xlabel() == "False-positive rate"
    assert any("AUC=0.820" in line.get_label() for line in roc.axes[0].lines)

    matrix = np.asarray(((4, 1, 0), (1, 3, 1), (0, 1, 4)))
    confusion = plots._confusion(
        SimpleNamespace(
            confusion_matrices=(
                {
                    "case_id": "cnn",
                    "class_order": (0, 1, 2),
                    "confusion_matrix": matrix.tolist(),
                },
            )
        ),
        pyplot,
    )
    np.testing.assert_array_equal(confusion.axes[0].images[0].get_array(), matrix)
    assert confusion.axes[0].get_ylabel() == "True"

    learning = plots._learning_curves(
        SimpleNamespace(
            history_rows=tuple(
                {
                    "case_id": "cnn",
                    "repeat": 0,
                    "fold": 0,
                    "epoch": epoch,
                    "training_loss": loss,
                }
                for epoch, loss in ((0, 1.0), (1, 0.7), (2, 0.5))
            )
        ),
        pyplot,
    )
    np.testing.assert_allclose(learning.axes[0].lines[0].get_ydata(), [1.0, 0.7, 0.5])
    assert "Learning curves" in learning.axes[0].get_title()

    calibration = plots._calibration(
        SimpleNamespace(
            calibration_bins=tuple(
                {
                    "case_id": "cnn",
                    "bin_index": index,
                    "mean_confidence": confidence,
                    "accuracy": accuracy,
                }
                for index, (confidence, accuracy) in enumerate(((0.2, 0.1), (0.8, 0.75)))
            )
        ),
        pyplot,
    )
    assert calibration.axes[0].get_xlabel() == "Mean confidence"
    assert calibration.axes[0].get_ylabel() == "Observed accuracy"
    assert len(calibration.axes[0].lines) == 2
