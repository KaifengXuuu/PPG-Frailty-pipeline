"""No-training contracts for quality-aware report aggregation replay."""

from __future__ import annotations

import unittest
from dataclasses import asdict, replace
from pathlib import Path

from ppg_frailty.reporting.analyze import analyze_study
from ppg_frailty.reporting.collect import CollectedStudy
from ppg_frailty.training.aggregation import (
    LINE_A_EQUAL_FILES,
    LINE_B_EQUAL_ROLE_FAMILIES,
    QUALITY_WEIGHT_SOURCE_LEGACY_WINDOW,
    aggregate_hierarchy,
)
from ppg_frailty.training.oof import OofPredictionRow


CASE_ID = "quality_weighted_case"


def _window_row(
    *,
    participant_id: str,
    label: int,
    file_id: str,
    role: str,
    probabilities: tuple[float, float, float],
    quality_score: float,
    window_index: int,
) -> OofPredictionRow:
    return OofPredictionRow(
        participant_id=participant_id,
        file_id=file_id,
        role=role,
        label=label,
        probabilities=probabilities,
        repeat=0,
        fold=0,
        split_seed=42,
        training_seed=42,
        config_hash="a" * 64,
        manifest_hash="b" * 64,
        fold_hash="c" * 64,
        preprocessing_hash="d" * 64,
        feature_hash="e" * 64,
        model_hash="f" * 64,
        representation_mode="raw",
        signal_route="direct",
        quality_score=quality_score,
        retained=True,
        level="window",
        window_id=f"{file_id}::window_{window_index}",
        class_order=(0, 1, 2),
        aggregation_rule=LINE_B_EQUAL_ROLE_FAMILIES,
    )


def _weighted_bundle() -> CollectedStudy:
    windows: list[OofPredictionRow] = []
    for label in (0, 1, 2):
        participant_id = f"P{label}"
        correct = tuple(1.0 if index == label else 0.0 for index in range(3))
        wrong_label = (label + 1) % 3
        wrong = tuple(1.0 if index == wrong_label else 0.0 for index in range(3))
        for role, quality_score, probabilities in (
            ("B", 0.9, correct),
            ("R1", 0.1, wrong),
            ("R2", 0.9, wrong),
        ):
            file_id = f"{participant_id}_{role}"
            for window_index in range(2):
                windows.append(
                    _window_row(
                        participant_id=participant_id,
                        label=label,
                        file_id=file_id,
                        role=role,
                        probabilities=probabilities,
                        quality_score=quality_score,
                        window_index=window_index,
                    )
                )
    persisted = aggregate_hierarchy(
        windows,
        balance_line=LINE_B_EQUAL_ROLE_FAMILIES,
        quality_weighted=True,
        quality_weight_source=QUALITY_WEIGHT_SOURCE_LEGACY_WINDOW,
    )
    return CollectedStudy(
        root=Path("."),
        plan={
            "execution": {"repeats": [0], "folds": [0]},
            "report": {"calibration_bins": 3},
        },
        manifest={
            "cases": [{"case_id": CASE_ID, "is_reference": True}],
            "reference_case_id": CASE_ID,
        },
        case_records=({"case_id": CASE_ID, "status": "passed"},),
        varied_parameters=(),
        controlled_parameters=(),
        cell_rows=(
            {
                "case_id": CASE_ID,
                "status": "passed",
                "repeat": 0,
                "fold": 0,
            },
        ),
        history_rows=(),
        file_oof_rows=tuple(
            {"case_id": CASE_ID, **asdict(row)} for row in persisted.file_rows
        ),
        subject_oof_rows=tuple(
            {"case_id": CASE_ID, **asdict(row)}
            for row in persisted.participant_rows
        ),
        role_oof_rows=tuple(
            {"case_id": CASE_ID, **asdict(row)} for row in persisted.role_rows
        ),
        quality_rows=(),
        trusted_config_metrics=(),
        limitations=(),
        window_oof_rows=tuple(
            {"case_id": CASE_ID, **asdict(row)} for row in windows
        ),
        resolved_aggregation_configs=(
            {
                "case_id": CASE_ID,
                "resolved_config_path": "raw/quality_weighted_case/resolved_config.yaml",
                "aggregation": {
                    "balance_line": LINE_B_EQUAL_ROLE_FAMILIES,
                    "quality_weighting": True,
                    "quality_weight_source": QUALITY_WEIGHT_SOURCE_LEGACY_WINDOW,
                },
            },
        ),
    )


class QualityWeightedReportingReplayTests(unittest.TestCase):
    def test_source_line_matches_persisted_oof_and_both_lines_keep_weights(self) -> None:
        analysis = analyze_study(_weighted_bundle())
        by_line = {
            str(row["balance_line"]): row
            for row in analysis.aggregation_line_comparison
        }

        self.assertEqual(
            set(by_line),
            {LINE_A_EQUAL_FILES, LINE_B_EQUAL_ROLE_FAMILIES},
        )
        self.assertEqual(
            {
                (row["quality_weighting"], row["quality_weight_source"])
                for row in by_line.values()
            },
            {(True, QUALITY_WEIGHT_SOURCE_LEGACY_WINDOW)},
        )
        self.assertTrue(
            str(
                by_line[LINE_B_EQUAL_ROLE_FAMILIES]["source_replay_validation"]
            ).startswith("exact_match")
        )
        self.assertAlmostEqual(
            float(
                by_line[LINE_B_EQUAL_ROLE_FAMILIES][
                    "participant_mean_balanced_accuracy"
                ]
            ),
            1.0,
        )
        self.assertAlmostEqual(
            float(
                by_line[LINE_A_EQUAL_FILES][
                    "participant_mean_balanced_accuracy"
                ]
            ),
            0.0,
        )

    def test_runtime_line_overrides_resolved_catalog_default(self) -> None:
        bundle = _weighted_bundle()
        source_windows = tuple(
            replace(
                OofPredictionRow(
                    **{
                        key: value
                        for key, value in row.items()
                        if key != "case_id"
                    }
                ),
                aggregation_rule=LINE_A_EQUAL_FILES,
            )
            for row in bundle.window_oof_rows
        )
        persisted = aggregate_hierarchy(
            source_windows,
            balance_line=LINE_A_EQUAL_FILES,
            quality_weighted=True,
            quality_weight_source=QUALITY_WEIGHT_SOURCE_LEGACY_WINDOW,
        )
        runtime_line_a = replace(
            bundle,
            window_oof_rows=tuple(
                {"case_id": CASE_ID, **asdict(row)} for row in source_windows
            ),
            file_oof_rows=tuple(
                {"case_id": CASE_ID, **asdict(row)}
                for row in persisted.file_rows
            ),
            subject_oof_rows=tuple(
                {"case_id": CASE_ID, **asdict(row)}
                for row in persisted.participant_rows
            ),
            role_oof_rows=tuple(
                {"case_id": CASE_ID, **asdict(row)}
                for row in persisted.role_rows
            ),
        )

        analysis = analyze_study(runtime_line_a)
        by_line = {
            str(row["balance_line"]): row
            for row in analysis.aggregation_line_comparison
        }
        self.assertEqual(
            set(by_line),
            {LINE_A_EQUAL_FILES, LINE_B_EQUAL_ROLE_FAMILIES},
        )
        self.assertEqual(
            {
                str(row["aggregation_view"])
                for row in analysis.aggregation_view_comparison
            },
            {
                "window_balanced_to_participant",
                LINE_A_EQUAL_FILES,
                LINE_B_EQUAL_ROLE_FAMILIES,
            },
        )
        for row in by_line.values():
            self.assertEqual(row["declared_source_line"], LINE_A_EQUAL_FILES)
            self.assertEqual(
                row["resolved_config_balance_line"],
                LINE_B_EQUAL_ROLE_FAMILIES,
            )
            self.assertEqual(
                row["source_line_provenance"],
                "selected_file_oof_effective_line",
            )
        self.assertTrue(by_line[LINE_A_EQUAL_FILES]["primary_ranking_eligible"])
        self.assertFalse(
            by_line[LINE_B_EQUAL_ROLE_FAMILIES]["primary_ranking_eligible"]
        )
        self.assertTrue(
            str(by_line[LINE_A_EQUAL_FILES]["source_replay_validation"]).startswith(
                "exact_match"
            )
        )

    def test_inconsistent_effective_source_lines_remain_fail_closed(self) -> None:
        bundle = _weighted_bundle()
        rows = [dict(row) for row in bundle.file_oof_rows]
        rows[0]["aggregation_rule"] = LINE_A_EQUAL_FILES
        rejected = analyze_study(replace(bundle, file_oof_rows=tuple(rows)))

        self.assertEqual(rejected.aggregation_line_comparison, ())
        self.assertEqual(
            {
                str(row["aggregation_view"])
                for row in rejected.aggregation_view_comparison
            },
            {"window_balanced_to_participant"},
        )
        self.assertTrue(
            any(
                "multiple or unsupported effective source lines" in note
                for note in rejected.notes
            )
        )

    def test_missing_config_or_row_weight_suppresses_replay(self) -> None:
        bundle = _weighted_bundle()

        missing_config = analyze_study(
            replace(bundle, resolved_aggregation_configs=())
        )
        self.assertEqual(missing_config.aggregation_line_comparison, ())
        self.assertTrue(
            any(
                "requires exactly one persisted resolved config" in note
                for note in missing_config.notes
            )
        )

        file_rows = [dict(row) for row in bundle.file_oof_rows]
        file_rows[0].pop("quality_score")
        missing_weight = analyze_study(
            replace(bundle, file_oof_rows=tuple(file_rows))
        )
        self.assertEqual(missing_weight.aggregation_line_comparison, ())
        self.assertTrue(any("quality_score" in note for note in missing_weight.notes))

    def test_replay_does_not_silently_drop_the_declared_modifier(self) -> None:
        bundle = _weighted_bundle()
        unweighted_config = (
            {
                "case_id": CASE_ID,
                "resolved_config_path": "synthetic/unweighted.yaml",
                "aggregation": {
                    "balance_line": LINE_B_EQUAL_ROLE_FAMILIES,
                    "quality_weighting": False,
                    "quality_weight_source": "none",
                },
            },
        )
        analysis = analyze_study(
            replace(bundle, resolved_aggregation_configs=unweighted_config)
        )

        self.assertEqual(analysis.aggregation_line_comparison, ())
        self.assertTrue(
            any("source-line replay probability mismatch" in note for note in analysis.notes)
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
