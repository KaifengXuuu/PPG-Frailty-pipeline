from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from ppg_frailty.reporting.plots import STATIC_FIGURE_NAMES
from ppg_frailty.reporting.specialized import (
    STAGE5_FIGURE_SOURCES,
    generate_hyperparameter_report,
    generate_motion_peak_report,
)


EXPECTED_STAGE5_FIGURES = {
    "motion_detector_metrics",
    "motion_internal_confusion_matrix",
    "motion_internal_file_confusion_matrix",
    "motion_ptt_confusion_matrix",
    "motion_ptt_file_confusion_matrix",
    "motion_ptt_training_oof_confusion_matrix",
    "motion_ptt_training_oof_file_confusion_matrix",
    "motion_internal_reverse_confusion_matrix",
    "motion_internal_reverse_file_confusion_matrix",
    "frailty29_trained_window_score_distribution",
    "frailty29_trained_file_score_distribution",
    "ptt22_trained_window_score_distribution",
    "ptt22_trained_file_score_distribution",
    "frailty29_trained_window_prediction_tsne",
    "frailty29_trained_file_prediction_tsne",
    "ptt22_trained_window_prediction_tsne",
    "ptt22_trained_file_prediction_tsne",
    "frailty29_trained_window_roc_auc_curve",
    "frailty29_trained_file_roc_auc_curve",
    "ptt22_trained_window_roc_auc_curve",
    "ptt22_trained_file_roc_auc_curve",
    "motion_training_learning_curves",
    "denoiser_interval_rmse",
    "denoiser_beat_f1",
    "denoiser_beat_sensitivity",
    "denoiser_beat_ppv",
    "denoiser_runtime",
    "static_peak_detector_f1",
    "static_peak_detector_sensitivity",
    "static_peak_detector_ppv",
    "static_peak_detector_interval_rmse",
    "static_peak_detector_runtime",
}


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _prediction_rows(target: str, shift: float) -> list[dict[str, object]]:
    rows = []
    for participant_index in range(2):
        participant = f"{target}-{participant_index}"
        for label in (0, 1):
            probability = (0.12 + 0.08 * participant_index if label == 0 else 0.78 + 0.08 * participant_index) + shift
            rows.append(
                {
                    "repeat_index": 0,
                    "fold_index": 0,
                    "window_id": f"{participant}-{label}",
                    "participant_id": participant,
                    "file_id": f"{participant}-{label}",
                    "activity_label": label,
                    "p_active": probability,
                    "threshold": 0.5,
                    "predicted_activity": int(probability >= 0.5),
                }
            )
    return rows


def _stage5_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "stage5"
    stages = {
        "internal_motion_oof": ("motion_internal", "motion_window_oof.parquet", "frailty29", 0.0),
        "ptt_motion_external": ("motion_external", "motion_ptt_window_predictions.parquet", "ptt22", 0.0),
        "ptt_motion_training_ablation": ("motion_ptt_training", "motion_ptt_training_oof.parquet", "ptt22", -0.02),
        "frailty29_reverse_evaluation": ("motion_reverse", "motion_internal_reverse_predictions.parquet", "frailty29", -0.02),
    }
    manifest_stages = {}
    for stage, (directory, filename, target, shift) in stages.items():
        path = root / directory
        path.mkdir(parents=True)
        pd.DataFrame(_prediction_rows(target, shift)).to_parquet(path / filename, index=False)
        if stage in {"internal_motion_oof", "ptt_motion_training_ablation"}:
            _json(
                path / "repeat_0/fold_0/motion_training_history.json",
                {
                    "repeat_index": 0,
                    "fold_index": 0,
                    "rows": [
                        {"epoch": 1, "training_loss": 0.8, "training_balanced_accuracy": 0.6},
                        {"epoch": 2, "training_loss": 0.5, "training_balanced_accuracy": 0.8},
                    ],
                },
            )
        manifest_stages[stage] = {"status": "passed", "artifact_dir": directory}
    denoiser_rows = []
    for participant in ("P0", "P1"):
        for activity in ("static", "dynamic"):
            for channel in ("RED", "IR"):
                for algorithm, correct in (("identity", 8), ("reducer", 9)):
                    denoiser_rows.append(
                        {
                            "participant_id": participant,
                            "record_id": f"{participant}-{activity}",
                            "segment_start_s": 0.0,
                            "activity_group": activity,
                            "channel": channel,
                            "algorithm_or_reducer": algorithm,
                            "status": "passed",
                            "true_positives": correct,
                            "false_positives": 10 - correct,
                            "false_negatives": 10 - correct,
                            "matched_interval_count": 2,
                            "ibi_ppi_rmse_ms": 12.0 - correct,
                            "runtime_s": 0.1,
                        }
                    )
    denoiser = root / "denoiser"
    _json(denoiser / "denoiser_benchmark.json", {"rows": denoiser_rows})
    manifest_stages["ptt_denoiser_benchmark"] = {
        "status": "passed",
        "artifact_dir": "denoiser",
    }
    _json(
        root / "study_manifest.json",
        {
            "study_id": "stage5",
            "study_type": "stage5_pre_motion_ptt",
            "stages": manifest_stages,
        },
    )
    (root / "resolved_plan.yaml").write_text(
        yaml.safe_dump(
            {
                "report": {
                    "classification_tsne_perplexity": 2.0,
                    "classification_tsne_max_samples": 20,
                    "participant_cluster_bootstrap_resamples": 20,
                    "participant_paired_permutation_resamples": 20,
                }
            }
        ),
        encoding="utf-8",
    )
    return root


def test_stage5_inventory_is_exact_and_all_motion_figures_render(tmp_path: Path) -> None:
    assert set(STAGE5_FIGURE_SOURCES) == EXPECTED_STAGE5_FIGURES
    source = _stage5_fixture(tmp_path)
    target = tmp_path / "report"

    result = generate_motion_peak_report(source, output_dir=target)
    statuses = json.loads((target / "tables/reporter_output_status.json").read_text())
    motion = [row for row in statuses if not row["figure"].startswith("static_peak_detector_")]

    assert len(motion) == 27
    assert {row["status"] for row in motion} == {"generated"}
    assert all((target / row["path"]).is_file() for row in motion)
    assert {
        "motion_detector_training_source_inference",
        "motion_detector_participant_metrics_raw",
        "denoiser_paired_inference",
        "motion_training_history",
        "v2_v5_specialized_inventory",
    } <= set(result["tables"])


def test_hyper_report_declares_all_ordinary_phase_figure_ids(
    tmp_path: Path, monkeypatch
) -> None:
    import ppg_frailty.reporting.specialized as specialized

    source = tmp_path / "hyper"
    phase = source / "phase"
    phase.mkdir(parents=True)
    (source / "tables").mkdir()
    (source / "study_plan.yaml").write_text(
        "study:\n  study_id: hyper\nresource:\n  ranking_metric: balanced_accuracy\n",
        encoding="utf-8",
    )
    _json(
        source / "study_manifest.json",
        {
            "ranking_tables": ["full_cv_ranking"],
            "phase_directories": {"full_cv": "phase"},
            "selected_case_id": "candidate",
        },
    )
    _json(
        source / "tables/full_cv_ranking.json",
        [{"case_id": "candidate", "balanced_accuracy_mean": 0.8, "balanced_accuracy_sd": 0.0}],
    )
    _json(phase / "study_manifest.json", {"cases": [{"case_id": "candidate"}]})

    def fake_phase(source_path, target, *, preferred_reference):
        target.mkdir(parents=True)
        statuses = [
            {"figure": name, "status": "N/A", "path": f"figures/{name}.NA.txt"}
            for name in STATIC_FIGURE_NAMES
        ]
        for row in statuses:
            path = target / row["path"]
            path.parent.mkdir(exist_ok=True)
            path.write_text("N/A\n", encoding="utf-8")
        return {"figure_status": statuses}, {"case_summary": ()}

    monkeypatch.setattr(specialized, "_phase_report", fake_phase)
    result = generate_hyperparameter_report(source, output_dir=tmp_path / "hyper-report")

    assert result["ordinary_phase_figure_count"] == 35
    assert tuple(result["ordinary_phase_figure_ids"]) == STATIC_FIGURE_NAMES
    assert result["phase_reports"] == {"full_cv": "phases/full_cv/report_manifest.json"}
    pairs = json.loads((tmp_path / "hyper-report/tables/table_figure_pairs.json").read_text())
    leaderboard = next(row for row in pairs if row["figure"] == "leaderboard")
    assert leaderboard["table"] == "predictive_leaderboard"
