from __future__ import annotations

import argparse
import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


BASE_CONFIG_COLUMNS = [
    "model",
    "resolved_model",
    "extra_input",
    "cnn_epochs",
    "cnn_patience",
    "window_sec",
    "hop_sec",
    "overlap_pct",
    "max_windows_fraction",
]
OVERFITTING_CONFIG_COLUMNS = [
    "overfit_stage",
    "overfit_config_id",
    "overfit_config_name",
    "stage1_screen_group",
    "stage1_regularization_factor",
    "stage1_regularization_value",
    "cnn_lr",
    "cnn_weight_decay",
    "cnn_dropout",
    "cnn_label_smoothing",
    "dynamic_data_mode",
    "stage2_fixed_epoch",
]
CONFIG_COLUMNS = BASE_CONFIG_COLUMNS + OVERFITTING_CONFIG_COLUMNS

DEFAULT_SWEEP_DIRS = ["results_frailty3/20260527_1320_cnn_inceptionTime"]
DEFAULT_MODELS = ["cnn", "inceptiontime"]
MODEL_ALIASES = {
    "cnn1d": "cnn",
    "cnn_1d": "cnn",
    "1d-cnn": "cnn",
    "1d_cnn": "cnn",
    "inception_time": "inceptiontime",
    "inception-time": "inceptiontime",
    "shapeformer-new-discovery": "shapeformer_pisd",
    "shapeformer_new_discovery": "shapeformer_pisd",
}
CLASS_KEYS = ["pre_frail", "robust_non_frail", "young"]
CLASS_NAMES = {
    "pre_frail": "Pre-Frail",
    "robust_non_frail": "Robust/Non-Frail",
    "young": "Young",
}
RANK_COLUMNS = [
    "subject_balanced_accuracy_mean",
    "subject_macro_f1_mean",
    "subject_balanced_accuracy_ci95_low",
    "subject_macro_f1_ci95_low",
    "worst_class_recall_mean",
    "worst_class_f1_mean",
    "subject_balanced_accuracy_std",
]


def normalize_user_path(value: str) -> Path:
    text = str(value).strip().replace("\\", "/")
    if text.startswith("home/"):
        text = "/" + text
    return Path(text).expanduser()


def parse_csv_list(value: str) -> List[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_repeated_csv_list(values: Optional[Sequence[str]]) -> List[str]:
    out: List[str] = []
    if not values:
        return out
    for value in values:
        out.extend(parse_csv_list(value))
    return out


def canonical_model_name(value: object) -> str:
    text = str(value).strip().lower().replace(" ", "_")
    return MODEL_ALIASES.get(text, text)


def unique_output_dir(root: Path, model_names: Sequence[str], prefix: str = "") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    tag = "_".join(sanitize_token(name) for name in model_names)
    if prefix:
        tag = f"{sanitize_token(prefix)}_{tag}"
    base = root / f"{datetime.now().strftime('%Y%m%d_%H%M')}_{tag}"
    if not base.exists():
        base.mkdir(parents=True)
        return base
    for idx in range(2, 1000):
        candidate = root / f"{base.name}_{idx:02d}"
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate
    raise RuntimeError(f"Could not create a unique output directory under {root}")


def sanitize_token(value: object) -> str:
    text = str(value).strip().replace(" ", "_")
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in text)


def format_number(value: object, digits: int = 6) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, (float, np.floating)):
        text = f"{float(value):.{digits}g}"
        return text.replace(".", "p").replace("-", "m")
    return sanitize_token(value)


def make_config_key(row: pd.Series) -> str:
    parts = []
    for col in CONFIG_COLUMNS:
        parts.append(f"{col}={format_number(row.get(col))}")
    return "|".join(parts)


def read_manifest(sweep_dir: Path) -> Dict[str, object]:
    manifest_path = sweep_dir / "sweep_manifest.json"
    if not manifest_path.exists():
        manifest_path = sweep_dir / "overfitting_manifest.json"
    if not manifest_path.exists():
        return {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def infer_expected_repeats(manifest: Dict[str, object], runs: pd.DataFrame) -> Optional[int]:
    if manifest:
        if "repeats" in manifest:
            repeats = safe_float(manifest.get("repeats"))
            if math.isfinite(repeats) and repeats > 0:
                return int(round(repeats))
        total_groups = manifest.get("total_groups")
        total_runs = manifest.get("total_requested_runs")
        if total_groups and total_runs:
            repeats = float(total_runs) / float(total_groups)
            if repeats > 0 and abs(repeats - round(repeats)) < 1e-6:
                return int(round(repeats))
    if "repeat" in runs.columns and not runs.empty:
        repeats = pd.to_numeric(runs["repeat"], errors="coerce").dropna()
        if not repeats.empty:
            return int(repeats.max())
    return None


def resolve_artifact_path(raw_value: object, sweep_dir: Path, fallback_subdir: str) -> str:
    if pd.isna(raw_value) or str(raw_value).strip() == "":
        return ""
    raw_text = str(raw_value).strip().replace("\\", "/")
    candidates = []
    path = Path(raw_text)
    candidates.append(path)
    if not path.is_absolute():
        candidates.append(Path.cwd() / path)
    parts = path.parts
    if len(parts) >= 3 and parts[0] == "results_frailty3" and parts[1] == sweep_dir.name:
        candidates.append(sweep_dir.joinpath(*parts[2:]))
    candidates.append(sweep_dir / fallback_subdir / path.name)
    candidates.append(sweep_dir / path.name)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    fallback_root = sweep_dir / fallback_subdir
    if fallback_root.exists() and path.name:
        for candidate in fallback_root.rglob(path.name):
            if candidate.exists():
                return str(candidate.resolve())
    return raw_text


def load_json(path: str) -> Optional[Dict[str, object]]:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.exists():
        return None
    try:
        with candidate.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def find_class_report(report: Dict[str, object], class_key: str) -> Dict[str, float]:
    subject_report = report.get("subject_classification_report") or {}
    if class_key in subject_report:
        return subject_report.get(class_key) or {}
    wanted = class_key.replace("_", "").lower()
    for key, value in subject_report.items():
        normalized = str(key).replace("-", "").replace("/", "").replace(" ", "").replace("_", "").lower()
        if normalized == wanted:
            return value or {}
    return {}


def extract_report_metrics(report_path: str) -> Dict[str, object]:
    report = load_json(report_path)
    metrics: Dict[str, object] = {
        "report_found": bool(report),
        "subject_confusion_matrix_json": "",
    }
    for class_key in CLASS_KEYS:
        for metric in ["precision", "recall", "f1-score", "support"]:
            out_name = metric.replace("f1-score", "f1").replace("-", "_")
            metrics[f"{class_key}_{out_name}"] = np.nan
    if not report:
        return metrics

    class_recalls: List[float] = []
    class_f1s: List[float] = []
    for class_key in CLASS_KEYS:
        class_report = find_class_report(report, class_key)
        precision = safe_float(class_report.get("precision"))
        recall = safe_float(class_report.get("recall"))
        f1 = safe_float(class_report.get("f1-score"))
        support = safe_float(class_report.get("support"))
        metrics[f"{class_key}_precision"] = precision
        metrics[f"{class_key}_recall"] = recall
        metrics[f"{class_key}_f1"] = f1
        metrics[f"{class_key}_support"] = support
        if math.isfinite(recall):
            class_recalls.append(recall)
        if math.isfinite(f1):
            class_f1s.append(f1)

    matrix = report.get("subject_confusion_matrix")
    if matrix is not None:
        metrics["subject_confusion_matrix_json"] = json.dumps(matrix)
    metrics["worst_class_recall_run"] = min(class_recalls) if class_recalls else np.nan
    metrics["worst_class_f1_run"] = min(class_f1s) if class_f1s else np.nan
    return metrics


def safe_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return np.nan
    return out if math.isfinite(out) else np.nan


def t_critical_975(n: int) -> float:
    if n <= 1:
        return np.nan
    df = n - 1
    table = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        21: 2.080,
        22: 2.074,
        23: 2.069,
        24: 2.064,
        25: 2.060,
        26: 2.056,
        27: 2.052,
        28: 2.048,
        29: 2.045,
        30: 2.042,
    }
    if df in table:
        return table[df]
    if df <= 40:
        return 2.021
    if df <= 60:
        return 2.000
    if df <= 120:
        return 1.980
    return 1.960


def mean_std_ci(values: pd.Series) -> Tuple[float, float, float, float, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    n = len(numeric)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    mean = float(numeric.mean())
    std = float(numeric.std(ddof=1)) if n > 1 else 0.0
    if n > 1:
        margin = t_critical_975(n) * std / math.sqrt(n)
        return mean, std, mean - margin, mean + margin, margin
    return mean, std, np.nan, np.nan, np.nan


def prepare_runs(sweep_dir: Path, models: Sequence[str]) -> Tuple[pd.DataFrame, Dict[str, object], int]:
    runs_path = sweep_dir / "sweep_runs.csv"
    source_kind = "sweep"
    if not runs_path.exists():
        runs_path = sweep_dir / "overfitting_runs.csv"
        source_kind = "overfitting"
    if not runs_path.exists():
        raise FileNotFoundError(f"Missing sweep_runs.csv or overfitting_runs.csv: {sweep_dir}")
    runs = pd.read_csv(runs_path)
    manifest = read_manifest(sweep_dir)
    expected_repeats = infer_expected_repeats(manifest, runs)
    if expected_repeats is None:
        expected_repeats = 1

    runs["source_sweep_dir"] = str(sweep_dir)
    runs["source_sweep_name"] = sweep_dir.name
    runs["source_sweep_kind"] = source_kind
    runs["source_expected_repeats"] = int(expected_repeats)

    wanted_models = {canonical_model_name(model) for model in models}
    model_values = runs["model"].map(canonical_model_name)
    if "resolved_model" in runs.columns:
        resolved_values = runs["resolved_model"].map(canonical_model_name)
    else:
        resolved_values = pd.Series("", index=runs.index)
    model_mask = model_values.isin(wanted_models) | resolved_values.isin(wanted_models)
    runs = runs[model_mask].copy()
    if runs.empty:
        runs["config_key"] = pd.Series(dtype=str)
        runs["expected_repeats"] = int(expected_repeats)
        return runs, manifest, int(expected_repeats)

    if "status" not in runs.columns:
        runs["status"] = "ok"
    runs["status"] = runs["status"].astype(str).str.lower()
    runs["is_ok"] = runs["status"].eq("ok")
    runs["report_path_resolved"] = runs["report_path"].apply(lambda value: resolve_artifact_path(value, sweep_dir, "reports"))
    runs["learning_curve_csv_resolved"] = runs["learning_curve_csv"].apply(
        lambda value: resolve_artifact_path(value, sweep_dir, "learning_curves")
    )
    runs["learning_curve_png_resolved"] = runs["learning_curve_png"].apply(
        lambda value: resolve_artifact_path(value, sweep_dir, "learning_curves")
    )
    for col in BASE_CONFIG_COLUMNS:
        if col not in runs.columns:
            raise KeyError(f"Missing required config column in {runs_path.name}: {col}")
    for col in CONFIG_COLUMNS:
        if col not in runs.columns:
            runs[col] = ""
    for col in [
        "cnn_epochs",
        "cnn_patience",
        "window_sec",
        "hop_sec",
        "overlap_pct",
        "max_windows_fraction",
        "cnn_lr",
        "cnn_weight_decay",
        "cnn_dropout",
        "cnn_label_smoothing",
        "stage2_fixed_epoch",
        "subject_balanced_accuracy",
        "subject_macro_f1",
        "file_balanced_accuracy",
        "file_macro_f1",
        "window_balanced_accuracy",
        "window_macro_f1",
        "duration_sec",
        "n_windows",
        "n_extra_features",
        "repeat",
        "seed",
    ]:
        if col in runs.columns:
            runs[col] = pd.to_numeric(runs[col], errors="coerce")
    for col in ["window_sec", "hop_sec", "overlap_pct", "max_windows_fraction"]:
        if col in runs.columns:
            runs[col] = runs[col].round(6)

    extracted_rows = []
    for _, row in runs.iterrows():
        if bool(row.get("is_ok")):
            extracted_rows.append(extract_report_metrics(str(row.get("report_path_resolved", ""))))
        else:
            extracted_rows.append(extract_report_metrics(""))
    extracted = pd.DataFrame(extracted_rows, index=runs.index)
    for col in extracted.columns:
        if col in runs.columns:
            runs[col] = runs[col].combine_first(extracted[col])
        else:
            runs[col] = extracted[col]
    runs["config_key"] = runs.apply(make_config_key, axis=1)
    runs["expected_repeats"] = int(expected_repeats)
    reference_repeats = safe_float(manifest.get("reference_repeats")) if manifest else np.nan
    if math.isfinite(reference_repeats) and reference_repeats > 0:
        reference_mask = pd.Series(False, index=runs.index)
        if "is_reference" in runs.columns:
            reference_mask = reference_mask | runs["is_reference"].astype(str).str.lower().isin({"true", "1", "yes"})
        if "overfit_stage" in runs.columns:
            reference_mask = reference_mask | runs["overfit_stage"].astype(str).str.lower().isin({"reference", "fixed_reference"})
        if "overfit_config_id" in runs.columns:
            reference_mask = reference_mask | runs["overfit_config_id"].astype(str).str.startswith("ref_")
        runs.loc[reference_mask, "source_expected_repeats"] = int(round(reference_repeats))
        runs.loc[reference_mask, "expected_repeats"] = int(round(reference_repeats))
    return runs, manifest, int(expected_repeats)


def aggregate_config_summary(clean_runs: pd.DataFrame, default_expected_repeats: int) -> pd.DataFrame:
    summary_rows: List[Dict[str, object]] = []
    grouped = clean_runs.groupby(CONFIG_COLUMNS, dropna=False, sort=False)
    for config_values, group in grouped:
        ok = group[group["is_ok"]].copy()
        row = {col: value for col, value in zip(CONFIG_COLUMNS, config_values)}
        row["config_key"] = make_config_key(pd.Series(row))
        if "source_sweep_name" in group:
            source_names = sorted(group["source_sweep_name"].dropna().astype(str).unique())
            row["source_sweep_names"] = ",".join(source_names)
            row["source_sweep_count"] = int(len(source_names))
        else:
            row["source_sweep_names"] = ""
            row["source_sweep_count"] = 0
        if "source_sweep_kind" in group:
            source_kinds = sorted(group["source_sweep_kind"].dropna().astype(str).unique())
            row["source_sweep_kinds"] = ",".join(source_kinds)
        else:
            row["source_sweep_kinds"] = ""
        row["n_rows_total"] = int(len(group))
        if "source_sweep_dir" in group and "source_expected_repeats" in group:
            source_expected = (
                group[["source_sweep_dir", "source_expected_repeats"]]
                .drop_duplicates()
                ["source_expected_repeats"]
                .pipe(pd.to_numeric, errors="coerce")
                .dropna()
            )
            expected_repeats = int(source_expected.sum()) if len(source_expected) else int(default_expected_repeats)
        else:
            expected_repeats = int(default_expected_repeats)
        if "source_sweep_dir" in ok and "repeat" in ok:
            row["n_repeats_done"] = int(ok.groupby("source_sweep_dir")["repeat"].nunique().sum())
        elif "repeat" in ok:
            row["n_repeats_done"] = int(ok["repeat"].nunique())
        else:
            row["n_repeats_done"] = int(len(ok))
        row["n_repeats_expected"] = int(expected_repeats)
        row["failed_count"] = int((~group["is_ok"]).sum())
        row["missing_repeats"] = max(0, int(expected_repeats) - int(row["n_repeats_done"]))
        row["is_complete"] = bool(row["n_repeats_done"] >= expected_repeats and row["failed_count"] == 0)

        for metric in [
            "subject_balanced_accuracy",
            "subject_macro_f1",
            "file_balanced_accuracy",
            "file_macro_f1",
            "window_balanced_accuracy",
            "window_macro_f1",
            "best_val_loss",
            "best_val_accuracy",
            "best_epoch",
            "validation_subject_balanced_accuracy",
            "validation_subject_macro_f1",
            "best_epoch_train_balanced_accuracy",
            "best_epoch_val_balanced_accuracy",
            "train_val_balanced_accuracy_gap",
            "best_epoch_train_loss",
            "best_epoch_val_loss",
            "val_train_loss_gap",
            "duration_sec",
            "n_windows",
            "n_extra_features",
        ]:
            if metric not in ok.columns:
                continue
            mean, std, ci_low, ci_high, ci_margin = mean_std_ci(ok[metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95_low"] = ci_low
            row[f"{metric}_ci95_high"] = ci_high
            row[f"{metric}_ci95_margin"] = ci_margin
            row[f"{metric}_min"] = safe_float(pd.to_numeric(ok[metric], errors="coerce").min()) if len(ok) else np.nan
            row[f"{metric}_max"] = safe_float(pd.to_numeric(ok[metric], errors="coerce").max()) if len(ok) else np.nan

        class_recall_means = []
        class_f1_means = []
        for class_key in CLASS_KEYS:
            for metric in ["precision", "recall", "f1", "support"]:
                col = f"{class_key}_{metric}"
                if col not in ok.columns:
                    continue
                mean, std, ci_low, ci_high, ci_margin = mean_std_ci(ok[col])
                row[f"{col}_mean"] = mean
                row[f"{col}_std"] = std
                row[f"{col}_ci95_low"] = ci_low
                row[f"{col}_ci95_high"] = ci_high
                if metric == "recall" and math.isfinite(mean):
                    class_recall_means.append(mean)
                if metric == "f1" and math.isfinite(mean):
                    class_f1_means.append(mean)

        row["worst_class_recall_mean"] = min(class_recall_means) if class_recall_means else np.nan
        row["worst_class_f1_mean"] = min(class_f1_means) if class_f1_means else np.nan
        row["worst_class_recall_run_mean"] = mean_std_ci(ok.get("worst_class_recall_run", pd.Series(dtype=float)))[0]
        row["worst_class_f1_run_mean"] = mean_std_ci(ok.get("worst_class_f1_run", pd.Series(dtype=float)))[0]
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    return sort_configs(summary)


def sort_configs(summary: pd.DataFrame) -> pd.DataFrame:
    summary = summary.drop(columns=["rank_all_configs"], errors="ignore")
    for col in RANK_COLUMNS:
        if col not in summary.columns:
            summary[col] = np.nan
    out = summary.sort_values(
        by=RANK_COLUMNS,
        ascending=[False, False, False, False, False, False, True],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)
    out.insert(0, "rank_all_configs", np.arange(1, len(out) + 1))
    return out


def has_overfitting_data(frame: pd.DataFrame) -> bool:
    if frame.empty or "overfit_config_id" not in frame.columns:
        return False
    values = frame["overfit_config_id"].astype(str).str.strip()
    values = values[values.ne("") & values.str.lower().ne("nan")]
    return not values.empty


def build_leaderboard(summary: pd.DataFrame, top_k: int, include_incomplete: bool) -> pd.DataFrame:
    if include_incomplete:
        base = summary.copy()
    else:
        base = summary[summary["is_complete"]].copy()
    base = sort_configs(base)
    base = base.head(int(top_k)).copy()
    if "rank_all_configs" in base.columns:
        base = base.drop(columns=["rank_all_configs"])
    base.insert(0, "rank", np.arange(1, len(base) + 1))
    return base


def build_class_level_summary(summary: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for _, config in summary.iterrows():
        for class_key in CLASS_KEYS:
            row = {
                "config_key": config["config_key"],
                "model": config["model"],
                "resolved_model": config["resolved_model"],
                "source_sweep_names": config.get("source_sweep_names", ""),
                "extra_input": config["extra_input"],
                "cnn_epochs": config["cnn_epochs"],
                "cnn_patience": config["cnn_patience"],
                "window_sec": config["window_sec"],
                "hop_sec": config["hop_sec"],
                "overlap_pct": config["overlap_pct"],
                "max_windows_fraction": config["max_windows_fraction"],
                "class_key": class_key,
                "class_name": CLASS_NAMES[class_key],
                "is_complete": config["is_complete"],
                "n_repeats_done": config["n_repeats_done"],
            }
            for col in OVERFITTING_CONFIG_COLUMNS:
                if col in config:
                    row[col] = config.get(col, "")
            for metric in ["precision", "recall", "f1", "support"]:
                for stat in ["mean", "std", "ci95_low", "ci95_high"]:
                    source = f"{class_key}_{metric}_{stat}"
                    row[f"{metric}_{stat}"] = config.get(source, np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


def matrix_from_json(value: object) -> Optional[np.ndarray]:
    if pd.isna(value) or not str(value).strip():
        return None
    try:
        matrix = np.asarray(json.loads(str(value)), dtype=float)
    except Exception:
        return None
    if matrix.shape != (len(CLASS_KEYS), len(CLASS_KEYS)):
        return None
    return matrix


def aggregate_confusion_matrices(clean_runs: pd.DataFrame, top_configs: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for _, config in top_configs.iterrows():
        config_key = config["config_key"]
        matches = clean_runs[(clean_runs["config_key"] == config_key) & (clean_runs["is_ok"])]
        total = np.zeros((len(CLASS_KEYS), len(CLASS_KEYS)), dtype=float)
        n_matrices = 0
        for _, run in matches.iterrows():
            matrix = matrix_from_json(run.get("subject_confusion_matrix_json"))
            if matrix is None:
                continue
            total += matrix
            n_matrices += 1
        row_sums = total.sum(axis=1, keepdims=True)
        normalized = np.divide(total, row_sums, out=np.zeros_like(total), where=row_sums > 0)
        for i, true_key in enumerate(CLASS_KEYS):
            for j, pred_key in enumerate(CLASS_KEYS):
                rows.append(
                    {
                        "rank": int(config["rank"]),
                        "config_key": config_key,
                        "model": config["model"],
                        "resolved_model": config.get("resolved_model", ""),
                        "extra_input": config["extra_input"],
                        "overfit_config_id": config.get("overfit_config_id", ""),
                        "overfit_config_name": config.get("overfit_config_name", ""),
                        "stage1_regularization_factor": config.get("stage1_regularization_factor", ""),
                        "stage1_regularization_value": config.get("stage1_regularization_value", ""),
                        "cnn_epochs": config.get("cnn_epochs", ""),
                        "window_sec": config["window_sec"],
                        "overlap_pct": config["overlap_pct"],
                        "cnn_patience": config["cnn_patience"],
                        "cnn_lr": config.get("cnn_lr", ""),
                        "cnn_weight_decay": config.get("cnn_weight_decay", ""),
                        "cnn_dropout": config.get("cnn_dropout", ""),
                        "cnn_label_smoothing": config.get("cnn_label_smoothing", ""),
                        "max_windows_fraction": config.get("max_windows_fraction", ""),
                        "dynamic_data_mode": config.get("dynamic_data_mode", ""),
                        "n_matrices": n_matrices,
                        "true_class": CLASS_NAMES[true_key],
                        "pred_class": CLASS_NAMES[pred_key],
                        "count": int(total[i, j]),
                        "row_normalized": float(normalized[i, j]),
                    }
                )
    return pd.DataFrame(rows)


def write_top_confusion_csvs(confusion_long: pd.DataFrame, out_dir: Path) -> None:
    matrix_dir = out_dir / "top_confusion_matrices"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    for rank, sub in confusion_long.groupby("rank", sort=True):
        count_matrix = sub.pivot(index="true_class", columns="pred_class", values="count").reindex(
            index=[CLASS_NAMES[key] for key in CLASS_KEYS],
            columns=[CLASS_NAMES[key] for key in CLASS_KEYS],
        )
        norm_matrix = sub.pivot(index="true_class", columns="pred_class", values="row_normalized").reindex(
            index=[CLASS_NAMES[key] for key in CLASS_KEYS],
            columns=[CLASS_NAMES[key] for key in CLASS_KEYS],
        )
        count_matrix.to_csv(matrix_dir / f"rank_{int(rank):02d}_confusion_counts.csv")
        norm_matrix.to_csv(matrix_dir / f"rank_{int(rank):02d}_confusion_row_normalized.csv")


def copy_top_learning_curve_images(clean_runs: pd.DataFrame, top_configs: pd.DataFrame, out_dir: Path) -> None:
    target_dir = out_dir / "top_learning_curve_png"
    target_dir.mkdir(parents=True, exist_ok=True)
    for _, config in top_configs.iterrows():
        matches = clean_runs[(clean_runs["config_key"] == config["config_key"]) & (clean_runs["is_ok"])]
        for _, run in matches.iterrows():
            path = Path(str(run.get("learning_curve_png_resolved", "")))
            if path.exists():
                target = target_dir / f"rank_{int(config['rank']):02d}_{path.name}"
                if not target.exists():
                    shutil.copy2(path, target)


def write_overfitting_extra_outputs(summary: pd.DataFrame, leaderboard: pd.DataFrame, out_dir: Path) -> None:
    if not has_overfitting_data(summary):
        return
    complete = summary[summary["is_complete"]].copy() if "is_complete" in summary else summary.copy()
    if complete.empty:
        return

    top_plus_reference = complete.head(0).copy()
    if not leaderboard.empty:
        top_plus_reference = pd.concat([top_plus_reference, leaderboard.head(20)], ignore_index=True)
    reference_mask = (
        complete.get("overfit_stage", pd.Series("", index=complete.index)).astype(str).str.lower().eq("reference")
        | complete.get("overfit_config_id", pd.Series("", index=complete.index)).astype(str).str.startswith("ref_")
    )
    references = complete[reference_mask].copy()
    if not references.empty:
        top_plus_reference = pd.concat([top_plus_reference, references], ignore_index=True)
    if not top_plus_reference.empty:
        top_plus_reference = top_plus_reference.drop_duplicates(subset=["config_key"])
        top_plus_reference = sort_configs(top_plus_reference)
        top_plus_reference.to_csv(out_dir / "overfitting_top_configs_with_reference.csv", index=False)

    for name, group_cols in [
        ("overfitting_stage_summary.csv", ["overfit_stage"]),
        ("overfitting_epoch_summary.csv", ["cnn_epochs"]),
        ("overfitting_factor_summary.csv", ["stage1_regularization_factor"]),
        ("overfitting_factor_value_summary.csv", ["stage1_regularization_factor", "stage1_regularization_value"]),
        ("overfitting_factor_epoch_summary.csv", ["stage1_regularization_factor", "cnn_epochs"]),
        ("overfitting_regularization_grid_summary.csv", ["cnn_weight_decay", "cnn_dropout", "cnn_label_smoothing", "max_windows_fraction", "cnn_epochs"]),
    ]:
        table = summarize_overfitting_groups(complete, group_cols)
        if not table.empty:
            table.to_csv(out_dir / name, index=False)


def summarize_overfitting_groups(summary: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    available_cols = [col for col in group_cols if col in summary.columns]
    if not available_cols:
        return pd.DataFrame()
    data = summary.copy()
    for col in available_cols:
        data = data[data[col].astype(str).str.strip().ne("")]
        data = data[data[col].astype(str).str.lower().ne("nan")]
    if data.empty:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    grouped = data.groupby(available_cols, dropna=False, sort=False)
    for values, group in grouped:
        if not isinstance(values, tuple):
            values = (values,)
        ranked = sort_configs(group.copy())
        best = ranked.iloc[0]
        row: Dict[str, object] = {col: value for col, value in zip(available_cols, values)}
        row["n_configs"] = int(len(group))
        row["best_overfit_config_id"] = best.get("overfit_config_id", "")
        row["best_overfit_config_name"] = best.get("overfit_config_name", "")
        row["best_subject_balanced_accuracy_mean"] = best.get("subject_balanced_accuracy_mean", np.nan)
        row["best_subject_macro_f1_mean"] = best.get("subject_macro_f1_mean", np.nan)
        row["best_subject_balanced_accuracy_ci95_low"] = best.get("subject_balanced_accuracy_ci95_low", np.nan)
        row["best_worst_class_f1_mean"] = best.get("worst_class_f1_mean", np.nan)
        for metric in [
            "subject_balanced_accuracy_mean",
            "subject_macro_f1_mean",
            "worst_class_f1_mean",
            "subject_balanced_accuracy_std",
            "train_val_balanced_accuracy_gap_mean",
            "val_train_loss_gap_mean",
        ]:
            if metric in group.columns:
                values_numeric = pd.to_numeric(group[metric], errors="coerce")
                row[f"{metric}_mean_across_configs"] = safe_float(values_numeric.mean())
                row[f"{metric}_max_across_configs"] = safe_float(values_numeric.max())
                row[f"{metric}_min_across_configs"] = safe_float(values_numeric.min())
        rows.append(row)
    out = pd.DataFrame(rows)
    if "best_subject_balanced_accuracy_mean" in out.columns:
        out = out.sort_values(
            by=["best_subject_balanced_accuracy_mean", "best_subject_balanced_accuracy_ci95_low", "best_worst_class_f1_mean"],
            ascending=[False, False, False],
            na_position="last",
            kind="mergesort",
        ).reset_index(drop=True)
    return out


def try_import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def save_plots(clean_runs: pd.DataFrame, summary: pd.DataFrame, leaderboard: pd.DataFrame, confusion_long: pd.DataFrame, out_dir: Path) -> None:
    plt = try_import_matplotlib()
    if plt is None:
        return
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    ok_runs = clean_runs[clean_runs["is_ok"]].copy()
    complete_summary = summary[summary["is_complete"]].copy()

    plot_leaderboard(plt, leaderboard, fig_dir / "leaderboard_top_configs.png")
    plot_top10_worst_class_f1_stability(plt, leaderboard, fig_dir / "top10_worst_class_f1_stability.png")
    plot_top10_mean_std_stability(plt, leaderboard, fig_dir / "top10_subject_balanced_accuracy_mean_std.png")
    plot_boxplot(plt, ok_runs, "model", "subject_balanced_accuracy", fig_dir / "score_by_model_boxplot.png")
    plot_boxplot(plt, ok_runs, "extra_input", "subject_balanced_accuracy", fig_dir / "score_by_extra_input_boxplot.png")
    plot_boxplot(plt, ok_runs, "window_sec", "subject_balanced_accuracy", fig_dir / "score_by_window_sec_boxplot.png")
    plot_boxplot(plt, ok_runs, "overlap_pct", "subject_balanced_accuracy", fig_dir / "score_by_overlap_pct_boxplot.png")
    plot_window_overlap_heatmap(plt, complete_summary, fig_dir / "heatmap_window_overlap.png")
    plot_model_extra_heatmap(plt, complete_summary, fig_dir / "heatmap_model_extra_input.png")
    if has_overfitting_data(ok_runs):
        plot_boxplot(plt, ok_runs, "stage1_regularization_factor", "subject_balanced_accuracy", fig_dir / "overfitting_score_by_factor_boxplot.png")
        plot_boxplot(plt, ok_runs, "cnn_epochs", "subject_balanced_accuracy", fig_dir / "overfitting_score_by_epoch_boxplot.png")
        plot_boxplot(plt, ok_runs, "cnn_weight_decay", "subject_balanced_accuracy", fig_dir / "overfitting_score_by_weight_decay_boxplot.png")
        plot_boxplot(plt, ok_runs, "cnn_dropout", "subject_balanced_accuracy", fig_dir / "overfitting_score_by_dropout_boxplot.png")
        plot_boxplot(
            plt,
            ok_runs,
            "cnn_label_smoothing",
            "subject_balanced_accuracy",
            fig_dir / "overfitting_score_by_label_smoothing_boxplot.png",
        )
        plot_boxplot(
            plt,
            ok_runs,
            "max_windows_fraction",
            "subject_balanced_accuracy",
            fig_dir / "overfitting_score_by_max_windows_fraction_boxplot.png",
        )
        plot_overfitting_factor_epoch_heatmap(plt, complete_summary, fig_dir / "overfitting_heatmap_factor_epoch.png")
    plot_confusion_matrices(plt, confusion_long, fig_dir / "top_config_confusion_matrices.png", max_ranks=6)
    plot_top_learning_curves(plt, clean_runs, leaderboard, fig_dir / "top_learning_curves_accuracy.png")
    plt.close("all")


def plot_leaderboard(plt, leaderboard: pd.DataFrame, path: Path) -> None:
    if leaderboard.empty:
        return
    data = leaderboard.head(20).copy()
    labels = [top_config_label(row) for _, row in data.iterrows()]
    values = data["subject_balanced_accuracy_mean"].astype(float).to_numpy()
    fig, ax = plt.subplots(figsize=(11, max(4, 0.45 * len(data))))
    y = np.arange(len(data))
    ax.barh(y, values, color="#4477AA")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Mean subject balanced accuracy")
    ax.set_title("Top Configs by Subject Balanced Accuracy")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def top_config_label(row: pd.Series) -> str:
    overfit_id = str(row.get("overfit_config_id", "")).strip()
    if overfit_id:
        factor = str(row.get("stage1_regularization_factor", "")).strip()
        value = str(row.get("stage1_regularization_value", "")).strip()
        factor_text = f" {factor}={value}" if factor and value else ""
        return f"#{int(row['rank'])} {overfit_id}{factor_text} ep{int(row['cnn_epochs'])}"
    return (
        f"#{int(row['rank'])} {row['model']} {row['extra_input']} "
        f"{float(row['window_sec']):g}s ov{float(row['overlap_pct']):g}% p{int(row['cnn_patience'])}"
    )


def plot_top10_worst_class_f1_stability(plt, leaderboard: pd.DataFrame, path: Path) -> None:
    needed = {"worst_class_f1_mean", "subject_balanced_accuracy_std"}
    if leaderboard.empty or not needed.issubset(leaderboard.columns):
        return
    data = leaderboard.head(10).copy()
    data = data.sort_values(
        by=["worst_class_f1_mean", "subject_balanced_accuracy_std", "subject_balanced_accuracy_mean"],
        ascending=[False, True, False],
        na_position="last",
    ).reset_index(drop=True)
    data.insert(0, "stability_rank", np.arange(1, len(data) + 1))
    labels = [f"S{int(row['stability_rank'])} {top_config_label(row)}" for _, row in data.iterrows()]

    y = np.arange(len(data))
    worst_f1 = pd.to_numeric(data["worst_class_f1_mean"], errors="coerce").to_numpy(dtype=float)
    std_values = pd.to_numeric(data["subject_balanced_accuracy_std"], errors="coerce").to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(12, max(4.5, 0.5 * len(data))))
    ax.barh(y, worst_f1, color="#66AA77", alpha=0.9, label="worst class F1 mean")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Worst class F1 mean")
    ax.set_title("Top 10 Reordered by Worst-Class F1 and Repeat Stability")
    ax.grid(axis="x", alpha=0.25)

    ax2 = ax.twiny()
    ax2.scatter(std_values, y, color="#AA3377", marker="D", s=34, label="subject BA std")
    ax2.set_xlabel("Subject balanced accuracy std")
    lines, line_labels = ax.get_legend_handles_labels()
    lines2, line_labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, line_labels + line_labels2, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_top10_mean_std_stability(plt, leaderboard: pd.DataFrame, path: Path) -> None:
    needed = {"subject_balanced_accuracy_mean", "subject_balanced_accuracy_std"}
    if leaderboard.empty or not needed.issubset(leaderboard.columns):
        return
    data = leaderboard.head(10).copy()
    data = data.sort_values(
        by=["subject_balanced_accuracy_std", "subject_balanced_accuracy_mean"],
        ascending=[True, False],
        na_position="last",
    ).reset_index(drop=True)
    labels = [top_config_label(row) for _, row in data.iterrows()]
    values = pd.to_numeric(data["subject_balanced_accuracy_mean"], errors="coerce").to_numpy(dtype=float)
    errors = pd.to_numeric(data["subject_balanced_accuracy_std"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    y = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(12, max(4.5, 0.5 * len(data))))
    ax.barh(y, values, xerr=errors, color="#4477AA", ecolor="#333333", capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Subject balanced accuracy mean +/- std")
    ax.set_title("Top 10 Reordered by Lowest Repeat Std")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_boxplot(plt, data: pd.DataFrame, group_col: str, metric_col: str, path: Path) -> None:
    if data.empty or group_col not in data or metric_col not in data:
        return
    groups = []
    labels = []
    for label, sub in data.groupby(group_col, dropna=False):
        values = pd.to_numeric(sub[metric_col], errors="coerce").dropna().to_numpy()
        if len(values):
            groups.append(values)
            labels.append(str(label))
    if not groups:
        return
    fig, ax = plt.subplots(figsize=(8, 4.8))
    try:
        ax.boxplot(groups, tick_labels=labels, showmeans=True)
    except TypeError:
        ax.boxplot(groups, labels=labels, showmeans=True)
    ax.set_ylabel(metric_col)
    ax.set_title(f"{metric_col} by {group_col}")
    ax.grid(axis="y", alpha=0.25)
    fig.autofmt_xdate(rotation=25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_window_overlap_heatmap(plt, summary: pd.DataFrame, path: Path) -> None:
    if summary.empty:
        return
    pivot = summary.pivot_table(
        index="window_sec",
        columns="overlap_pct",
        values="subject_balanced_accuracy_mean",
        aggfunc="mean",
    ).sort_index(ascending=False)
    plot_heatmap(plt, pivot, "Mean Subject Balanced Accuracy by Window/Overlap", path)


def plot_model_extra_heatmap(plt, summary: pd.DataFrame, path: Path) -> None:
    if summary.empty:
        return
    pivot = summary.pivot_table(
        index="model",
        columns="extra_input",
        values="subject_balanced_accuracy_mean",
        aggfunc="mean",
    )
    plot_heatmap(plt, pivot, "Mean Subject Balanced Accuracy by Model/Extra Input", path)


def plot_overfitting_factor_epoch_heatmap(plt, summary: pd.DataFrame, path: Path) -> None:
    needed = {"stage1_regularization_factor", "cnn_epochs", "subject_balanced_accuracy_mean"}
    if summary.empty or not needed.issubset(summary.columns):
        return
    data = summary.copy()
    data = data[data["stage1_regularization_factor"].astype(str).str.strip().ne("")]
    data = data[data["stage1_regularization_factor"].astype(str).str.lower().ne("nan")]
    if data.empty:
        return
    pivot = data.pivot_table(
        index="stage1_regularization_factor",
        columns="cnn_epochs",
        values="subject_balanced_accuracy_mean",
        aggfunc="mean",
    )
    plot_heatmap(plt, pivot, "Mean Subject Balanced Accuracy by Overfitting Factor/Epoch", path)


def plot_heatmap(plt, pivot: pd.DataFrame, title: str, path: Path) -> None:
    if pivot.empty:
        return
    matrix = pivot.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(col) for col in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(idx) for idx in pivot.index])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if math.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", color="white", fontsize=8)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_confusion_matrices(plt, confusion_long: pd.DataFrame, path: Path, max_ranks: int = 6) -> None:
    if confusion_long.empty:
        return
    ranks = sorted(confusion_long["rank"].dropna().unique())[:max_ranks]
    if not ranks:
        return
    ncols = min(3, len(ranks))
    nrows = math.ceil(len(ranks) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows), squeeze=False)
    class_names = [CLASS_NAMES[key] for key in CLASS_KEYS]
    for ax in axes.ravel():
        ax.axis("off")
    for ax, rank in zip(axes.ravel(), ranks):
        sub = confusion_long[confusion_long["rank"] == rank]
        matrix = sub.pivot(index="true_class", columns="pred_class", values="row_normalized").reindex(
            index=class_names,
            columns=class_names,
        )
        values = matrix.to_numpy(dtype=float)
        ax.axis("on")
        image = ax.imshow(values, vmin=0.0, vmax=1.0, cmap="Blues")
        ax.set_title(f"Rank {int(rank)}")
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_yticklabels(class_names, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_top_learning_curves(plt, clean_runs: pd.DataFrame, leaderboard: pd.DataFrame, path: Path, max_configs: int = 6) -> None:
    if leaderboard.empty:
        return
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    plotted = 0
    for _, config in leaderboard.head(max_configs).iterrows():
        matches = clean_runs[(clean_runs["config_key"] == config["config_key"]) & (clean_runs["is_ok"])]
        frames = []
        for _, run in matches.iterrows():
            csv_path = Path(str(run.get("learning_curve_csv_resolved", "")))
            if not csv_path.exists():
                continue
            try:
                curve = pd.read_csv(csv_path)
            except Exception:
                continue
            if {"epoch", "val_balanced_accuracy"}.issubset(curve.columns):
                frames.append(curve[["epoch", "val_balanced_accuracy"]].copy())
        if not frames:
            continue
        combined = pd.concat(frames, ignore_index=True)
        mean_curve = combined.groupby("epoch", as_index=False)["val_balanced_accuracy"].mean()
        label = top_config_label(config)
        ax.plot(mean_curve["epoch"], mean_curve["val_balanced_accuracy"], linewidth=1.8, label=label)
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        return
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean validation balanced accuracy")
    ax.set_title("Top Config Learning Curves")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_markdown_report(
    out_dir: Path,
    sweep_dirs: Sequence[Path],
    models: Sequence[str],
    expected_repeats_label: str,
    clean_runs: pd.DataFrame,
    summary: pd.DataFrame,
    leaderboard: pd.DataFrame,
) -> None:
    report_path = out_dir / "analysis_report.md"
    complete_count = int(summary["is_complete"].sum()) if "is_complete" in summary else 0
    matched_models = ", ".join(sorted(clean_runs["model"].astype(str).unique())) if "model" in clean_runs else ""
    if "resolved_model" in clean_runs:
        matched_resolved_models = ", ".join(sorted(clean_runs["resolved_model"].astype(str).unique()))
    else:
        matched_resolved_models = ""
    lines = [
        "# Sweep Analysis Report",
        "",
        "- Input sweep directories:",
        *[f"  - `{sweep_dir}`" for sweep_dir in sweep_dirs],
        f"- Requested model filters: `{', '.join(models)}`",
        f"- Matched `model` values: `{matched_models}`",
        f"- Matched `resolved_model` values: `{matched_resolved_models}`",
        f"- Expected repeats per config: `{expected_repeats_label}`",
        f"- Model-filtered runs: `{len(clean_runs)}`",
        f"- Configs: `{len(summary)}`",
        f"- Complete configs: `{complete_count}`",
        "",
        "Ranking uses config-level aggregation, not single-run ranking. Runtime is retained as a reference field only.",
        "",
        "## Ranking Order",
        "",
        "1. `subject_balanced_accuracy_mean` descending",
        "2. `subject_macro_f1_mean` descending",
        "3. `subject_balanced_accuracy_ci95_low` descending",
        "4. `subject_macro_f1_ci95_low` descending",
        "5. `worst_class_recall_mean` descending",
        "6. `worst_class_f1_mean` descending",
        "7. `subject_balanced_accuracy_std` ascending",
        "",
        "## Top Configs",
        "",
    ]
    if leaderboard.empty:
        lines.append("No complete configs were available for the leaderboard.")
    else:
        display_cols = [
            "rank",
            "model",
            "resolved_model",
            "source_sweep_names",
            "source_sweep_kinds",
            "extra_input",
            "overfit_config_id",
            "overfit_config_name",
            "stage1_regularization_factor",
            "stage1_regularization_value",
            "cnn_epochs",
            "cnn_patience",
            "cnn_lr",
            "cnn_weight_decay",
            "cnn_dropout",
            "cnn_label_smoothing",
            "window_sec",
            "overlap_pct",
            "max_windows_fraction",
            "subject_balanced_accuracy_mean",
            "subject_macro_f1_mean",
            "subject_balanced_accuracy_ci95_low",
            "worst_class_recall_mean",
            "worst_class_f1_mean",
            "train_val_balanced_accuracy_gap_mean",
            "val_train_loss_gap_mean",
            "n_repeats_done",
        ]
        table = leaderboard[[col for col in display_cols if col in leaderboard.columns]].head(20)
        lines.append(dataframe_to_markdown(table))
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `clean_runs.csv`: filtered run-level table with resolved artifact paths and class-level metrics.",
            "- `config_summary.csv`: repeat-aggregated config-level summary.",
            "- `leaderboard_top_configs.csv`: top complete configs after config-level ranking.",
            "- `leaderboard_top10_worst_class_f1_stability.csv`: top 10 resorted by worst-class F1 and repeat std.",
            "- `incomplete_configs.csv`: configs that did not complete all expected repeats.",
            "- `class_level_summary.csv`: class-level precision/recall/F1/support by config.",
            "- `top_config_confusion_matrices_long.csv`: aggregated confusion matrices for top configs.",
            "- `figures/`: leaderboard, boxplots, heatmaps, learning curves, and confusion matrix figures.",
        ]
    )
    if has_overfitting_data(summary):
        lines.extend(
            [
                "",
                "## Overfitting Sweep Extras",
                "",
                "- `overfitting_top_configs_with_reference.csv`: top ranked configs plus reference configs.",
                "- `overfitting_epoch_summary.csv`: config-level summary grouped by fixed epoch.",
                "- `overfitting_factor_summary.csv`: config-level summary grouped by regularization factor.",
                "- `overfitting_factor_value_summary.csv`: config-level summary grouped by regularization factor and value.",
                "- `overfitting_factor_epoch_summary.csv`: config-level summary grouped by factor and epoch.",
                "- `overfitting_regularization_grid_summary.csv`: config-level summary grouped by weight decay, dropout, label smoothing, max windows fraction, and epoch.",
            ]
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    columns = list(frame.columns)
    rows = []
    rows.append("| " + " | ".join(columns) + " |")
    rows.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for _, row in frame.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                values.append("" if pd.isna(value) else f"{float(value):.4f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def save_outputs(
    clean_runs: pd.DataFrame,
    summary: pd.DataFrame,
    leaderboard: pd.DataFrame,
    class_summary: pd.DataFrame,
    confusion_long: pd.DataFrame,
    out_dir: Path,
) -> None:
    clean_runs.to_csv(out_dir / "clean_runs.csv", index=False)
    summary.to_csv(out_dir / "config_summary.csv", index=False)
    leaderboard.to_csv(out_dir / "leaderboard_top_configs.csv", index=False)
    stability_cols = [
        "worst_class_f1_mean",
        "subject_balanced_accuracy_std",
        "subject_balanced_accuracy_mean",
    ]
    if not leaderboard.empty and set(stability_cols).issubset(leaderboard.columns):
        stability = leaderboard.head(10).sort_values(
            by=stability_cols,
            ascending=[False, True, False],
            na_position="last",
        )
        stability.to_csv(out_dir / "leaderboard_top10_worst_class_f1_stability.csv", index=False)
    summary[~summary["is_complete"]].to_csv(out_dir / "incomplete_configs.csv", index=False)
    class_summary.to_csv(out_dir / "class_level_summary.csv", index=False)
    confusion_long.to_csv(out_dir / "top_config_confusion_matrices_long.csv", index=False)
    write_top_confusion_csvs(confusion_long, out_dir)
    copy_top_learning_curve_images(clean_runs, leaderboard, out_dir)
    write_overfitting_extra_outputs(summary, leaderboard, out_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze frailty sweep outputs at the config level. "
            "Model filters match both model and resolved_model, so --models shapeformer also includes "
            "shapeformer_pisd runs whose resolved_model is shapeformer."
        )
    )
    parser.add_argument(
        "--sweep-dir",
        "--sweep-dirs",
        dest="sweep_dirs",
        action="append",
        default=None,
        help=(
            "Directory containing sweep_runs.csv or overfitting_runs.csv plus reports/ and learning_curves/. "
            "Can be repeated or comma-separated for combined analysis."
        ),
    )
    parser.add_argument(
        "--output-root",
        default="results_frailty3/_sweep_analyse",
        help="Root directory where a timestamped analysis folder will be created.",
    )
    parser.add_argument(
        "--models",
        default="cnn,inceptiontime",
        help=(
            "Comma-separated model names to analyze. Default keeps ShapeFormer out. "
            "Use --models shapeformer for both shapeformer and shapeformer_pisd resolved as ShapeFormer; "
            "use --models shapeformer_pisd to isolate the PISD branch."
        ),
    )
    parser.add_argument("--top-k", type=int, default=20, help="Number of top configs to write to the leaderboard.")
    parser.add_argument(
        "--expected-repeats",
        type=int,
        default=0,
        help="Override expected repeats per config. If omitted, infer from sweep_manifest.json or max repeat.",
    )
    parser.add_argument(
        "--include-incomplete-leaderboard",
        action="store_true",
        help="Allow incomplete configs in leaderboard. By default, leaderboard only ranks complete configs.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip matplotlib figure generation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sweep_dir_values = parse_repeated_csv_list(args.sweep_dirs) or DEFAULT_SWEEP_DIRS
    sweep_dirs = []
    seen_dirs = set()
    for value in sweep_dir_values:
        path = normalize_user_path(value)
        key = str(path)
        if key not in seen_dirs:
            sweep_dirs.append(path)
            seen_dirs.add(key)
    output_root = normalize_user_path(args.output_root)
    models = parse_csv_list(args.models) or DEFAULT_MODELS
    if len(sweep_dirs) > 1:
        out_prefix = "combined"
    elif sweep_dirs and (sweep_dirs[0] / "overfitting_runs.csv").exists():
        out_prefix = "overfitting"
    else:
        out_prefix = ""
    out_dir = unique_output_dir(output_root, models, prefix=out_prefix)

    run_frames: List[pd.DataFrame] = []
    inferred_repeats_by_dir: List[int] = []
    for sweep_dir in sweep_dirs:
        runs, _, inferred_repeats = prepare_runs(sweep_dir, models)
        if int(args.expected_repeats) > 0:
            runs["expected_repeats"] = int(args.expected_repeats)
            runs["source_expected_repeats"] = int(args.expected_repeats)
        run_frames.append(runs)
        inferred_repeats_by_dir.append(int(args.expected_repeats) if int(args.expected_repeats) > 0 else inferred_repeats)

    clean_runs = pd.concat(run_frames, ignore_index=True) if run_frames else pd.DataFrame()
    if clean_runs.empty:
        wanted = ", ".join(models)
        sources = ", ".join(str(path) for path in sweep_dirs)
        raise RuntimeError(f"No runs matched models [{wanted}] in sweep directories [{sources}]")

    if int(args.expected_repeats) > 0:
        default_expected_repeats = int(args.expected_repeats)
        expected_repeats_label = str(default_expected_repeats)
    else:
        default_expected_repeats = max(inferred_repeats_by_dir) if inferred_repeats_by_dir else 1
        expected_repeats_label = "per-source inferred (" + ", ".join(
            f"{path.name}:{repeats}" for path, repeats in zip(sweep_dirs, inferred_repeats_by_dir)
        ) + ")"

    summary = aggregate_config_summary(clean_runs, default_expected_repeats)
    leaderboard = build_leaderboard(summary, args.top_k, args.include_incomplete_leaderboard)
    class_summary = build_class_level_summary(summary)
    confusion_long = aggregate_confusion_matrices(clean_runs, leaderboard)

    save_outputs(clean_runs, summary, leaderboard, class_summary, confusion_long, out_dir)
    if not args.no_plots:
        save_plots(clean_runs, summary, leaderboard, confusion_long, out_dir)
    write_markdown_report(out_dir, sweep_dirs, models, expected_repeats_label, clean_runs, summary, leaderboard)

    print(f"Analysis output directory: {out_dir}")
    print("Input sweep directories:")
    for sweep_dir in sweep_dirs:
        print(f"  - {sweep_dir}")
    print(f"Model-filtered runs: {len(clean_runs)}")
    print(f"Config groups: {len(summary)}")
    print(f"Complete config groups: {int(summary['is_complete'].sum())}")
    print(f"Leaderboard rows: {len(leaderboard)}")
    if not leaderboard.empty:
        best = leaderboard.iloc[0]
        overfit_id = str(best.get("overfit_config_id", "")).strip()
        overfit_text = f", overfit_config={overfit_id}" if overfit_id else ""
        print(
            "Best config: "
            f"model={best['model']}{overfit_text}, extra={best['extra_input']}, "
            f"window={best['window_sec']}s, overlap={best['overlap_pct']}%, "
            f"subject_balanced_accuracy_mean={best['subject_balanced_accuracy_mean']:.3f}"
        )


if __name__ == "__main__":
    main()
