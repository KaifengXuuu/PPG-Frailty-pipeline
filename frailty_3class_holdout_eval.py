from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from frailty_3class_classifier import (
    CLASS_NAMES,
    RunConfig,
    aggregate_by_key,
    build_cnn_window_table,
    build_feature_table,
    cnn_device,
    cnn_predict_proba,
    finite_float,
    normalize_extra_input,
    save_learning_curve_artifacts,
    scaled_file_features_for_fold,
    select_extra_feature_columns,
    train_cnn_model,
)


CONFIG_COLUMNS = [
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
CLASS_KEYS = list(CLASS_NAMES)
DEFAULT_RANKS = [1, 2, 7]
DEFAULT_SEEDS = [42, 10042, 20042, 30042, 40042]


def normalize_user_path(value: str) -> Path:
    text = str(value).strip().replace("\\", "/")
    if text.startswith("home/"):
        text = "/" + text
    return Path(text).expanduser()


def parse_int_list(value: str) -> List[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def latest_analysis_dir(root: Path) -> Path:
    candidates = [path for path in root.iterdir() if path.is_dir() and (path / "leaderboard_top_configs.csv").exists()]
    if not candidates:
        raise FileNotFoundError(f"No analysis directory with leaderboard_top_configs.csv under {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def unique_output_dir(root: Path, ranks: Sequence[int]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    rank_tag = "rank" + "-".join(str(int(rank)) for rank in ranks)
    base = root / f"{datetime.now().strftime('%Y%m%d_%H%M')}_{rank_tag}_holdout"
    if not base.exists():
        base.mkdir(parents=True)
        return base
    for idx in range(2, 1000):
        candidate = root / f"{base.name}_{idx:02d}"
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate
    raise RuntimeError(f"Could not create a unique output directory under {root}")


def load_rank_configs(analysis_dir: Path, ranks: Sequence[int]) -> pd.DataFrame:
    path = analysis_dir / "leaderboard_top_configs.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing leaderboard_top_configs.csv: {path}")
    leaderboard = pd.read_csv(path)
    leaderboard["rank"] = pd.to_numeric(leaderboard["rank"], errors="coerce").astype("Int64")
    wanted = set(int(rank) for rank in ranks)
    selected = leaderboard[leaderboard["rank"].isin(wanted)].copy()
    missing = sorted(wanted - set(int(rank) for rank in selected["rank"].dropna().tolist()))
    if missing:
        raise ValueError(f"Ranks not found in leaderboard: {missing}")
    selected = selected.sort_values("rank").reset_index(drop=True)
    for col in CONFIG_COLUMNS:
        if col not in selected.columns:
            raise KeyError(f"Missing config column in leaderboard: {col}")
    return selected


def resolve_model_name(row: pd.Series) -> str:
    resolved = str(row.get("resolved_model", "")).strip()
    if resolved in {"cnn1d", "inception_time"}:
        return resolved
    model = str(row.get("model", "")).strip().lower()
    if model == "cnn":
        return "cnn1d"
    if model in {"inceptiontime", "inception_time"}:
        return "inception_time"
    raise ValueError(f"Holdout script currently supports cnn/inceptiontime only, got: {model}")


def config_from_rank_row(args: argparse.Namespace, row: pd.Series) -> RunConfig:
    config = RunConfig(
        data_root=str(normalize_user_path(args.data_root)),
        folds=1,
        seed=int(args.base_seed),
        cnn_target_fs=float(args.cnn_target_fs),
        cnn_batch_size=int(args.cnn_batch_size),
        cnn_lr=float(args.cnn_lr),
        cnn_num_workers=int(args.cnn_num_workers),
        cnn_weight_decay=float(args.cnn_weight_decay),
        cnn_dropout=float(args.cnn_dropout),
        cnn_label_smoothing=float(args.cnn_label_smoothing),
    )
    config.cnn_seq_sec = float(row["window_sec"])
    config.cnn_hop_sec = float(row["hop_sec"])
    config.cnn_max_windows_fraction = float(row["max_windows_fraction"])
    config.cnn_epochs = int(args.epochs_override) if int(args.epochs_override) > 0 else int(row["cnn_epochs"])
    config.cnn_patience = int(args.patience_override) if int(args.patience_override) > 0 else int(row["cnn_patience"])
    config.extra_input = normalize_extra_input(str(row["extra_input"]))
    return config


def subject_split_class_counts(subject_table: pd.DataFrame, subjects: np.ndarray) -> Dict[str, int]:
    mask = subject_table["subject"].astype(str).isin(subjects.astype(str))
    return subject_table.loc[mask, "class_name"].value_counts().reindex(CLASS_NAMES, fill_value=0).astype(int).to_dict()


def subject_train_val_test_split(
    features: pd.DataFrame,
    seed: int,
    val_size: float,
    test_size: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    subject_table = (
        features.groupby("subject", as_index=False)
        .agg(label=("label", "first"), class_name=("class_name", "first"))
        .sort_values("subject")
        .reset_index(drop=True)
    )
    if not (0.0 < float(test_size) < 1.0):
        raise ValueError("--test-size must be between 0 and 1.")
    if not (0.0 < float(val_size) < 1.0):
        raise ValueError("--val-size must be between 0 and 1.")
    if float(test_size) + float(val_size) >= 1.0:
        raise ValueError("--test-size + --val-size must be less than 1.")

    all_subjects = subject_table["subject"].astype(str).to_numpy()
    all_labels = subject_table["label"].to_numpy(dtype=int)
    train_val_subjects, test_subjects = train_test_split(
        all_subjects,
        test_size=float(test_size),
        random_state=int(seed),
        stratify=all_labels,
    )
    train_val_subjects = np.asarray(train_val_subjects, dtype=str)
    test_subjects = np.asarray(test_subjects, dtype=str)

    train_val_table = subject_table[subject_table["subject"].astype(str).isin(train_val_subjects)].copy()
    relative_val_size = float(val_size) / (1.0 - float(test_size))
    train_subjects, val_subjects = train_test_split(
        train_val_table["subject"].astype(str).to_numpy(),
        test_size=relative_val_size,
        random_state=int(seed) + 7919,
        stratify=train_val_table["label"].to_numpy(dtype=int),
    )
    train_subjects = np.asarray(train_subjects, dtype=str)
    val_subjects = np.asarray(val_subjects, dtype=str)
    test_subjects = np.asarray(test_subjects, dtype=str)
    train_idx = features.index[features["subject"].astype(str).isin(train_subjects)].to_numpy(dtype=np.int64)
    val_idx = features.index[features["subject"].astype(str).isin(val_subjects)].to_numpy(dtype=np.int64)
    test_idx = features.index[features["subject"].astype(str).isin(test_subjects)].to_numpy(dtype=np.int64)
    meta = {
        "split_design": "subject_stratified_train64_val16_test20",
        "train_size": finite_float(1.0 - float(val_size) - float(test_size)),
        "val_size": float(val_size),
        "test_size": float(test_size),
        "relative_val_size_within_train_val": finite_float(relative_val_size),
        "split_seed": int(seed),
        "n_train_subjects": int(len(train_subjects)),
        "n_val_subjects": int(len(val_subjects)),
        "n_test_subjects": int(len(test_subjects)),
        "train_subjects": sorted(train_subjects.astype(str).tolist()),
        "val_subjects": sorted(val_subjects.astype(str).tolist()),
        "test_subjects": sorted(test_subjects.astype(str).tolist()),
        "subject_class_counts": subject_table["class_name"].value_counts().to_dict(),
        "train_class_counts": subject_split_class_counts(subject_table, train_subjects),
        "val_class_counts": subject_split_class_counts(subject_table, val_subjects),
        "test_class_counts": subject_split_class_counts(subject_table, test_subjects),
    }
    return train_idx, val_idx, test_idx, meta


def report_class_metrics(report: Dict[str, object]) -> Dict[str, float]:
    class_report = report.get("subject_classification_report") or {}
    out: Dict[str, float] = {}
    recalls: List[float] = []
    f1s: List[float] = []
    for class_key in CLASS_KEYS:
        item = class_report.get(class_key) or {}
        recall = safe_float(item.get("recall"))
        f1 = safe_float(item.get("f1-score"))
        out[f"{class_key}_precision"] = safe_float(item.get("precision"))
        out[f"{class_key}_recall"] = recall
        out[f"{class_key}_f1"] = f1
        out[f"{class_key}_support"] = safe_float(item.get("support"))
        if math.isfinite(recall):
            recalls.append(recall)
        if math.isfinite(f1):
            f1s.append(f1)
    out["worst_class_recall"] = min(recalls) if recalls else np.nan
    out["worst_class_f1"] = min(f1s) if f1s else np.nan
    return out


def safe_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return np.nan
    return out if math.isfinite(out) else np.nan


def evaluate_window_subset(
    *,
    model,
    x_win: np.ndarray,
    y_win: np.ndarray,
    subject_win: np.ndarray,
    file_win: np.ndarray,
    mask: np.ndarray,
    batch_size: int,
    extra: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    labels = y_win[mask].astype(int)
    probs = cnn_predict_proba(
        model,
        x_win[mask],
        batch_size,
        cnn_device(),
        extra=extra[mask] if extra is not None else None,
    )
    preds = np.argmax(probs, axis=1).astype(int).tolist()
    window_true = labels.astype(int).tolist()
    file_true, file_pred = aggregate_by_key(probs, labels, file_win[mask].tolist())
    subject_true, subject_pred = aggregate_by_key(probs, labels, subject_win[mask].tolist())
    return {
        "window_balanced_accuracy": finite_float(balanced_accuracy_score(window_true, preds)),
        "window_macro_f1": finite_float(f1_score(window_true, preds, labels=[0, 1, 2], average="macro", zero_division=0)),
        "file_balanced_accuracy": finite_float(balanced_accuracy_score(file_true, file_pred)),
        "file_macro_f1": finite_float(f1_score(file_true, file_pred, labels=[0, 1, 2], average="macro", zero_division=0)),
        "subject_balanced_accuracy": finite_float(balanced_accuracy_score(subject_true, subject_pred)),
        "subject_macro_f1": finite_float(f1_score(subject_true, subject_pred, labels=[0, 1, 2], average="macro", zero_division=0)),
        "window_confusion_matrix": confusion_matrix(window_true, preds, labels=[0, 1, 2]).tolist(),
        "file_confusion_matrix": confusion_matrix(file_true, file_pred, labels=[0, 1, 2]).tolist(),
        "subject_confusion_matrix": confusion_matrix(subject_true, subject_pred, labels=[0, 1, 2]).tolist(),
        "file_classification_report": classification_report(
            file_true,
            file_pred,
            labels=[0, 1, 2],
            target_names=list(CLASS_NAMES),
            zero_division=0,
            output_dict=True,
        ),
        "subject_classification_report": classification_report(
            subject_true,
            subject_pred,
            labels=[0, 1, 2],
            target_names=list(CLASS_NAMES),
            zero_division=0,
            output_dict=True,
        ),
    }


def train_eval_holdout_once(
    *,
    features: pd.DataFrame,
    config: RunConfig,
    rank_row: pd.Series,
    rank: int,
    repeat: int,
    seed: int,
    val_size: float,
    test_size: float,
    refresh_cnn_windows: bool,
    report_dir: Path,
    learning_curve_dir: Path,
) -> Tuple[Dict[str, object], Path]:
    start_time = time.perf_counter()
    features = features.reset_index(drop=True)
    config.seed = int(seed)
    model_name = resolve_model_name(rank_row)
    train_idx, val_idx, test_idx, split_meta = subject_train_val_test_split(
        features,
        seed=seed,
        val_size=val_size,
        test_size=test_size,
    )

    x_win, y_win, subject_win, file_win, cnn_cache_path = build_cnn_window_table(
        features,
        config,
        refresh=refresh_cnn_windows,
    )
    train_mask = np.isin(file_win, train_idx)
    val_mask = np.isin(file_win, val_idx)
    test_mask = np.isin(file_win, test_idx)
    if not np.any(train_mask) or not np.any(val_mask) or not np.any(test_mask):
        raise RuntimeError("Empty train, validation, or test window set after holdout split.")

    extra_cols = select_extra_feature_columns(features, config.extra_input, config.manual_features)
    window_extra: Optional[np.ndarray] = None
    if extra_cols:
        scaled_file_features = scaled_file_features_for_fold(features, extra_cols, train_idx)
        window_extra = scaled_file_features[file_win]

    model, train_info = train_cnn_model(
        x_win[train_mask],
        y_win[train_mask],
        config,
        seed=seed,
        model_name=model_name,
        x_val=x_win[val_mask],
        y_val=y_win[val_mask],
        extra_train=window_extra[train_mask] if window_extra is not None else None,
        extra_val=window_extra[val_mask] if window_extra is not None else None,
    )

    validation_eval = evaluate_window_subset(
        model=model,
        x_win=x_win,
        y_win=y_win,
        subject_win=subject_win,
        file_win=file_win,
        mask=val_mask,
        batch_size=config.cnn_batch_size,
        extra=window_extra,
    )
    test_eval = evaluate_window_subset(
        model=model,
        x_win=x_win,
        y_win=y_win,
        subject_win=subject_win,
        file_win=file_win,
        mask=test_mask,
        batch_size=config.cnn_batch_size,
        extra=window_extra,
    )

    fold_summary = {
        "fold": int(repeat),
        "n_train_files": int(len(train_idx)),
        "n_val_files": int(len(val_idx)),
        "n_test_files": int(len(test_idx)),
        "n_train_windows": int(np.sum(train_mask)),
        "n_val_windows": int(np.sum(val_mask)),
        "n_test_windows": int(np.sum(test_mask)),
        "val_subjects": split_meta["val_subjects"],
        "test_subjects": split_meta["test_subjects"],
        "best_epoch": int(train_info["best_epoch"]),
        "best_window_balanced_accuracy": train_info["best_window_balanced_accuracy"],
        "history": train_info["history"],
        "validation_file_balanced_accuracy": validation_eval["file_balanced_accuracy"],
        "validation_subject_balanced_accuracy": validation_eval["subject_balanced_accuracy"],
        "test_file_balanced_accuracy": test_eval["file_balanced_accuracy"],
        "test_subject_balanced_accuracy": test_eval["subject_balanced_accuracy"],
    }
    report: Dict[str, object] = {
        "model": str(rank_row["model"]),
        "resolved_model": model_name,
        "rank": int(rank),
        "repeat": int(repeat),
        "seed": int(seed),
        "holdout_split": split_meta,
        "early_stopping_source": "inner_validation_set",
        "n_files": int(len(features)),
        "n_subjects": int(features["subject"].nunique()),
        "n_windows": int(len(y_win)),
        "n_train_windows": int(np.sum(train_mask)),
        "n_val_windows": int(np.sum(val_mask)),
        "n_test_windows": int(np.sum(test_mask)),
        "extra_input": normalize_extra_input(config.extra_input),
        "n_extra_features": int(len(extra_cols)),
        "cnn_cache": str(cnn_cache_path),
        "validation_window_balanced_accuracy": validation_eval["window_balanced_accuracy"],
        "validation_window_macro_f1": validation_eval["window_macro_f1"],
        "validation_file_balanced_accuracy": validation_eval["file_balanced_accuracy"],
        "validation_file_macro_f1": validation_eval["file_macro_f1"],
        "validation_subject_balanced_accuracy": validation_eval["subject_balanced_accuracy"],
        "validation_subject_macro_f1": validation_eval["subject_macro_f1"],
        "validation_window_confusion_matrix": validation_eval["window_confusion_matrix"],
        "validation_file_confusion_matrix": validation_eval["file_confusion_matrix"],
        "validation_subject_confusion_matrix": validation_eval["subject_confusion_matrix"],
        "validation_file_classification_report": validation_eval["file_classification_report"],
        "validation_subject_classification_report": validation_eval["subject_classification_report"],
        "window_balanced_accuracy": test_eval["window_balanced_accuracy"],
        "window_macro_f1": test_eval["window_macro_f1"],
        "file_balanced_accuracy": test_eval["file_balanced_accuracy"],
        "file_macro_f1": test_eval["file_macro_f1"],
        "subject_balanced_accuracy": test_eval["subject_balanced_accuracy"],
        "subject_macro_f1": test_eval["subject_macro_f1"],
        "window_confusion_matrix": test_eval["window_confusion_matrix"],
        "file_confusion_matrix": test_eval["file_confusion_matrix"],
        "subject_confusion_matrix": test_eval["subject_confusion_matrix"],
        "file_classification_report": test_eval["file_classification_report"],
        "subject_classification_report": test_eval["subject_classification_report"],
        "folds": [fold_summary],
        "feature_columns": extra_cols,
        "config": asdict(config),
    }
    tag = (
        f"rank{int(rank):02d}_r{int(repeat)}_{rank_row['model']}_extra-{config.extra_input}_"
        f"seq{config.cnn_seq_sec:g}_ov{float(rank_row['overlap_pct']):g}_seed{int(seed)}"
    ).replace(".", "p")
    report.update(save_learning_curve_artifacts(report, out_dir=learning_curve_dir, filename_prefix=tag))
    report["duration_sec"] = float(time.perf_counter() - start_time)
    report_path = report_dir / f"{tag}_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report, report_path


def single_run_row(report: Dict[str, object], rank_row: pd.Series, report_path: Path) -> Dict[str, object]:
    row: Dict[str, object] = {
        "rank": int(report["rank"]),
        "repeat": int(report["repeat"]),
        "seed": int(report["seed"]),
        "model": str(rank_row["model"]),
        "resolved_model": str(report["resolved_model"]),
        "extra_input": str(report["extra_input"]),
        "cnn_epochs": int(report["config"]["cnn_epochs"]),
        "cnn_patience": int(report["config"]["cnn_patience"]),
        "window_sec": float(report["config"]["cnn_seq_sec"]),
        "hop_sec": float(report["config"]["cnn_hop_sec"]),
        "overlap_pct": float(rank_row["overlap_pct"]),
        "max_windows_fraction": float(report["config"]["cnn_max_windows_fraction"]),
        "cnn_lr": float(report["config"].get("cnn_lr", np.nan)),
        "cnn_weight_decay": float(report["config"].get("cnn_weight_decay", np.nan)),
        "cnn_dropout": float(report["config"].get("cnn_dropout", np.nan)),
        "cnn_label_smoothing": float(report["config"].get("cnn_label_smoothing", np.nan)),
        "n_train_subjects": int(report["holdout_split"]["n_train_subjects"]),
        "n_val_subjects": int(report["holdout_split"]["n_val_subjects"]),
        "n_test_subjects": int(report["holdout_split"]["n_test_subjects"]),
        "n_train_windows": int(report["n_train_windows"]),
        "n_val_windows": int(report["n_val_windows"]),
        "n_test_windows": int(report["n_test_windows"]),
        "validation_window_balanced_accuracy": report["validation_window_balanced_accuracy"],
        "validation_window_macro_f1": report["validation_window_macro_f1"],
        "validation_file_balanced_accuracy": report["validation_file_balanced_accuracy"],
        "validation_file_macro_f1": report["validation_file_macro_f1"],
        "validation_subject_balanced_accuracy": report["validation_subject_balanced_accuracy"],
        "validation_subject_macro_f1": report["validation_subject_macro_f1"],
        "window_balanced_accuracy": report["window_balanced_accuracy"],
        "window_macro_f1": report["window_macro_f1"],
        "file_balanced_accuracy": report["file_balanced_accuracy"],
        "file_macro_f1": report["file_macro_f1"],
        "subject_balanced_accuracy": report["subject_balanced_accuracy"],
        "subject_macro_f1": report["subject_macro_f1"],
        "best_epoch": report["folds"][0]["best_epoch"],
        "best_window_balanced_accuracy": report["folds"][0]["best_window_balanced_accuracy"],
        "duration_sec": report["duration_sec"],
        "report_path": str(report_path),
        "learning_curve_csv": report.get("learning_curve_csv", ""),
        "learning_curve_png": report.get("learning_curve_png", ""),
    }
    row.update(report_class_metrics(report))
    return row


def t_critical_975(n: int) -> float:
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
    if n <= 1:
        return np.nan
    return table.get(n - 1, 1.960)


def metric_summary(values: pd.Series) -> Dict[str, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    n = len(numeric)
    if n == 0:
        return {"mean": np.nan, "std": np.nan, "ci95_low": np.nan, "ci95_high": np.nan, "min": np.nan, "max": np.nan}
    mean = float(numeric.mean())
    std = float(numeric.std(ddof=1)) if n > 1 else 0.0
    if n > 1:
        margin = t_critical_975(n) * std / math.sqrt(n)
        ci_low = mean - margin
        ci_high = mean + margin
    else:
        ci_low = np.nan
        ci_high = np.nan
    return {"mean": mean, "std": std, "ci95_low": ci_low, "ci95_high": ci_high, "min": float(numeric.min()), "max": float(numeric.max())}


def build_summary(run_rows: List[Dict[str, object]]) -> pd.DataFrame:
    runs = pd.DataFrame(run_rows)
    rows: List[Dict[str, object]] = []
    group_cols = ["rank"] + CONFIG_COLUMNS
    for values, sub in runs.groupby(group_cols, dropna=False, sort=True):
        row = {col: value for col, value in zip(group_cols, values)}
        row["n_repeats"] = int(sub["repeat"].nunique())
        for metric in [
            "validation_subject_balanced_accuracy",
            "validation_subject_macro_f1",
            "validation_file_balanced_accuracy",
            "validation_file_macro_f1",
            "validation_window_balanced_accuracy",
            "validation_window_macro_f1",
            "subject_balanced_accuracy",
            "subject_macro_f1",
            "file_balanced_accuracy",
            "file_macro_f1",
            "window_balanced_accuracy",
            "window_macro_f1",
            "duration_sec",
            "best_epoch",
            "worst_class_recall",
            "worst_class_f1",
        ]:
            stats = metric_summary(sub[metric])
            for stat, stat_value in stats.items():
                row[f"{metric}_{stat}"] = stat_value
        for class_key in CLASS_KEYS:
            for metric in ["precision", "recall", "f1", "support"]:
                col = f"{class_key}_{metric}"
                stats = metric_summary(sub[col])
                row[f"{col}_mean"] = stats["mean"]
                row[f"{col}_std"] = stats["std"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("rank").reset_index(drop=True)


def write_confusion_aggregates(reports: Sequence[Dict[str, object]], out_dir: Path) -> None:
    matrix_dir = out_dir / "confusion_matrices"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    for rank in sorted({int(report["rank"]) for report in reports}):
        total = np.zeros((len(CLASS_KEYS), len(CLASS_KEYS)), dtype=float)
        for report in reports:
            if int(report["rank"]) == rank:
                total += np.asarray(report["subject_confusion_matrix"], dtype=float)
        row_sums = total.sum(axis=1, keepdims=True)
        normalized = np.divide(total, row_sums, out=np.zeros_like(total), where=row_sums > 0)
        index = list(CLASS_NAMES)
        columns = list(CLASS_NAMES)
        pd.DataFrame(total.astype(int), index=index, columns=columns).to_csv(
            matrix_dir / f"rank_{rank:02d}_subject_confusion_counts.csv"
        )
        pd.DataFrame(normalized, index=index, columns=columns).to_csv(
            matrix_dir / f"rank_{rank:02d}_subject_confusion_row_normalized.csv"
        )


def write_manifest(
    out_dir: Path,
    args: argparse.Namespace,
    analysis_dir: Path,
    rank_configs: pd.DataFrame,
    seeds: Sequence[int],
) -> None:
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "analysis_dir": str(analysis_dir),
        "data_root": str(normalize_user_path(args.data_root)),
        "ranks": parse_int_list(args.ranks),
        "seeds": [int(seed) for seed in seeds],
        "train_size": finite_float(1.0 - float(args.val_size) - float(args.test_size)),
        "val_size": float(args.val_size),
        "test_size": float(args.test_size),
        "note": "Subject-stratified train/inner-validation/test split. Early stopping uses only inner validation; final test is evaluated once after model selection.",
        "rank_configs": rank_configs.to_dict(orient="records"),
    }
    (out_dir / "holdout_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run subject-level train/inner-validation/test repeats for selected sweep leaderboard ranks."
    )
    parser.add_argument("--analysis-dir", default="", help="Analysis output directory containing leaderboard_top_configs.csv. Defaults to latest.")
    parser.add_argument("--analysis-root", default="results_frailty3/_sweep_analyse")
    parser.add_argument("--output-root", default="results_frailty3/_holdout_eval")
    parser.add_argument("--data-root", default="PPG_Testing_05_01_2026")
    parser.add_argument("--ranks", default="1,2,7")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seeds", default="", help="Optional comma-separated seeds. Defaults to 42,10042,...")
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.16, help="Inner-validation fraction of all subjects.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--cnn-target-fs", type=float, default=400.0, help="Deprecated compatibility option; raw windows now stay at --fs.")
    parser.add_argument("--cnn-batch-size", type=int, default=32)
    parser.add_argument("--cnn-lr", type=float, default=1e-3)
    parser.add_argument("--cnn-weight-decay", type=float, default=1e-4)
    parser.add_argument("--cnn-dropout", type=float, default=-1.0, help="-1 keeps each model's historical default.")
    parser.add_argument("--cnn-label-smoothing", type=float, default=0.0)
    parser.add_argument("--cnn-num-workers", type=int, default=0)
    parser.add_argument("--epochs-override", type=int, default=0, help="Debug/ablation override; 0 uses rank config.")
    parser.add_argument("--patience-override", type=int, default=0, help="Debug/ablation override; 0 uses rank config.")
    parser.add_argument("--refresh-features", action="store_true")
    parser.add_argument("--refresh-cnn-windows", action="store_true")
    parser.add_argument("--max-runs", type=int, default=0, help="Debug cap; 0 runs all rank x repeat jobs.")
    parser.add_argument("--dry-run", action="store_true", help="Create manifest and print planned jobs without training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ranks = parse_int_list(args.ranks)
    if args.seeds:
        seeds = parse_int_list(args.seeds)
    else:
        seeds = DEFAULT_SEEDS[: int(args.repeats)]
        while len(seeds) < int(args.repeats):
            seeds.append(int(args.base_seed) + len(seeds) * 10000)
    seeds = seeds[: int(args.repeats)]

    analysis_dir = normalize_user_path(args.analysis_dir) if args.analysis_dir else latest_analysis_dir(normalize_user_path(args.analysis_root))
    out_dir = unique_output_dir(normalize_user_path(args.output_root), ranks)
    report_dir = out_dir / "reports"
    learning_curve_dir = out_dir / "learning_curves"
    report_dir.mkdir(parents=True, exist_ok=True)
    learning_curve_dir.mkdir(parents=True, exist_ok=True)

    rank_configs = load_rank_configs(analysis_dir, ranks)
    write_manifest(out_dir, args, analysis_dir, rank_configs, seeds)

    jobs = [(rank_row, int(rank_row["rank"]), repeat, seed) for _, rank_row in rank_configs.iterrows() for repeat, seed in enumerate(seeds, start=1)]
    if args.max_runs > 0:
        jobs = jobs[: int(args.max_runs)]

    print(f"Holdout output directory: {out_dir}")
    print(f"Analysis dir: {analysis_dir}")
    print(
        "Split: "
        f"train={1.0 - float(args.val_size) - float(args.test_size):.2f}, "
        f"inner validation={float(args.val_size):.2f}, test={float(args.test_size):.2f}"
    )
    print(f"Planned jobs: {len(jobs)}")
    for rank_row, rank, repeat, seed in jobs:
        print(
            f"  rank={rank} repeat={repeat} seed={seed} model={rank_row['model']} "
            f"extra={rank_row['extra_input']} window={rank_row['window_sec']}s overlap={rank_row['overlap_pct']}%"
        )
    if args.dry_run:
        print("Dry run only. No training started.")
        return

    first_config = config_from_rank_row(args, rank_configs.iloc[0])
    features, skipped, cache_path = build_feature_table(first_config, refresh=args.refresh_features)
    print(f"Loaded features: files={len(features)}, subjects={features['subject'].nunique()}, cache={cache_path}")
    run_rows: List[Dict[str, object]] = []
    reports: List[Dict[str, object]] = []
    runs_csv = out_dir / "holdout_runs.csv"
    summary_csv = out_dir / "holdout_summary.csv"

    for job_idx, (rank_row, rank, repeat, seed) in enumerate(jobs, start=1):
        config = config_from_rank_row(args, rank_row)
        print(f"[{job_idx}/{len(jobs)}] rank {rank} repeat {repeat} seed {seed}", flush=True)
        report, report_path = train_eval_holdout_once(
            features=features,
            config=config,
            rank_row=rank_row,
            rank=rank,
            repeat=repeat,
            seed=seed,
            val_size=float(args.val_size),
            test_size=float(args.test_size),
            refresh_cnn_windows=args.refresh_cnn_windows,
            report_dir=report_dir,
            learning_curve_dir=learning_curve_dir,
        )
        run_rows.append(single_run_row(report, rank_row, report_path))
        reports.append(report)
        pd.DataFrame(run_rows).to_csv(runs_csv, index=False)
        build_summary(run_rows).to_csv(summary_csv, index=False)
        write_confusion_aggregates(reports, out_dir)
        print(
            f"  subject BA={report['subject_balanced_accuracy']:.3f}, "
            f"macro F1={report['subject_macro_f1']:.3f}, report={report_path.name}",
            flush=True,
        )

    print(f"Holdout runs CSV: {runs_csv}")
    print(f"Holdout summary CSV: {summary_csv}")
    print(f"Confusion matrices: {out_dir / 'confusion_matrices'}")


if __name__ == "__main__":
    main()
