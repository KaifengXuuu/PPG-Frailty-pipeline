from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from frailty_3class_classifier import CLASS_NAMES, RunConfig, build_feature_table, finite_float
from frailty_3class_classifier import (
    aggregate_by_key,
    build_cnn_window_table,
    cnn_device,
    cnn_predict_proba,
    normalize_extra_input,
    save_learning_curve_artifacts,
    select_extra_feature_columns,
    train_cnn_model,
)
from frailty_3class_holdout_eval import (
    CONFIG_COLUMNS,
    DEFAULT_SEEDS,
    config_from_rank_row,
    evaluate_window_subset,
    latest_analysis_dir,
    load_rank_configs,
    metric_summary,
    normalize_user_path,
    parse_int_list,
    resolve_model_name,
    single_run_row,
    subject_train_val_test_split,
    train_eval_holdout_once,
)


OVERFIT_COLUMNS = [
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
SUMMARY_GROUP_COLUMNS = ["rank"] + CONFIG_COLUMNS + OVERFIT_COLUMNS
RANK_METRICS = [
    "subject_balanced_accuracy_mean",
    "subject_macro_f1_mean",
    "worst_class_f1_mean",
    "worst_class_recall_mean",
    "train_val_balanced_accuracy_gap_mean",
    "subject_balanced_accuracy_std",
]
REFERENCE_CONFIG_ID = "ref_rank2_fixed_epoch"
REFERENCE_REPEATS_DEFAULT = 5
CV_FOLDS_DEFAULT = 5
STAGE1_EPOCHS_DEFAULT = "9,10,15"


def parse_float_list(value: str) -> List[float]:
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def unique_output_dir(root: Path, stage: str, ranks: Sequence[int]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    rank_tag = "rank" + "-".join(str(int(rank)) for rank in ranks)
    base = root / f"{datetime.now().strftime('%Y%m%d_%H%M')}_overfitting_sweep_{stage}_{rank_tag}"
    if not base.exists():
        base.mkdir(parents=True)
        return base
    for idx in range(2, 1000):
        candidate = root / f"{base.name}_{idx:02d}"
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate
    raise RuntimeError(f"Could not create a unique output directory under {root}")


def stage1_grid(args: argparse.Namespace) -> List[Dict[str, object]]:
    epochs = parse_int_list(args.stage1_epochs)
    if not epochs:
        raise ValueError("--stage1-epochs must contain at least one epoch value.")
    base = {
        "lr": 1e-3,
        "weight_decay": 5e-3,
        "dropout": 0.5,
        "label_smoothing": 0.20,
        "max_windows_fraction": 0.9,
    }

    def token(value: object) -> str:
        return str(value).replace(".", "p").replace("-", "m").replace("+", "")

    templates: List[Dict[str, object]] = []
    seen = set()

    def add_template(
        screen_group: str,
        factor: str,
        factor_value: object,
        *,
        lr: float = base["lr"],
        weight_decay: float = base["weight_decay"],
        dropout: float = base["dropout"],
        label_smoothing: float = base["label_smoothing"],
        max_windows_fraction: float = base["max_windows_fraction"],
    ) -> None:
        key = (
            round(float(lr), 10),
            round(float(weight_decay), 10),
            round(float(dropout), 10),
            round(float(label_smoothing), 10),
            round(float(max_windows_fraction), 10),
        )
        if key in seen:
            return
        seen.add(key)
        name = (
            f"{screen_group}_{factor}-{token(factor_value)}_"
            f"lr{float(lr):g}_wd{float(weight_decay):g}_do{float(dropout):g}_"
            f"ls{float(label_smoothing):g}_mw{int(round(float(max_windows_fraction) * 100)):02d}"
        )
        templates.append(
            {
                "name": name,
                "stage1_screen_group": screen_group,
                "stage1_regularization_factor": factor,
                "stage1_regularization_value": str(factor_value),
                "cnn_lr": float(lr),
                "cnn_weight_decay": float(weight_decay),
                "cnn_dropout": float(dropout),
                "cnn_label_smoothing": float(label_smoothing),
                "max_windows_fraction": float(max_windows_fraction),
            }
        )

    add_template("main_effect", "baseline", "wd0p005_do0p5_ls0p2")
    for weight_decay in [1e-3, 2e-3, 5e-3, 1e-2, 2e-2]:
        add_template("main_effect", "weight_decay", weight_decay, weight_decay=weight_decay)
    for dropout in [0.3, 0.4, 0.5, 0.6, 0.7]:
        add_template("main_effect", "dropout", dropout, dropout=dropout)
    for label_smoothing in [0.05, 0.10, 0.15, 0.20, 0.30]:
        add_template("main_effect", "label_smoothing", label_smoothing, label_smoothing=label_smoothing)
    for max_windows_fraction in [1.0, 0.9, 0.7, 0.5]:
        add_template("main_effect", "max_windows_fraction", max_windows_fraction, max_windows_fraction=max_windows_fraction)

    strong_combinations = [
        (1e-3, weight_decay, dropout, label_smoothing, 0.9)
        for weight_decay in [5e-3, 1e-2]
        for dropout in [0.4, 0.5, 0.6]
        for label_smoothing in [0.15, 0.20, 0.30]
    ]
    for lr, weight_decay, dropout, label_smoothing, max_windows_fraction in strong_combinations:
        add_template(
            "focused_combo",
            "combined_regularization",
            f"wd{weight_decay:g}_do{dropout:g}_ls{label_smoothing:g}_mw{max_windows_fraction:g}",
            lr=lr,
            weight_decay=weight_decay,
            dropout=dropout,
            label_smoothing=label_smoothing,
            max_windows_fraction=max_windows_fraction,
        )

    rows: List[Dict[str, object]] = []
    for epoch in epochs:
        for template in templates:
            config_id = f"s1_{len(rows) + 1:03d}"
            rows.append(
                {
                    "overfit_stage": "stage1",
                    "overfit_config_id": config_id,
                    "overfit_config_name": f"{template['name']}_ep{int(epoch)}",
                    "stage1_screen_group": template["stage1_screen_group"],
                    "stage1_regularization_factor": template["stage1_regularization_factor"],
                    "stage1_regularization_value": template["stage1_regularization_value"],
                    "cnn_lr": float(template["cnn_lr"]),
                    "cnn_weight_decay": float(template["cnn_weight_decay"]),
                    "cnn_dropout": float(template["cnn_dropout"]),
                    "cnn_label_smoothing": float(template["cnn_label_smoothing"]),
                    "dynamic_data_mode": "static_only",
                    "stage2_fixed_epoch": 0,
                    "cnn_epochs": int(epoch),
                    "cnn_patience": 0,
                    "max_windows_fraction": float(template["max_windows_fraction"]),
                }
            )
    return rows


def normalize_dynamic_data_mode(value: str) -> str:
    text = str(value).strip().lower().replace("-", "_")
    aliases = {
        "0": "static_only",
        "none": "static_only",
        "static": "static_only",
        "static_only": "static_only",
        "no_dynamic": "static_only",
        "1": "train_all_roles",
        "dynamic": "train_all_roles",
        "all_roles": "train_all_roles",
        "with_dynamic": "train_all_roles",
        "train_all_roles": "train_all_roles",
    }
    if text not in aliases:
        raise ValueError("Unknown dynamic data mode. Use static_only or train_all_roles.")
    return aliases[text]


def stage2_grid(args: argparse.Namespace) -> List[Dict[str, object]]:
    if not args.stage2_source_dir:
        raise ValueError("--stage2-source-dir is required for --stage stage2 or --stage both.")
    summary_path = normalize_user_path(args.stage2_source_dir) / "overfitting_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing stage1 summary: {summary_path}")
    summary = pd.read_csv(summary_path)
    if summary.empty:
        raise ValueError(f"Stage1 summary is empty: {summary_path}")
    for col in ["subject_balanced_accuracy_mean", "subject_macro_f1_mean", "worst_class_f1_mean", "subject_balanced_accuracy_std"]:
        if col not in summary.columns:
            raise KeyError(f"Missing required stage1 summary column: {col}")
    if "overfit_stage" in summary.columns:
        summary = summary[summary["overfit_stage"].astype(str).ne("reference")].copy()
    if "is_reference" in summary.columns:
        summary = summary[~summary["is_reference"].astype(str).str.lower().isin({"true", "1"})].copy()
    if summary.empty:
        raise ValueError("No non-reference stage1 configs are available for stage2.")
    ranked = summary.sort_values(
        by=["subject_balanced_accuracy_mean", "subject_macro_f1_mean", "worst_class_f1_mean", "subject_balanced_accuracy_std"],
        ascending=[False, False, False, True],
        na_position="last",
    )
    top = ranked.head(int(args.stage2_top_n)).copy()
    if int(args.stage2_fixed_epoch) > 0:
        fixed_epoch = int(args.stage2_fixed_epoch)
        fixed_epoch_source = "manual_override"
    else:
        stable = top.sort_values(
            by=["subject_balanced_accuracy_std", "subject_balanced_accuracy_mean", "subject_macro_f1_mean"],
            ascending=[True, False, False],
            na_position="last",
        ).iloc[0]
        fixed_epoch = max(1, int(round(float(stable.get("best_epoch_mean", 1)))))
        fixed_epoch_source = str(stable["overfit_config_id"])
    lrs = parse_float_list(args.stage2_lrs)
    max_fracs = parse_float_list(args.stage2_max_window_fracs)
    dynamic_modes = [item.strip() for item in str(args.stage2_dynamic_data_modes).split(",") if item.strip()]
    rows: List[Dict[str, object]] = []
    seen = set()
    for _, base in top.iterrows():
        source_id = str(base["overfit_config_id"])
        for lr in lrs:
            for max_frac in max_fracs:
                for dynamic_mode in dynamic_modes:
                    dynamic_mode = normalize_dynamic_data_mode(dynamic_mode)
                    key = (
                        source_id,
                        float(lr),
                        int(fixed_epoch),
                        float(max_frac),
                        float(base["cnn_weight_decay"]),
                        float(base["cnn_dropout"]),
                        float(base["cnn_label_smoothing"]),
                        dynamic_mode,
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    config_id = f"s2_{len(rows) + 1:03d}"
                    rows.append(
                        {
                            "overfit_stage": "stage2",
                            "overfit_config_id": config_id,
                            "overfit_config_name": (
                                f"from-{source_id}_lr{lr:g}_ep{fixed_epoch}_"
                                f"mw{int(round(float(max_frac) * 100)):02d}_{dynamic_mode}"
                            ),
                            "stage2_source_config_id": source_id,
                            "stage2_fixed_epoch": int(fixed_epoch),
                            "stage2_fixed_epoch_source_config_id": fixed_epoch_source,
                            "cnn_lr": float(lr),
                            "cnn_weight_decay": float(base["cnn_weight_decay"]),
                            "cnn_dropout": float(base["cnn_dropout"]),
                            "cnn_label_smoothing": float(base["cnn_label_smoothing"]),
                            "dynamic_data_mode": dynamic_mode,
                            "cnn_patience": 0,
                            "max_windows_fraction": float(max_frac),
                        }
                    )
    return rows


def build_grid(args: argparse.Namespace) -> List[Dict[str, object]]:
    grids: List[Dict[str, object]] = []
    if args.stage in {"stage1", "both"}:
        grids.extend(stage1_grid(args))
    if args.stage in {"stage2", "both"}:
        grids.extend(stage2_grid(args))
    if args.max_configs > 0:
        grids = grids[: int(args.max_configs)]
    return grids


def seeds_from_args(args: argparse.Namespace) -> List[int]:
    if args.seeds:
        seeds = parse_int_list(args.seeds)
    else:
        seeds = DEFAULT_SEEDS[: int(args.repeats)]
        while len(seeds) < int(args.repeats):
            seeds.append(int(args.base_seed) + len(seeds) * 10000)
    return seeds[: int(args.repeats)]


def reference_seeds_from_args(args: argparse.Namespace) -> List[int]:
    repeats = max(0, int(args.reference_repeats))
    if repeats == 0:
        return []
    if args.reference_seeds:
        seeds = parse_int_list(args.reference_seeds)
    elif args.seeds:
        seeds = parse_int_list(args.seeds)
    else:
        seeds = DEFAULT_SEEDS[:repeats]
        while len(seeds) < repeats:
            seeds.append(int(args.base_seed) + len(seeds) * 10000)
    while len(seeds) < repeats:
        seeds.append(int(args.base_seed) + len(seeds) * 10000)
    return seeds[:repeats]


def requested_cv_folds(args: argparse.Namespace) -> int:
    legacy_stage2_folds = int(getattr(args, "stage2_cv_folds", 0) or 0)
    if legacy_stage2_folds > 0:
        return legacy_stage2_folds
    return int(args.cv_folds)


def reference_grid_row(epoch: int = 0) -> Dict[str, object]:
    epoch = int(epoch)
    config_id = f"{REFERENCE_CONFIG_ID}_ep{epoch}" if epoch > 0 else REFERENCE_CONFIG_ID
    config_name = "original_rank_parameters_no_early_stopping_static_only"
    if epoch > 0:
        config_name = f"{config_name}_ep{epoch}"
    return {
        "overfit_stage": "reference",
        "overfit_config_id": config_id,
        "overfit_config_name": config_name,
        "stage1_screen_group": "reference",
        "stage1_regularization_factor": "reference",
        "stage1_regularization_value": str(epoch) if epoch > 0 else "",
        "stage2_source_config_id": "",
        "cnn_lr": 1e-3,
        "cnn_weight_decay": 1e-4,
        "cnn_dropout": -1.0,
        "cnn_label_smoothing": 0.0,
        "dynamic_data_mode": "static_only",
        "stage2_fixed_epoch": int(epoch),
        "cnn_epochs": int(epoch),
        "cnn_patience": 0,
        "max_windows_fraction": -1.0,
        "is_reference": True,
    }


def reference_grid_rows(args: argparse.Namespace, grid: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    epochs = sorted(
        {
            int(value)
            for item in grid
            for value in (item.get("cnn_epochs", 0), item.get("stage2_fixed_epoch", 0))
            if int(value or 0) > 0
        }
    )
    if not epochs and args.stage in {"stage1", "both"}:
        epochs = parse_int_list(args.stage1_epochs)
    if not epochs:
        epochs = [0]
    return [reference_grid_row(epoch) for epoch in epochs]


def original_config_from_rank_row(args: argparse.Namespace, row: pd.Series, grid: Optional[Dict[str, object]] = None) -> RunConfig:
    config = RunConfig(
        data_root=str(normalize_user_path(args.data_root)),
        folds=1,
        seed=int(args.base_seed),
        cnn_target_fs=float(args.cnn_target_fs),
        cnn_batch_size=int(args.cnn_batch_size),
        cnn_lr=1e-3,
        cnn_num_workers=int(args.cnn_num_workers),
        cnn_weight_decay=1e-4,
        cnn_dropout=-1.0,
        cnn_label_smoothing=0.0,
        role_mode="static_only",
    )
    config.cnn_seq_sec = float(row["window_sec"])
    config.cnn_hop_sec = float(row["hop_sec"])
    config.cnn_max_windows_fraction = float(row["max_windows_fraction"])
    reference_epoch = int((grid or {}).get("cnn_epochs") or 0)
    config.cnn_epochs = reference_epoch if reference_epoch > 0 else int(row["cnn_epochs"])
    config.cnn_patience = 0
    config.cnn_select_best_epoch = False
    config.extra_input = normalize_extra_input(str(row["extra_input"]))
    return config


def apply_overfit_config(config: RunConfig, grid: Dict[str, object], args: argparse.Namespace) -> RunConfig:
    config.cnn_lr = float(grid["cnn_lr"])
    config.cnn_weight_decay = float(grid["cnn_weight_decay"])
    config.cnn_dropout = float(grid["cnn_dropout"])
    config.cnn_label_smoothing = float(grid["cnn_label_smoothing"])
    config.cnn_patience = int(grid["cnn_patience"])
    config.cnn_max_windows_fraction = float(grid["max_windows_fraction"])
    config.role_mode = "all_roles" if normalize_dynamic_data_mode(grid.get("dynamic_data_mode", "static_only")) == "train_all_roles" else "static_only"
    if "cnn_epochs" in grid:
        config.cnn_epochs = int(grid["cnn_epochs"])
    if str(grid.get("overfit_stage", "")) == "stage2":
        config.cnn_epochs = int(grid["stage2_fixed_epoch"])
    if str(grid.get("overfit_stage", "")) in {"stage1", "stage2"}:
        config.cnn_patience = 0
        config.cnn_select_best_epoch = False
    if int(args.epochs_override) > 0:
        config.cnn_epochs = int(args.epochs_override)
    return config


def best_epoch_history_metrics(report: Dict[str, object]) -> Dict[str, float]:
    selected_rows: List[Dict[str, object]] = []
    for fold in report.get("folds") or []:
        history = fold.get("history") or []
        best_epoch = int(fold.get("best_epoch") or 0)
        selected = None
        for row in history:
            if int(row.get("epoch", -1)) == best_epoch:
                selected = row
                break
        if selected is None and history:
            selected = history[-1]
        if selected is not None:
            selected_rows.append(selected)
    if not selected_rows:
        return {
            "best_epoch_train_balanced_accuracy": np.nan,
            "best_epoch_val_balanced_accuracy": np.nan,
            "train_val_balanced_accuracy_gap": np.nan,
            "best_epoch_train_loss": np.nan,
            "best_epoch_val_loss": np.nan,
            "val_train_loss_gap": np.nan,
        }

    def mean_selected(name: str) -> float:
        values = [safe_float(row.get(name)) for row in selected_rows]
        values = [value for value in values if math.isfinite(value)]
        return float(np.mean(values)) if values else np.nan

    train_ba = mean_selected("train_balanced_accuracy")
    val_ba = mean_selected("val_balanced_accuracy")
    train_loss = mean_selected("train_loss")
    val_loss = mean_selected("val_loss")
    return {
        "best_epoch_train_balanced_accuracy": train_ba,
        "best_epoch_val_balanced_accuracy": val_ba,
        "train_val_balanced_accuracy_gap": train_ba - val_ba if math.isfinite(train_ba) and math.isfinite(val_ba) else np.nan,
        "best_epoch_train_loss": train_loss,
        "best_epoch_val_loss": val_loss,
        "val_train_loss_gap": val_loss - train_loss if math.isfinite(train_loss) and math.isfinite(val_loss) else np.nan,
    }


def safe_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return np.nan
    return out if math.isfinite(out) else np.nan


def scale_extra_between_tables(
    train_features: pd.DataFrame,
    transform_features: pd.DataFrame,
    cols: Sequence[str],
) -> np.ndarray:
    train_raw = train_features[list(cols)].to_numpy(dtype=np.float32)
    transform_raw = transform_features[list(cols)].to_numpy(dtype=np.float32)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_imputed = imputer.fit_transform(train_raw)
    scaler.fit(train_imputed)
    return scaler.transform(imputer.transform(transform_raw)).astype(np.float32)


def train_eval_with_train_all_roles_once(
    *,
    static_features: pd.DataFrame,
    all_role_features: pd.DataFrame,
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
    import time

    start_time = time.perf_counter()
    static_features = static_features.reset_index(drop=True)
    all_role_features = all_role_features.reset_index(drop=True)
    config.seed = int(seed)
    model_name = resolve_model_name(rank_row)
    train_idx_static, val_idx_static, test_idx_static, split_meta = subject_train_val_test_split(
        static_features,
        seed=seed,
        val_size=val_size,
        test_size=test_size,
    )
    train_subjects = set(str(subject) for subject in split_meta["train_subjects"])
    train_idx_all = all_role_features.index[all_role_features["subject"].astype(str).isin(train_subjects)].to_numpy(dtype=np.int64)
    if len(train_idx_all) == 0:
        raise RuntimeError("No all-role training files found for train subjects.")

    train_config = RunConfig(**asdict(config))
    train_config.role_mode = "all_roles"
    eval_config = RunConfig(**asdict(config))
    eval_config.role_mode = "static_only"

    x_train_all, y_train_all, _subject_train_all, file_train_all, train_cache_path = build_cnn_window_table(
        all_role_features,
        train_config,
        refresh=refresh_cnn_windows,
    )
    x_static, y_static, subject_static, file_static, eval_cache_path = build_cnn_window_table(
        static_features,
        eval_config,
        refresh=refresh_cnn_windows,
    )
    train_mask = np.isin(file_train_all, train_idx_all)
    val_mask = np.isin(file_static, val_idx_static)
    test_mask = np.isin(file_static, test_idx_static)
    if not np.any(train_mask) or not np.any(val_mask) or not np.any(test_mask):
        raise RuntimeError("Empty train, validation, or test window set after dynamic augmentation split.")

    extra_cols = select_extra_feature_columns(static_features, config.extra_input)
    train_window_extra: Optional[np.ndarray] = None
    static_window_extra: Optional[np.ndarray] = None
    if extra_cols:
        train_file_extra = scale_extra_between_tables(all_role_features.iloc[train_idx_all], all_role_features, extra_cols)
        static_file_extra = scale_extra_between_tables(all_role_features.iloc[train_idx_all], static_features, extra_cols)
        train_window_extra = train_file_extra[file_train_all]
        static_window_extra = static_file_extra[file_static]

    model, train_info = train_cnn_model(
        x_train_all[train_mask],
        y_train_all[train_mask],
        config,
        seed=seed,
        model_name=model_name,
        x_val=x_static[val_mask],
        y_val=y_static[val_mask],
        extra_train=train_window_extra[train_mask] if train_window_extra is not None else None,
        extra_val=static_window_extra[val_mask] if static_window_extra is not None else None,
    )

    validation_eval = evaluate_window_subset(
        model=model,
        x_win=x_static,
        y_win=y_static,
        subject_win=subject_static,
        file_win=file_static,
        mask=val_mask,
        batch_size=config.cnn_batch_size,
        extra=static_window_extra,
    )
    test_eval = evaluate_window_subset(
        model=model,
        x_win=x_static,
        y_win=y_static,
        subject_win=subject_static,
        file_win=file_static,
        mask=test_mask,
        batch_size=config.cnn_batch_size,
        extra=static_window_extra,
    )

    fold_summary = {
        "fold": int(repeat),
        "n_train_files": int(len(train_idx_all)),
        "n_static_train_files": int(len(train_idx_static)),
        "n_dynamic_added_train_files": int(max(0, len(train_idx_all) - len(train_idx_static))),
        "n_val_files": int(len(val_idx_static)),
        "n_test_files": int(len(test_idx_static)),
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
        "early_stopping_source": "inner_validation_set_static_roles",
        "dynamic_data_mode": "train_all_roles",
        "train_role_mode": "all_roles",
        "validation_role_mode": "static_only",
        "test_role_mode": "static_only",
        "n_files": int(len(static_features)),
        "n_all_role_files": int(len(all_role_features)),
        "n_subjects": int(static_features["subject"].nunique()),
        "n_windows": int(len(y_static)),
        "n_train_windows": int(np.sum(train_mask)),
        "n_val_windows": int(np.sum(val_mask)),
        "n_test_windows": int(np.sum(test_mask)),
        "n_static_train_files": int(len(train_idx_static)),
        "n_all_role_train_files": int(len(train_idx_all)),
        "n_dynamic_added_train_files": int(max(0, len(train_idx_all) - len(train_idx_static))),
        "extra_input": normalize_extra_input(config.extra_input),
        "n_extra_features": int(len(extra_cols)),
        "cnn_cache": str(eval_cache_path),
        "train_all_roles_cnn_cache": str(train_cache_path),
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
        f"seq{config.cnn_seq_sec:g}_ov{float(rank_row['overlap_pct']):g}_dyn-train-all_seed{int(seed)}"
    ).replace(".", "p")
    report.update(save_learning_curve_artifacts(report, out_dir=learning_curve_dir, filename_prefix=tag))
    report["duration_sec"] = float(time.perf_counter() - start_time)
    report_path = report_dir / f"{tag}_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report, report_path


def train_eval_overfit_once(
    *,
    static_features: pd.DataFrame,
    all_role_features: pd.DataFrame,
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
    dynamic_data_mode: str,
) -> Tuple[Dict[str, object], Path]:
    mode = normalize_dynamic_data_mode(dynamic_data_mode)
    if mode == "static_only":
        config.role_mode = "static_only"
        report, path = train_eval_holdout_once(
            features=static_features,
            config=config,
            rank_row=rank_row,
            rank=rank,
            repeat=repeat,
            seed=seed,
            val_size=val_size,
            test_size=test_size,
            refresh_cnn_windows=refresh_cnn_windows,
            report_dir=report_dir,
            learning_curve_dir=learning_curve_dir,
        )
        report["dynamic_data_mode"] = "static_only"
        report["train_role_mode"] = "static_only"
        report["validation_role_mode"] = "static_only"
        report["test_role_mode"] = "static_only"
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report, path
    return train_eval_with_train_all_roles_once(
        static_features=static_features,
        all_role_features=all_role_features,
        config=config,
        rank_row=rank_row,
        rank=rank,
        repeat=repeat,
        seed=seed,
        val_size=val_size,
        test_size=test_size,
        refresh_cnn_windows=refresh_cnn_windows,
        report_dir=report_dir,
        learning_curve_dir=learning_curve_dir,
    )


def train_eval_groupkfold_once(
    *,
    static_features: pd.DataFrame,
    all_role_features: pd.DataFrame,
    config: RunConfig,
    rank_row: pd.Series,
    rank: int,
    repeat: int,
    seed: int,
    cv_folds: int,
    dynamic_data_mode: str,
    refresh_cnn_windows: bool,
    report_dir: Path,
    learning_curve_dir: Path,
) -> Tuple[Dict[str, object], Path]:
    import time

    start_time = time.perf_counter()
    static_features = static_features.reset_index(drop=True)
    all_role_features = all_role_features.reset_index(drop=True)
    config.seed = int(seed)
    mode = normalize_dynamic_data_mode(dynamic_data_mode)
    model_name = resolve_model_name(rank_row)
    y_file = static_features["label"].to_numpy(dtype=int)
    groups = static_features["subject"].astype(str).to_numpy()
    subject_labels = static_features.groupby("subject")["label"].first()
    min_class_subjects = int(subject_labels.value_counts().min())
    n_splits = max(2, min(int(cv_folds), min_class_subjects))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))

    eval_config = RunConfig(**asdict(config))
    eval_config.role_mode = "static_only"
    x_static, y_static, subject_static, file_static, eval_cache_path = build_cnn_window_table(
        static_features,
        eval_config,
        refresh=refresh_cnn_windows,
    )
    train_config = RunConfig(**asdict(config))
    train_config.role_mode = "all_roles" if mode == "train_all_roles" else "static_only"
    if mode == "train_all_roles":
        x_train_source, y_train_source, _subject_train_source, file_train_source, train_cache_path = build_cnn_window_table(
            all_role_features,
            train_config,
            refresh=refresh_cnn_windows,
        )
        train_feature_source = all_role_features
    else:
        x_train_source, y_train_source, _subject_train_source, file_train_source = x_static, y_static, subject_static, file_static
        train_cache_path = eval_cache_path
        train_feature_source = static_features

    extra_cols = select_extra_feature_columns(static_features, config.extra_input)
    device = cnn_device()
    window_true: List[int] = []
    window_pred: List[int] = []
    file_true: List[int] = []
    file_pred: List[int] = []
    subject_true: List[int] = []
    subject_pred: List[int] = []
    fold_summaries: List[Dict[str, object]] = []
    total_train_windows = 0
    total_val_windows = 0

    for fold, (train_idx_static, val_idx_static) in enumerate(cv.split(np.zeros(len(static_features)), y_file, groups), start=1):
        train_subjects = set(static_features.iloc[train_idx_static]["subject"].astype(str))
        if mode == "train_all_roles":
            train_idx_source = train_feature_source.index[
                train_feature_source["subject"].astype(str).isin(train_subjects)
            ].to_numpy(dtype=np.int64)
        else:
            train_idx_source = train_idx_static
        train_mask = np.isin(file_train_source, train_idx_source)
        val_mask = np.isin(file_static, val_idx_static)
        if not np.any(train_mask) or not np.any(val_mask):
            raise RuntimeError(f"Empty train or validation windows in CV fold {fold}.")

        train_window_extra: Optional[np.ndarray] = None
        static_window_extra: Optional[np.ndarray] = None
        if extra_cols:
            train_file_extra = scale_extra_between_tables(train_feature_source.iloc[train_idx_source], train_feature_source, extra_cols)
            static_file_extra = scale_extra_between_tables(train_feature_source.iloc[train_idx_source], static_features, extra_cols)
            train_window_extra = train_file_extra[file_train_source]
            static_window_extra = static_file_extra[file_static]

        model, fold_info = train_cnn_model(
            x_train_source[train_mask],
            y_train_source[train_mask],
            config,
            seed=int(seed) + fold,
            model_name=model_name,
            x_val=x_static[val_mask],
            y_val=y_static[val_mask],
            extra_train=train_window_extra[train_mask] if train_window_extra is not None else None,
            extra_val=static_window_extra[val_mask] if static_window_extra is not None else None,
        )
        probs = cnn_predict_proba(
            model,
            x_static[val_mask],
            config.cnn_batch_size,
            device,
            extra=static_window_extra[val_mask] if static_window_extra is not None else None,
        )
        preds = np.argmax(probs, axis=1).astype(int)
        y_val = y_static[val_mask].astype(int)
        window_true.extend(y_val.tolist())
        window_pred.extend(preds.tolist())
        fold_file_true, fold_file_pred = aggregate_by_key(probs, y_val, file_static[val_mask].tolist())
        fold_subject_true, fold_subject_pred = aggregate_by_key(probs, y_val, subject_static[val_mask].tolist())
        file_true.extend(fold_file_true)
        file_pred.extend(fold_file_pred)
        subject_true.extend(fold_subject_true)
        subject_pred.extend(fold_subject_pred)
        total_train_windows += int(np.sum(train_mask))
        total_val_windows += int(np.sum(val_mask))
        fold_summaries.append(
            {
                "fold": int(fold),
                "n_train_subjects": int(len(train_subjects)),
                "n_val_subjects": int(len(np.unique(groups[val_idx_static]))),
                "n_train_files": int(len(train_idx_source)),
                "n_static_train_files": int(len(train_idx_static)),
                "n_dynamic_added_train_files": int(max(0, len(train_idx_source) - len(train_idx_static))),
                "n_val_files": int(len(val_idx_static)),
                "n_train_windows": int(np.sum(train_mask)),
                "n_val_windows": int(np.sum(val_mask)),
                "val_subjects": sorted(map(str, np.unique(groups[val_idx_static]))),
                "best_epoch": int(fold_info["best_epoch"]),
                "best_window_balanced_accuracy": fold_info["best_window_balanced_accuracy"],
                "history": fold_info["history"],
                "file_balanced_accuracy": finite_float(balanced_accuracy_score(fold_file_true, fold_file_pred)),
                "subject_balanced_accuracy": finite_float(balanced_accuracy_score(fold_subject_true, fold_subject_pred)),
            }
        )

    metrics = {
        "window_balanced_accuracy": finite_float(balanced_accuracy_score(window_true, window_pred)),
        "window_macro_f1": finite_float(f1_score(window_true, window_pred, labels=[0, 1, 2], average="macro", zero_division=0)),
        "file_balanced_accuracy": finite_float(balanced_accuracy_score(file_true, file_pred)),
        "file_macro_f1": finite_float(f1_score(file_true, file_pred, labels=[0, 1, 2], average="macro", zero_division=0)),
        "subject_balanced_accuracy": finite_float(balanced_accuracy_score(subject_true, subject_pred)),
        "subject_macro_f1": finite_float(f1_score(subject_true, subject_pred, labels=[0, 1, 2], average="macro", zero_division=0)),
        "window_confusion_matrix": confusion_matrix(window_true, window_pred, labels=[0, 1, 2]).tolist(),
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
    mean_train_subjects = int(round(float(np.mean([fold["n_train_subjects"] for fold in fold_summaries]))))
    mean_val_subjects = int(round(float(np.mean([fold["n_val_subjects"] for fold in fold_summaries]))))
    report: Dict[str, object] = {
        "model": str(rank_row["model"]),
        "resolved_model": model_name,
        "rank": int(rank),
        "repeat": int(repeat),
        "seed": int(seed),
        "eval_protocol": "stratified_group_kfold",
        "requested_cv_folds": int(cv_folds),
        "n_splits": int(n_splits),
        "holdout_split": {
            "split_design": "stratified_group_kfold",
            "split_seed": int(seed),
            "requested_cv_folds": int(cv_folds),
            "n_splits": int(n_splits),
            "n_train_subjects": mean_train_subjects,
            "n_val_subjects": mean_val_subjects,
            "n_test_subjects": int(static_features["subject"].nunique()),
        },
        "early_stopping_source": "none_final_epoch_fixed" if not bool(config.cnn_select_best_epoch) else "cv_validation_fold",
        "dynamic_data_mode": mode,
        "train_role_mode": "all_roles" if mode == "train_all_roles" else "static_only",
        "validation_role_mode": "static_only",
        "test_role_mode": "static_only",
        "n_files": int(len(static_features)),
        "n_all_role_files": int(len(all_role_features)),
        "n_subjects": int(static_features["subject"].nunique()),
        "n_windows": int(len(y_static)),
        "n_train_windows": int(total_train_windows),
        "n_val_windows": int(total_val_windows),
        "n_test_windows": int(total_val_windows),
        "n_static_train_files": int(round(float(np.mean([fold["n_static_train_files"] for fold in fold_summaries])))),
        "n_all_role_train_files": int(round(float(np.mean([fold["n_train_files"] for fold in fold_summaries])))),
        "n_dynamic_added_train_files": int(round(float(np.mean([fold["n_dynamic_added_train_files"] for fold in fold_summaries])))),
        "extra_input": normalize_extra_input(config.extra_input),
        "n_extra_features": int(len(extra_cols)),
        "cnn_cache": str(eval_cache_path),
        "train_all_roles_cnn_cache": str(train_cache_path),
        "validation_window_balanced_accuracy": metrics["window_balanced_accuracy"],
        "validation_window_macro_f1": metrics["window_macro_f1"],
        "validation_file_balanced_accuracy": metrics["file_balanced_accuracy"],
        "validation_file_macro_f1": metrics["file_macro_f1"],
        "validation_subject_balanced_accuracy": metrics["subject_balanced_accuracy"],
        "validation_subject_macro_f1": metrics["subject_macro_f1"],
        "validation_window_confusion_matrix": metrics["window_confusion_matrix"],
        "validation_file_confusion_matrix": metrics["file_confusion_matrix"],
        "validation_subject_confusion_matrix": metrics["subject_confusion_matrix"],
        "validation_file_classification_report": metrics["file_classification_report"],
        "validation_subject_classification_report": metrics["subject_classification_report"],
        **metrics,
        "folds": fold_summaries,
        "feature_columns": extra_cols,
        "config": asdict(config),
    }
    tag = (
        f"rank{int(rank):02d}_r{int(repeat)}_{rank_row['model']}_extra-{config.extra_input}_"
        f"seq{config.cnn_seq_sec:g}_ov{float(rank_row['overlap_pct']):g}_cv{int(n_splits)}_"
        f"{mode}_seed{int(seed)}"
    ).replace(".", "p")
    report.update(save_learning_curve_artifacts(report, out_dir=learning_curve_dir, filename_prefix=tag))
    report["duration_sec"] = float(time.perf_counter() - start_time)
    report_path = report_dir / f"{tag}_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report, report_path


def overfit_run_row(report: Dict[str, object], rank_row: pd.Series, report_path: Path, grid: Dict[str, object]) -> Dict[str, object]:
    row = single_run_row(report, rank_row, report_path)
    for col in OVERFIT_COLUMNS:
        row[col] = grid.get(col, "")
    row["stage2_source_config_id"] = grid.get("stage2_source_config_id", "")
    row["is_reference"] = bool(grid.get("is_reference", False))
    row["overfit_grid_cnn_patience"] = int(grid["cnn_patience"])
    row["overfit_grid_max_windows_fraction"] = float(grid["max_windows_fraction"])
    row["train_role_mode"] = str(report.get("train_role_mode", "static_only"))
    row["validation_role_mode"] = str(report.get("validation_role_mode", "static_only"))
    row["test_role_mode"] = str(report.get("test_role_mode", "static_only"))
    row["eval_protocol"] = str(report.get("eval_protocol", ""))
    row["requested_cv_folds"] = int(report.get("requested_cv_folds", 0) or 0)
    row["n_splits"] = int(report.get("n_splits", 0) or 0)
    row["early_stopping_source"] = str(report.get("early_stopping_source", ""))
    row["n_static_train_files"] = int(report.get("n_static_train_files", (report.get("folds") or [{}])[0].get("n_train_files", 0)))
    row["n_all_role_train_files"] = int(report.get("n_all_role_train_files", report.get("n_static_train_files", 0)))
    row["n_dynamic_added_train_files"] = int(report.get("n_dynamic_added_train_files", 0))
    fold_best_epochs = [safe_float(fold.get("best_epoch")) for fold in report.get("folds", [])]
    fold_best_epochs = [value for value in fold_best_epochs if math.isfinite(value)]
    if fold_best_epochs:
        row["best_epoch"] = float(np.mean(fold_best_epochs))
    fold_best_scores = [safe_float(fold.get("best_window_balanced_accuracy")) for fold in report.get("folds", [])]
    fold_best_scores = [value for value in fold_best_scores if math.isfinite(value)]
    if fold_best_scores:
        row["best_window_balanced_accuracy"] = float(np.mean(fold_best_scores))
    row.update(best_epoch_history_metrics(report))
    return row


def build_overfit_summary(run_rows: List[Dict[str, object]]) -> pd.DataFrame:
    runs = pd.DataFrame(run_rows)
    if runs.empty:
        return runs
    rows: List[Dict[str, object]] = []
    group_cols = [col for col in SUMMARY_GROUP_COLUMNS if col in runs.columns]
    metric_cols = [
        "validation_subject_balanced_accuracy",
        "validation_subject_macro_f1",
        "subject_balanced_accuracy",
        "subject_macro_f1",
        "worst_class_recall",
        "worst_class_f1",
        "best_epoch",
        "duration_sec",
        "best_epoch_train_balanced_accuracy",
        "best_epoch_val_balanced_accuracy",
        "train_val_balanced_accuracy_gap",
        "best_epoch_train_loss",
        "best_epoch_val_loss",
        "val_train_loss_gap",
    ]
    for values, sub in runs.groupby(group_cols, dropna=False, sort=False):
        row = {col: value for col, value in zip(group_cols, values)}
        row["n_repeats"] = int(sub["repeat"].nunique())
        for metric in metric_cols:
            if metric not in sub:
                continue
            stats = metric_summary(sub[metric])
            for stat, stat_value in stats.items():
                row[f"{metric}_{stat}"] = stat_value
        rows.append(row)
    summary = pd.DataFrame(rows)
    sort_cols = [col for col in RANK_METRICS if col in summary.columns]
    ascending = [False, False, False, False, True, True][: len(sort_cols)]
    if sort_cols:
        summary = summary.sort_values(sort_cols, ascending=ascending, na_position="last", kind="mergesort").reset_index(drop=True)
    summary.insert(0, "overfit_rank", np.arange(1, len(summary) + 1))
    return summary


def write_overfit_confusion_aggregates(report_items: Sequence[Tuple[Dict[str, object], Dict[str, object]]], out_dir: Path) -> None:
    matrix_dir = out_dir / "confusion_matrices"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    grouped: Dict[Tuple[int, str], List[Dict[str, object]]] = {}
    for report, grid in report_items:
        grouped.setdefault((int(report["rank"]), str(grid["overfit_config_id"])), []).append(report)
    for (rank, config_id), reports in grouped.items():
        total = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=float)
        for report in reports:
            total += np.asarray(report["subject_confusion_matrix"], dtype=float)
        row_sums = total.sum(axis=1, keepdims=True)
        normalized = np.divide(total, row_sums, out=np.zeros_like(total), where=row_sums > 0)
        index = list(CLASS_NAMES)
        columns = list(CLASS_NAMES)
        stem = f"rank_{rank:02d}_{config_id}_subject_confusion"
        pd.DataFrame(total.astype(int), index=index, columns=columns).to_csv(matrix_dir / f"{stem}_counts.csv")
        pd.DataFrame(normalized, index=index, columns=columns).to_csv(matrix_dir / f"{stem}_row_normalized.csv")


def write_manifest(
    out_dir: Path,
    args: argparse.Namespace,
    analysis_dir: Path,
    rank_configs: pd.DataFrame,
    grid: Sequence[Dict[str, object]],
    seeds: Sequence[int],
    reference_seeds: Sequence[int],
) -> None:
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "sweep_name": "overfitting_sweep",
        "stage": args.stage,
        "analysis_dir": str(analysis_dir),
        "data_root": str(normalize_user_path(args.data_root)),
        "ranks": parse_int_list(args.ranks),
        "seeds": [int(seed) for seed in seeds],
        "repeats": int(args.repeats),
        "reference_repeats": int(len(reference_seeds)),
        "reference_seeds": [int(seed) for seed in reference_seeds],
        "reference_configs": reference_grid_rows(args, grid),
        "cv_folds": int(requested_cv_folds(args)),
        "train_size": finite_float(1.0 - float(args.val_size) - float(args.test_size)),
        "val_size": float(args.val_size),
        "test_size": float(args.test_size),
        "stage_design": {
            "stage1": (
                "Expanded no-early-stopping regularization main-effect sweep. "
                "The grid includes one-factor-at-a-time ranges for weight decay, dropout, "
                "label smoothing, and max_windows_fraction, plus a small set of strong "
                "combined-regularization templates. Each template is evaluated at fixed "
                "epochs from --stage1-epochs."
            ),
            "stage2": (
                "Take top non-reference stage1 configs, use the fixed epoch from the most stable top config, "
                "disable early stopping for tuned configs, and evaluate with StratifiedGroupKFold."
            ),
            "all_stages_eval_protocol": "5-fold StratifiedGroupKFold by subject unless --cv-folds is changed.",
            "reference": "Reference configs use original rank parameters with cnn_select_best_epoch=False, cnn_patience=0, and the same fixed epoch values as the active grid.",
            "stage1_epochs": parse_int_list(args.stage1_epochs),
            "stage2_top_n": int(args.stage2_top_n),
            "stage2_requested_cv_folds": int(requested_cv_folds(args)),
            "stage2_fixed_epoch_override": int(args.stage2_fixed_epoch),
            "dynamic_data_mode": {
                "static_only": "Current behavior: train, validation, and test use only B/R1/R2/R3/R4.",
                "train_all_roles": "Data augmentation: split and evaluate on B/R1/R2/R3/R4, but train on all roles for train subjects only.",
            },
        },
        "rank_configs": rank_configs.to_dict(orient="records"),
        "grid": list(grid),
    }
    (out_dir / "overfitting_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the two-stage overfitting sweep for selected leaderboard ranks.")
    parser.add_argument("--analysis-dir", default="", help="Analysis directory containing leaderboard_top_configs.csv. Defaults to latest.")
    parser.add_argument("--analysis-root", default="results_frailty3/_sweep_analyse")
    parser.add_argument("--output-root", default="results_frailty3/_overfitting_sweep")
    parser.add_argument("--data-root", default="PPG_Testing_05_01_2026")
    parser.add_argument("--ranks", default="2")
    parser.add_argument("--stage", choices=("stage1", "stage2", "both"), default="stage1")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seeds", default="")
    parser.add_argument(
        "--reference-repeats",
        type=int,
        default=REFERENCE_REPEATS_DEFAULT,
        help="Append this many original-parameter, no-early-stopping reference repeats after the sweep grid.",
    )
    parser.add_argument("--reference-seeds", default="", help="Optional comma-separated seeds for the reference repeats.")
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=CV_FOLDS_DEFAULT, help="StratifiedGroupKFold folds for all stages.")
    parser.add_argument("--val-size", type=float, default=0.16)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--cnn-target-fs", type=float, default=400.0, help="Deprecated compatibility option; raw windows now stay at --fs.")
    parser.add_argument("--cnn-batch-size", type=int, default=32)
    parser.add_argument("--cnn-lr", type=float, default=1e-3)
    parser.add_argument("--cnn-weight-decay", type=float, default=1e-4)
    parser.add_argument("--cnn-dropout", type=float, default=-1.0)
    parser.add_argument("--cnn-label-smoothing", type=float, default=0.0)
    parser.add_argument("--cnn-num-workers", type=int, default=0)
    parser.add_argument("--epochs-override", type=int, default=0)
    parser.add_argument("--patience-override", type=int, default=0)
    parser.add_argument("--stage1-epochs", default=STAGE1_EPOCHS_DEFAULT)
    parser.add_argument("--stage2-source-dir", default="")
    parser.add_argument("--stage2-top-n", type=int, default=2)
    parser.add_argument("--stage2-lrs", default="0.001,0.0005,0.0002")
    parser.add_argument("--stage2-patiences", default="8,12,16", help="Deprecated for stage2 fixed-epoch mode; kept for CLI compatibility.")
    parser.add_argument("--stage2-max-window-fracs", default="0.9,0.7,0.5")
    parser.add_argument("--stage2-fixed-epoch", type=int, default=0, help="Override fixed epoch for stage2; 0 derives it from stage1 stability.")
    parser.add_argument("--stage2-cv-folds", type=int, default=0, help="Deprecated alias for --cv-folds; overrides --cv-folds when > 0.")
    parser.add_argument(
        "--stage2-dynamic-data-modes",
        default="static_only,train_all_roles",
        help="Stage2 data augmentation modes: static_only and/or train_all_roles.",
    )
    parser.add_argument("--refresh-features", action="store_true")
    parser.add_argument("--refresh-cnn-windows", action="store_true")
    parser.add_argument("--max-configs", type=int, default=0, help="Debug cap on grid configs; 0 uses all.")
    parser.add_argument("--max-runs", type=int, default=0, help="Debug cap on jobs; 0 runs all rank x config x repeat jobs.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ranks = parse_int_list(args.ranks)
    seeds = seeds_from_args(args)
    reference_seeds = reference_seeds_from_args(args)
    cv_folds = requested_cv_folds(args)
    analysis_dir = normalize_user_path(args.analysis_dir) if args.analysis_dir else latest_analysis_dir(normalize_user_path(args.analysis_root))
    out_dir = unique_output_dir(normalize_user_path(args.output_root), args.stage, ranks)
    report_dir = out_dir / "reports"
    learning_curve_dir = out_dir / "learning_curves"
    report_dir.mkdir(parents=True, exist_ok=True)
    learning_curve_dir.mkdir(parents=True, exist_ok=True)

    rank_configs = load_rank_configs(analysis_dir, ranks)
    grid = build_grid(args)
    reference_grid = reference_grid_rows(args, grid)
    write_manifest(out_dir, args, analysis_dir, rank_configs, grid, seeds, reference_seeds)

    grid_jobs = [
        (rank_row, int(rank_row["rank"]), grid_row, repeat, seed)
        for _, rank_row in rank_configs.iterrows()
        for grid_row in grid
        for repeat, seed in enumerate(seeds, start=1)
    ]
    reference_jobs = [
        (rank_row, int(rank_row["rank"]), reference_row, repeat, seed)
        for _, rank_row in rank_configs.iterrows()
        for reference_row in reference_grid
        for repeat, seed in enumerate(reference_seeds, start=1)
    ]
    jobs = grid_jobs + reference_jobs
    if args.max_runs > 0:
        jobs = jobs[: int(args.max_runs)]

    print(f"Overfitting sweep output directory: {out_dir}")
    print(f"Analysis dir: {analysis_dir}")
    print(
        f"Stage: {args.stage}; ranks={ranks}; grid configs={len(grid)}; "
        f"grid repeats={len(seeds)}; reference configs={len(reference_grid)}; reference repeats={len(reference_seeds)}; "
        f"cv folds={cv_folds}; planned jobs={len(jobs)}"
    )
    if args.dry_run:
        print("Dry run only. Manifest was written; no training started.")
        return

    first_config = config_from_rank_row(args, rank_configs.iloc[0])
    first_config.role_mode = "static_only"
    static_features, skipped, static_cache_path = build_feature_table(first_config, refresh=args.refresh_features)
    print(f"Loaded static features: files={len(static_features)}, subjects={static_features['subject'].nunique()}, cache={static_cache_path}")
    needs_all_roles = any(normalize_dynamic_data_mode(row.get("dynamic_data_mode", "static_only")) == "train_all_roles" for row in grid)
    all_role_features = static_features
    if needs_all_roles:
        all_config = RunConfig(**asdict(first_config))
        all_config.role_mode = "all_roles"
        all_role_features, all_skipped, all_cache_path = build_feature_table(all_config, refresh=args.refresh_features)
        print(
            f"Loaded all-role features: files={len(all_role_features)}, "
            f"subjects={all_role_features['subject'].nunique()}, cache={all_cache_path}"
        )

    run_rows: List[Dict[str, object]] = []
    report_items: List[Tuple[Dict[str, object], Dict[str, object]]] = []
    runs_csv = out_dir / "overfitting_runs.csv"
    summary_csv = out_dir / "overfitting_summary.csv"

    for job_idx, (rank_row, rank, grid_row, repeat, seed) in enumerate(jobs, start=1):
        if bool(grid_row.get("is_reference", False)):
            config = original_config_from_rank_row(args, rank_row, grid_row)
        else:
            config = config_from_rank_row(args, rank_row)
            config = apply_overfit_config(config, grid_row, args)
        per_config_report_dir = report_dir / str(grid_row["overfit_config_id"])
        per_config_curve_dir = learning_curve_dir / str(grid_row["overfit_config_id"])
        per_config_report_dir.mkdir(parents=True, exist_ok=True)
        per_config_curve_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[{job_idx}/{len(jobs)}] rank {rank} {grid_row['overfit_config_id']} repeat {repeat} seed {seed}",
            flush=True,
        )
        report, report_path = train_eval_groupkfold_once(
            static_features=static_features,
            all_role_features=all_role_features,
            config=config,
            rank_row=rank_row,
            rank=rank,
            repeat=repeat,
            seed=seed,
            cv_folds=cv_folds,
            dynamic_data_mode=str(grid_row.get("dynamic_data_mode", "static_only")),
            refresh_cnn_windows=args.refresh_cnn_windows,
            report_dir=per_config_report_dir,
            learning_curve_dir=per_config_curve_dir,
        )
        run_rows.append(overfit_run_row(report, rank_row, report_path, grid_row))
        report_items.append((report, grid_row))
        pd.DataFrame(run_rows).to_csv(runs_csv, index=False)
        build_overfit_summary(run_rows).to_csv(summary_csv, index=False)
        write_overfit_confusion_aggregates(report_items, out_dir)
        print(
            f"  cv subject BA={report['subject_balanced_accuracy']:.3f}, "
            f"macro F1={report['subject_macro_f1']:.3f}, best_epoch={report['folds'][0]['best_epoch']}",
            flush=True,
        )

    print(f"Overfitting runs CSV: {runs_csv}")
    print(f"Overfitting summary CSV: {summary_csv}")
    print(f"Confusion matrices: {out_dir / 'confusion_matrices'}")


if __name__ == "__main__":
    main()
