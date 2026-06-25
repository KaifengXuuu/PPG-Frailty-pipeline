from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from frailty_3class_classifier import (
    CLASS_NAMES,
    InceptionBlock,
    RunConfig,
    aggregate_by_key,
    build_cnn_window_table,
    build_feature_table,
    finite_float,
    print_dataset_summary,
    set_all_seeds,
)


META_COLS = {"path", "dataset", "subject", "role", "class_name", "label"}
FUSION_MODEL_CHOICES = ("cnn1d_fusion", "inception_time_fusion")


class FusionWindowDataset(Dataset):
    def __init__(self, x: np.ndarray, features: np.ndarray, y: np.ndarray) -> None:
        self.x = np.asarray(x, dtype=np.float32)
        self.features = np.asarray(features, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)

    def __len__(self) -> int:
        return int(len(self.y))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.x[idx]),
            torch.from_numpy(self.features[idx]),
            torch.tensor(self.y[idx], dtype=torch.long),
        )


class CnnFeatureFusionClassifier(nn.Module):
    def __init__(self, n_signal_channels: int, n_hand_features: int, n_classes: int) -> None:
        super().__init__()
        self.signal_encoder = nn.Sequential(
            nn.Conv1d(n_signal_channels, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(0.10),
            nn.Conv1d(32, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(0.15),
            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.feature_encoder = nn.Sequential(
            nn.Linear(n_hand_features, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, n_classes),
        )

    def forward(self, signal_x: torch.Tensor, hand_features: torch.Tensor) -> torch.Tensor:
        signal_repr = self.signal_encoder(signal_x)
        feature_repr = self.feature_encoder(hand_features)
        return self.classifier(torch.cat([signal_repr, feature_repr], dim=1))


class InceptionTimeFeatureFusionClassifier(nn.Module):
    def __init__(self, n_signal_channels: int, n_hand_features: int, n_classes: int) -> None:
        super().__init__()
        self.signal_encoder = InceptionBlock(
            in_channels=n_signal_channels,
            depth=6,
            n_filters=32,
            bottleneck_channels=32,
        )
        self.signal_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.feature_encoder = nn.Sequential(
            nn.Linear(n_hand_features, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.signal_encoder.out_channels + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, n_classes),
        )

    def forward(self, signal_x: torch.Tensor, hand_features: torch.Tensor) -> torch.Tensor:
        signal_repr = self.signal_pool(self.signal_encoder(signal_x))
        feature_repr = self.feature_encoder(hand_features)
        return self.classifier(torch.cat([signal_repr, feature_repr], dim=1))


def make_fusion_model(model_name: str, n_signal_channels: int, n_hand_features: int, n_classes: int) -> nn.Module:
    if model_name == "cnn1d_fusion":
        return CnnFeatureFusionClassifier(
            n_signal_channels=n_signal_channels,
            n_hand_features=n_hand_features,
            n_classes=n_classes,
        )
    if model_name == "inception_time_fusion":
        return InceptionTimeFeatureFusionClassifier(
            n_signal_channels=n_signal_channels,
            n_hand_features=n_hand_features,
            n_classes=n_classes,
        )
    raise ValueError(f"Unknown fusion model: {model_name}")


def device_for_run() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def feature_columns(features) -> List[str]:
    return [col for col in features.columns if col not in META_COLS]


def fit_feature_transform(
    file_features: np.ndarray,
    train_file_idx: np.ndarray,
) -> Tuple[SimpleImputer, StandardScaler, np.ndarray]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_imputed = imputer.fit_transform(file_features[train_file_idx])
    scaler.fit(train_imputed)
    all_scaled = scaler.transform(imputer.transform(file_features)).astype(np.float32)
    return imputer, scaler, all_scaled


def predict_proba(
    model: nn.Module,
    x: np.ndarray,
    hand_features: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    probs: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start : start + batch_size].astype(np.float32)).to(device)
            fb = torch.from_numpy(hand_features[start : start + batch_size].astype(np.float32)).to(device)
            probs.append(torch.softmax(model(xb, fb), dim=1).detach().cpu().numpy())
    return np.concatenate(probs, axis=0) if probs else np.empty((0, len(CLASS_NAMES)), dtype=np.float32)


def train_fusion_model(
    x_train: np.ndarray,
    feature_train: np.ndarray,
    y_train: np.ndarray,
    config: RunConfig,
    seed: int,
    model_name: str,
    x_val: Optional[np.ndarray] = None,
    feature_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
) -> Tuple[nn.Module, Dict[str, object]]:
    set_all_seeds(seed)
    device = device_for_run()
    model = make_fusion_model(
        model_name=model_name,
        n_signal_channels=x_train.shape[1],
        n_hand_features=feature_train.shape[1],
        n_classes=len(CLASS_NAMES),
    ).to(device)
    counts = np.bincount(y_train, minlength=len(CLASS_NAMES)).astype(np.float32)
    class_weights = counts.sum() / (len(CLASS_NAMES) * np.maximum(counts, 1.0))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.cnn_lr, weight_decay=1e-4)
    loader = DataLoader(
        FusionWindowDataset(x_train, feature_train, y_train),
        batch_size=config.cnn_batch_size,
        shuffle=True,
        num_workers=config.cnn_num_workers,
    )

    best_state = {key: val.detach().cpu().clone() for key, val in model.state_dict().items()}
    best_score = -1.0
    best_epoch = 0
    stale_epochs = 0
    history: List[Dict[str, float]] = []
    for epoch in range(1, config.cnn_epochs + 1):
        model.train()
        losses: List[float] = []
        for xb, fb, yb in loader:
            xb = xb.to(device)
            fb = fb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb, fb), yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        mean_loss = float(np.mean(losses)) if losses else 0.0
        if x_val is not None and feature_val is not None and y_val is not None and len(y_val):
            val_probs = predict_proba(model, x_val, feature_val, config.cnn_batch_size, device)
            val_pred = np.argmax(val_probs, axis=1)
            val_score = finite_float(balanced_accuracy_score(y_val, val_pred))
        else:
            val_score = -mean_loss
        history.append({"epoch": float(epoch), "train_loss": mean_loss, "val_balanced_accuracy": val_score})

        if val_score > best_score + 1e-6:
            best_score = val_score
            best_epoch = epoch
            stale_epochs = 0
            best_state = {key: val.detach().cpu().clone() for key, val in model.state_dict().items()}
        else:
            stale_epochs += 1
            if x_val is not None and config.cnn_patience > 0 and stale_epochs >= config.cnn_patience:
                break

    model.load_state_dict({key: val.to(device) for key, val in best_state.items()})
    return model, {"best_epoch": int(best_epoch), "best_window_balanced_accuracy": finite_float(best_score), "history": history}


def evaluate_fusion(features, config: RunConfig, model_name: str, refresh_cnn: bool) -> Dict[str, object]:
    features = features.reset_index(drop=True)
    cols = feature_columns(features)
    raw_file_features = features[cols].to_numpy(dtype=np.float32)
    x_win, y_win, subject_win, file_win, cnn_cache_path = build_cnn_window_table(features, config, refresh=refresh_cnn)
    y_file = features["label"].to_numpy(dtype=int)
    groups = features["subject"].to_numpy()

    subject_labels = features.groupby("subject")["label"].first()
    n_splits = max(2, min(config.folds, int(subject_labels.value_counts().min())))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=config.seed)
    device = device_for_run()

    window_true: List[int] = []
    window_pred: List[int] = []
    file_true: List[int] = []
    file_pred: List[int] = []
    subject_true: List[int] = []
    subject_pred: List[int] = []
    fold_summaries: List[Dict[str, object]] = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(np.zeros(len(features)), y_file, groups), start=1):
        _, _, scaled_file_features = fit_feature_transform(raw_file_features, train_idx)
        window_features = scaled_file_features[file_win]
        train_mask = np.isin(file_win, train_idx)
        test_mask = np.isin(file_win, test_idx)

        model, fold_info = train_fusion_model(
            x_win[train_mask],
            window_features[train_mask],
            y_win[train_mask],
            config,
            seed=config.seed + fold,
            model_name=model_name,
            x_val=x_win[test_mask],
            feature_val=window_features[test_mask],
            y_val=y_win[test_mask],
        )
        probs = predict_proba(model, x_win[test_mask], window_features[test_mask], config.cnn_batch_size, device)
        preds = np.argmax(probs, axis=1)
        y_test = y_win[test_mask]
        window_true.extend(y_test.tolist())
        window_pred.extend(preds.tolist())

        fold_file_true, fold_file_pred = aggregate_by_key(probs, y_test, file_win[test_mask].tolist())
        fold_subject_true, fold_subject_pred = aggregate_by_key(probs, y_test, subject_win[test_mask].tolist())
        file_true.extend(fold_file_true)
        file_pred.extend(fold_file_pred)
        subject_true.extend(fold_subject_true)
        subject_pred.extend(fold_subject_pred)

        fold_summaries.append(
            {
                "fold": int(fold),
                "n_train_files": int(len(train_idx)),
                "n_test_files": int(len(test_idx)),
                "n_train_windows": int(np.sum(train_mask)),
                "n_test_windows": int(np.sum(test_mask)),
                "test_subjects": sorted(map(str, np.unique(groups[test_idx]))),
                "best_epoch": fold_info["best_epoch"],
                "best_window_balanced_accuracy": fold_info["best_window_balanced_accuracy"],
                "file_balanced_accuracy": finite_float(balanced_accuracy_score(fold_file_true, fold_file_pred)),
                "subject_balanced_accuracy": finite_float(balanced_accuracy_score(fold_subject_true, fold_subject_pred)),
            }
        )

    return {
        "model": model_name,
        "n_files": int(len(features)),
        "n_subjects": int(features["subject"].nunique()),
        "n_windows": int(len(y_win)),
        "n_hand_features": int(len(cols)),
        "n_splits": int(n_splits),
        "cnn_cache": str(cnn_cache_path),
        "window_balanced_accuracy": finite_float(balanced_accuracy_score(window_true, window_pred)),
        "window_macro_f1": finite_float(f1_score(window_true, window_pred, average="macro")),
        "file_balanced_accuracy": finite_float(balanced_accuracy_score(file_true, file_pred)),
        "file_macro_f1": finite_float(f1_score(file_true, file_pred, average="macro")),
        "subject_balanced_accuracy": finite_float(balanced_accuracy_score(subject_true, subject_pred)),
        "subject_macro_f1": finite_float(f1_score(subject_true, subject_pred, average="macro")),
        "window_confusion_matrix": confusion_matrix(window_true, window_pred, labels=[0, 1, 2]).tolist(),
        "file_confusion_matrix": confusion_matrix(file_true, file_pred, labels=[0, 1, 2]).tolist(),
        "subject_confusion_matrix": confusion_matrix(subject_true, subject_pred, labels=[0, 1, 2]).tolist(),
        "subject_classification_report": classification_report(
            subject_true,
            subject_pred,
            labels=[0, 1, 2],
            target_names=list(CLASS_NAMES),
            zero_division=0,
            output_dict=True,
        ),
        "folds": fold_summaries,
        "feature_columns": cols,
    }


def save_final_model(features, config: RunConfig, report: Dict[str, object], model_name: str, refresh_cnn: bool) -> Path:
    features = features.reset_index(drop=True)
    cols = feature_columns(features)
    raw_file_features = features[cols].to_numpy(dtype=np.float32)
    x_win, y_win, _, file_win, cnn_cache_path = build_cnn_window_table(features, config, refresh=refresh_cnn)
    all_file_idx = np.arange(len(features), dtype=np.int64)
    imputer, scaler, scaled_file_features = fit_feature_transform(raw_file_features, all_file_idx)
    window_features = scaled_file_features[file_win]
    model, train_info = train_fusion_model(
        x_win,
        window_features,
        y_win,
        config,
        seed=config.seed,
        model_name=model_name,
    )

    out_dir = Path("models")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"frailty3_{model_name}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": model_name,
            "class_names": CLASS_NAMES,
            "config": asdict(config),
            "cv_report": report,
            "train_info": train_info,
            "feature_columns": cols,
            "cnn_cache": str(cnn_cache_path),
            "n_signal_channels": int(x_win.shape[1]),
            "n_hand_features": int(len(cols)),
        },
        out_path,
    )
    joblib.dump(
        {"imputer": imputer, "scaler": scaler, "feature_columns": cols},
        out_dir / f"frailty3_{model_name}_feature_scaler.joblib",
    )
    return out_path


def write_report(report: Dict[str, object], config: RunConfig, feature_cache_path: Path, skipped: Dict[str, List[str]]) -> Path:
    out_dir = Path("results_frailty3")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{report['model']}_report.json"
    payload = {
        "config": asdict(config),
        "feature_cache": str(feature_cache_path),
        "class_names": CLASS_NAMES,
        "skipped": skipped,
        **report,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate a raw-signal + handcrafted-feature fusion frailty classifier.")
    parser.add_argument("--data-root", default="PPG_Testing_05_01_2026")
    parser.add_argument("--model", choices=FUSION_MODEL_CHOICES, default="cnn1d_fusion")
    parser.add_argument("--fs", type=float, default=400.0)
    parser.add_argument("--win-sec", type=float, default=10.0)
    parser.add_argument("--hop-sec", type=float, default=5.0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cnn-target-fs", type=float, default=400.0, help="Deprecated compatibility option; raw windows now stay at --fs.")
    parser.add_argument("--cnn-seq-sec", type=float, default=30.0)
    parser.add_argument("--cnn-hop-sec", type=float, default=30.0)
    parser.add_argument("--cnn-max-windows-per-file", type=int, default=6)
    parser.add_argument("--cnn-epochs", type=int, default=8)
    parser.add_argument("--cnn-batch-size", type=int, default=32)
    parser.add_argument("--cnn-lr", type=float, default=1e-3)
    parser.add_argument("--cnn-patience", type=int, default=3)
    parser.add_argument("--cnn-num-workers", type=int, default=0)
    parser.add_argument("--refresh-features", action="store_true")
    parser.add_argument("--refresh-cnn-windows", action="store_true")
    parser.add_argument("--no-save-model", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RunConfig(
        data_root=args.data_root,
        fs=args.fs,
        win_sec=args.win_sec,
        hop_sec=args.hop_sec,
        folds=args.folds,
        seed=args.seed,
        cnn_target_fs=args.cnn_target_fs,
        cnn_seq_sec=args.cnn_seq_sec,
        cnn_hop_sec=args.cnn_hop_sec,
        cnn_max_windows_per_file=args.cnn_max_windows_per_file,
        cnn_epochs=args.cnn_epochs,
        cnn_batch_size=args.cnn_batch_size,
        cnn_lr=args.cnn_lr,
        cnn_patience=args.cnn_patience,
        cnn_num_workers=args.cnn_num_workers,
    )
    features, skipped, feature_cache_path = build_feature_table(config, refresh=args.refresh_features)
    print_dataset_summary(features, skipped)
    report = evaluate_fusion(features, config, model_name=args.model, refresh_cnn=args.refresh_cnn_windows)
    report_path = write_report(report, config, feature_cache_path, skipped)

    print("\nCross-validation summary")
    print(f"model: {args.model}")
    print(f"window balanced accuracy: {report['window_balanced_accuracy']:.3f}")
    print(f"window macro F1: {report['window_macro_f1']:.3f}")
    print(f"file balanced accuracy: {report['file_balanced_accuracy']:.3f}")
    print(f"file macro F1: {report['file_macro_f1']:.3f}")
    print(f"subject balanced accuracy: {report['subject_balanced_accuracy']:.3f}")
    print(f"subject macro F1: {report['subject_macro_f1']:.3f}")
    print("subject confusion matrix rows=true cols=pred, order=pre_frail, robust_non_frail, young")
    print(np.array(report["subject_confusion_matrix"]))
    print(f"report: {report_path}")

    if not args.no_save_model:
        model_path = save_final_model(features, config, report, model_name=args.model, refresh_cnn=False)
        print(f"saved model: {model_path}")


if __name__ == "__main__":
    main()
