"""Leakage-safe feature-vector baseline models / 防泄漏特征向量基线模型。"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class FeatureVectorBaseline:
    """Allow-listed sklearn baseline with fold-local preprocessing.

    带有 fold-local 预处理的白名单 sklearn 基线。特征名称必须在构造时冻结，预测
    阶段列数不匹配会直接失败。
    """

    _MODEL_IDS = {"logistic_regression", "rbf_svm", "extra_trees"}

    def __init__(
        self,
        model_id: str,
        feature_names: list[str] | tuple[str, ...],
        *,
        seed: int = 42,
        class_weight: str | dict[int, float] | None = None,
    ) -> None:
        if model_id not in self._MODEL_IDS:
            raise ValueError(f"unsupported feature baseline: {model_id}")
        if not feature_names or len(feature_names) != len(set(feature_names)):
            raise ValueError("feature_names must be non-empty and unique")
        self.model_id = model_id
        self.feature_names = tuple(feature_names)
        self.seed = int(seed)
        self.class_weight = class_weight
        self.pipeline = self._make_pipeline()
        self.fitted_participant_ids_: tuple[str, ...] = ()
        self.fitted_object_provenance_: dict[str, dict[str, object]] = {}

    def _make_pipeline(self) -> Pipeline:
        """Construct an unfitted allow-listed pipeline / 构造未拟合白名单管线。"""

        if self.model_id == "logistic_regression":
            estimator = LogisticRegression(
                max_iter=5_000, class_weight=self.class_weight, random_state=self.seed
            )
            return Pipeline(
                [
                    # Preserve the frozen registry width even when an outer-train
                    # fold has an all-missing route-specific field.
                    # 即使某 outer-train 折的路线特征全缺失，也保持冻结列宽。
                    ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                    ("scaler", StandardScaler()),
                    ("model", estimator),
                ]
            )
        if self.model_id == "rbf_svm":
            estimator = SVC(
                kernel="rbf", probability=True, class_weight=self.class_weight, random_state=self.seed
            )
            return Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                    ("scaler", StandardScaler()),
                    ("model", estimator),
                ]
            )
        estimator = ExtraTreesClassifier(
            n_estimators=500,
            class_weight=self.class_weight,
            random_state=self.seed,
            n_jobs=1,
        )
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("model", estimator),
            ]
        )

    def _validate_x(self, x: np.ndarray) -> np.ndarray:
        """Validate the frozen feature width / 校验冻结的特征宽度。"""

        array = np.asarray(x, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != len(self.feature_names):
            raise ValueError("feature vector width differs from frozen feature_names")
        return array

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        participant_ids: list[str] | tuple[str, ...],
        sample_weight: np.ndarray | None = None,
    ) -> "FeatureVectorBaseline":
        """Fit only on explicitly supplied training rows / 仅拟合显式训练行。"""

        array = self._validate_x(x)
        labels = np.asarray(y)
        if labels.shape != (array.shape[0],) or len(participant_ids) != array.shape[0]:
            raise ValueError("labels and participant_ids must align with feature rows")
        fit_arguments: dict[str, np.ndarray] = {}
        if sample_weight is not None:
            weights = np.asarray(sample_weight, dtype=np.float64)
            if weights.shape != (array.shape[0],) or not np.isfinite(weights).all() or np.any(weights < 0):
                raise ValueError("sample_weight must be finite, non-negative and row-aligned")
            fit_arguments["model__sample_weight"] = weights
        self.pipeline.fit(array, labels, **fit_arguments)
        self.fitted_participant_ids_ = tuple(sorted(set(str(value) for value in participant_ids)))
        # English: Every stateful pipeline step records the identical outer-fold
        # membership, making imputer/scaler provenance independently auditable.
        # 中文：每个有状态步骤记录相同的 outer-fold 成员，使 imputer/scaler 的
        # provenance 可以分别审计。
        self.fitted_object_provenance_ = {
            name: {
                "object_type": type(step).__name__,
                "fitted_participant_ids": self.fitted_participant_ids_,
            }
            for name, step in self.pipeline.named_steps.items()
        }
        return self

    @property
    def classes_(self) -> np.ndarray:
        """Return learned class order / 返回已学习类别顺序。"""

        return np.asarray(self.pipeline.named_steps["model"].classes_)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return class probabilities / 返回类别概率。"""

        return np.asarray(self.pipeline.predict_proba(self._validate_x(x)), dtype=np.float64)
