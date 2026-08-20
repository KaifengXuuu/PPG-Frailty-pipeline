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
        logistic_c: float = 1.0,
        logistic_max_iter: int = 5000,
        logistic_solver: str = "lbfgs",
        svm_kernel: str = "rbf",
        svm_probability: bool = True,
        svm_c: float = 1.0,
        svm_gamma: str | float = "scale",
        extra_trees_n_estimators: int = 500,
        extra_trees_n_jobs: int = 1,
        extra_trees_max_features: str | int | float | None = "sqrt",
        extra_trees_min_samples_leaf: int | float = 1,
    ) -> None:
        if model_id not in self._MODEL_IDS:
            raise ValueError(f"unsupported feature baseline: {model_id}")
        if not feature_names or len(feature_names) != len(set(feature_names)):
            raise ValueError("feature_names must be non-empty and unique")
        if class_weight is not None:
            raise ValueError(
                "model.class_weight is not an independent weighting capability; "
                "configure the single training.class_weighting strategy"
            )
        self.model_id = model_id
        self.feature_names = tuple(feature_names)
        self.seed = int(seed)
        self.class_weight = None
        self.logistic_c = logistic_c
        self.logistic_max_iter = logistic_max_iter
        self.logistic_solver = logistic_solver
        self.svm_kernel = svm_kernel
        self.svm_probability = svm_probability
        self.svm_c = svm_c
        self.svm_gamma = svm_gamma
        self.extra_trees_n_estimators = extra_trees_n_estimators
        self.extra_trees_n_jobs = extra_trees_n_jobs
        self.extra_trees_max_features = extra_trees_max_features
        self.extra_trees_min_samples_leaf = extra_trees_min_samples_leaf
        if model_id == "logistic_regression":
            if (
                isinstance(logistic_c, (bool, np.bool_))
                or not isinstance(logistic_c, (int, float, np.integer, np.floating))
                or not np.isfinite(float(logistic_c))
                or float(logistic_c) <= 0.0
            ):
                raise ValueError("logistic_c must be finite and positive")
            if (
                isinstance(logistic_max_iter, (bool, np.bool_))
                or not isinstance(logistic_max_iter, (int, np.integer))
                or int(logistic_max_iter) <= 0
            ):
                raise ValueError("logistic_max_iter must be a positive integer")
            supported_solvers = {
                "lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga",
            }
            if str(logistic_solver) not in supported_solvers:
                raise ValueError(
                    f"logistic_solver must be one of {sorted(supported_solvers)}"
                )
            self.logistic_c = float(logistic_c)
            self.logistic_max_iter = int(logistic_max_iter)
            self.logistic_solver = str(logistic_solver)
        if model_id == "rbf_svm" and (
            svm_kernel != "rbf" or svm_probability is not True or float(svm_c) <= 0.0
        ):
            raise ValueError("rbf_svm requires kernel=rbf, probability=true and positive C")
        if model_id == "extra_trees":
            if isinstance(extra_trees_n_estimators, (bool, np.bool_)) or not isinstance(
                extra_trees_n_estimators, (int, np.integer)
            ) or int(extra_trees_n_estimators) <= 0:
                raise ValueError("extra_trees_n_estimators must be a positive integer")
            if isinstance(extra_trees_n_jobs, (bool, np.bool_)) or not isinstance(
                extra_trees_n_jobs, (int, np.integer)
            ) or int(extra_trees_n_jobs) == 0:
                raise ValueError("extra_trees_n_jobs must be a non-zero integer")
            if extra_trees_max_features is None:
                normalized_max_features: str | int | float | None = None
            elif isinstance(extra_trees_max_features, str):
                if extra_trees_max_features not in {"sqrt", "log2"}:
                    raise ValueError("extra_trees_max_features string must be 'sqrt' or 'log2'")
                normalized_max_features = extra_trees_max_features
            elif isinstance(extra_trees_max_features, (bool, np.bool_)):
                raise ValueError("extra_trees_max_features cannot be boolean")
            elif isinstance(extra_trees_max_features, (int, np.integer)):
                if int(extra_trees_max_features) <= 0:
                    raise ValueError("integer extra_trees_max_features must be positive")
                normalized_max_features = int(extra_trees_max_features)
            elif isinstance(extra_trees_max_features, (float, np.floating)):
                if (
                    not np.isfinite(float(extra_trees_max_features))
                    or not 0.0 < float(extra_trees_max_features) <= 1.0
                ):
                    raise ValueError("float extra_trees_max_features must be in (0,1]")
                normalized_max_features = float(extra_trees_max_features)
            else:
                raise ValueError("unsupported extra_trees_max_features value")
            if isinstance(extra_trees_min_samples_leaf, (bool, np.bool_)):
                raise ValueError("extra_trees_min_samples_leaf cannot be boolean")
            if isinstance(extra_trees_min_samples_leaf, (int, np.integer)):
                if int(extra_trees_min_samples_leaf) <= 0:
                    raise ValueError("integer extra_trees_min_samples_leaf must be positive")
                normalized_min_samples_leaf: int | float = int(
                    extra_trees_min_samples_leaf
                )
            elif isinstance(extra_trees_min_samples_leaf, (float, np.floating)):
                if (
                    not np.isfinite(float(extra_trees_min_samples_leaf))
                    or not 0.0 < float(extra_trees_min_samples_leaf) <= 0.5
                ):
                    raise ValueError("float extra_trees_min_samples_leaf must be in (0,0.5]")
                normalized_min_samples_leaf = float(extra_trees_min_samples_leaf)
            else:
                raise ValueError("unsupported extra_trees_min_samples_leaf value")
            self.extra_trees_n_estimators = int(extra_trees_n_estimators)
            self.extra_trees_n_jobs = int(extra_trees_n_jobs)
            self.extra_trees_max_features = normalized_max_features
            self.extra_trees_min_samples_leaf = normalized_min_samples_leaf
        self.pipeline = self._make_pipeline()
        self.fitted_participant_ids_: tuple[str, ...] = ()
        self.fitted_object_provenance_: dict[str, dict[str, object]] = {}

    def _make_pipeline(self) -> Pipeline:
        """Construct an unfitted allow-listed pipeline / 构造未拟合白名单管线。"""

        if self.model_id == "logistic_regression":
            estimator = LogisticRegression(
                C=float(self.logistic_c),
                max_iter=int(self.logistic_max_iter),
                solver=str(self.logistic_solver),
                class_weight=self.class_weight,
                random_state=self.seed,
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
                kernel=str(self.svm_kernel),
                probability=bool(self.svm_probability),
                C=float(self.svm_c),
                gamma=self.svm_gamma,
                class_weight=self.class_weight,
                random_state=self.seed,
            )
            return Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                    ("scaler", StandardScaler()),
                    ("model", estimator),
                ]
            )
        estimator = ExtraTreesClassifier(
            n_estimators=int(self.extra_trees_n_estimators),
            max_features=self.extra_trees_max_features,
            min_samples_leaf=self.extra_trees_min_samples_leaf,
            class_weight=self.class_weight,
            random_state=self.seed,
            n_jobs=int(self.extra_trees_n_jobs),
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
