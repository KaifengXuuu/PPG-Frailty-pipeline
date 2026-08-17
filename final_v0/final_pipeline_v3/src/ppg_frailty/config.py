"""V2 严格配置与决策档案合同 / Strict V2 config and decision profiles.

正式运行只接受 ``ppg_frailty.pipeline_config.v2``。复制进 V2 目录的 V1 YAML
只能通过显式 ``allow_legacy=True`` 读取作 provenance，不得进入正式 runner。

Formal execution accepts only ``ppg_frailty.pipeline_config.v2``. Copied V1 YAML
can be read only with explicit ``allow_legacy=True`` for provenance and can never
enter the formal runner by accident.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


TOP_LEVEL_KEYS = {
    "schema_version",
    "config_id",
    "manifest",
    "splits",
    "output",
    "representation_mode",
    "roles",
    "signal",
    "windows",
    "quality",
    "artifact",
    "features",
    "model",
    "training",
    "aggregation",
    "evaluation",
}

V2_SCHEMA_VERSION = "ppg_frailty.pipeline_config.v2"
LEGACY_SCHEMA_VERSION = "ppg_frailty.pipeline_config.v1"
V2_DECISION_PROFILE_SCHEMA = "ppg_frailty.v2_decision_profile.v1"
V2_DEPENDENCY_PROFILE_SCHEMA = "ppg_frailty.v2_dependency_profiles.v1"
V2_LOCK_MANIFEST_SCHEMA = "ppg_frailty.v2_dependency_lock_manifest.v1"
V2_FORMAL_CATALOG_SCHEMA = "ppg_frailty.formal_experiment_catalog.v2"
V2_FORMAL_ABLATION_PROFILES_SCHEMA = "ppg_frailty.formal_ablation_profiles.v2"
V2_SPLIT_SEEDS = (42, 10042, 20042, 30042, 40042)
V2_DEPENDENCY_PROFILE_IDS = frozenset(
    {
        "core",
        "deep",
        "formal_benchmark",
        "onnx_winner_gate",
        "prv_aura_compare",
        "prv_rhenan_legacy_compare",
    }
)


def _strict_mapping(value: Any, name: str) -> dict[str, Any]:
    """验证对象类型 / Require a string-keyed mapping."""

    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a string-keyed mapping")
    return dict(value)


def _require_exact_keys(mapping: Mapping[str, Any], required: set[str], *, context: str) -> None:
    """拒绝缺字段和未知字段 / Reject missing and unknown fields."""

    observed = set(mapping)
    missing = sorted(required - observed)
    unknown = sorted(observed - required)
    if missing or unknown:
        raise ValueError(f"{context} key mismatch: missing={missing}, unknown={unknown}")


def canonical_json_bytes(value: Any) -> bytes:
    """稳定严格 JSON / Render canonical strict JSON bytes."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True)
class PipelineConfig:
    """规范实验配置 / Canonical experiment configuration."""

    payload: dict[str, Any]
    source_path: str
    sha256: str

    @property
    def config_id(self) -> str:
        """返回配置 ID / Return the configuration identity."""

        return str(self.payload["config_id"])

    @property
    def representation_mode(self) -> str:
        """返回表征模式 / Return the representation mode."""

        return str(self.payload["representation_mode"])

    @property
    def schema_version(self) -> str:
        """返回配置 schema / Return the explicit schema identity."""

        return str(self.payload["schema_version"])

    @property
    def is_legacy(self) -> bool:
        """V1 仅可作来源快照 / Whether this is a provenance-only V1 config."""

        return self.schema_version == LEGACY_SCHEMA_VERSION

    def section(self, name: str) -> dict[str, Any]:
        """读取一个显式 section / Return one explicit section."""

        if name not in TOP_LEVEL_KEYS:
            raise KeyError(name)
        return _strict_mapping(self.payload[name], name)

    def to_dict(self) -> dict[str, Any]:
        """复制可序列化配置 / Copy the serializable payload."""

        return json.loads(json.dumps(self.payload, allow_nan=False))


def _validate_common_payload(data: dict[str, Any]) -> None:
    """验证 V1/V2 共同结构 / Validate structure shared by V1 and V2."""

    _require_exact_keys(data, TOP_LEVEL_KEYS, context="config")
    if data["representation_mode"] not in {"raw", "feature_vector", "feature_matrix", "fusion"}:
        raise ValueError("unsupported representation_mode")
    roles = data["roles"]
    allowed_roles = {"B", "R1", "R2", "R3", "R4", "S1", "S2", "W1", "W2"}
    if not isinstance(roles, list) or not roles or not all(role in allowed_roles for role in roles):
        raise ValueError("roles must be a non-empty registered role list")
    for section in TOP_LEVEL_KEYS - {"schema_version", "config_id", "representation_mode", "roles"}:
        _strict_mapping(data[section], section)
    training = _strict_mapping(data["training"], "training")
    if training.get("epoch_rule") not in {"fixed_epoch", "inner_grouped_selection"}:
        raise ValueError("training.epoch_rule must be explicit")
    if training.get("outer_labels_visible_to_trainer") is not False:
        raise ValueError("outer labels must be unavailable to the trainer")
    artifact = _strict_mapping(data["artifact"], "artifact")
    if artifact.get("selection_scope") != "run_before_evaluation":
        raise ValueError("artifact route must be selected before evaluation")


def _validate_v2_quality(data: Mapping[str, Any]) -> None:
    """冻结 SQI 三态并关闭未经监督的 route / Freeze the three SQI modes."""

    quality = _strict_mapping(data["quality"], "quality")
    mode = quality.get("mode")
    if mode not in {"off", "diagnostics_only", "route"}:
        raise ValueError("quality.mode must be off, diagnostics_only, or route")
    ready = quality.get("supervised_route_ready")
    if not isinstance(ready, bool):
        raise ValueError("quality.supervised_route_ready must be boolean")
    if mode == "route":
        raise ValueError(
            "quality route is disabled until a supervised authority artifact and hash "
            "are registered in code; a YAML boolean cannot authorize it"
        )
    if ready:
        raise ValueError("supervised_route_ready is status-only and must remain false")
    artifact = _strict_mapping(data["artifact"], "artifact")
    if mode == "off" and list(data["roles"]) != ["B", "R1", "R2", "R3", "R4"]:
        raise ValueError("quality off formal inputs must be exactly B,R1,R2,R3,R4")
    if mode == "off" and artifact.get("motion_detector_enabled") is not False:
        raise ValueError("quality off requires motion_detector_enabled=false")


def _validate_v2_balance(data: Mapping[str, Any]) -> None:
    """强制训练与聚合使用同一条 A/B 线 / Keep train and aggregation matched."""

    training = _strict_mapping(data["training"], "training")
    aggregation = _strict_mapping(data["aggregation"], "aggregation")
    pair = (training.get("training_balance"), aggregation.get("balance_line"))
    allowed = {
        ("equal_files", "line_a_equal_files"): ["window", "file", "participant"],
        ("equal_role_families", "line_b_equal_role_families"): [
            "window",
            "file",
            "role_family",
            "participant",
        ],
    }
    if pair not in allowed:
        raise ValueError("training_balance and aggregation.balance_line must be matched A-A or B-B")
    if aggregation.get("hierarchy") != allowed[pair]:
        raise ValueError("aggregation hierarchy does not match the selected balance line")


def _validate_v2_protocol(data: Mapping[str, Any]) -> None:
    """验证 5x5、epoch 与统计合同 / Validate the frozen V2 protocol."""

    splits = _strict_mapping(data["splits"], "splits")
    if (
        splits.get("n_splits") != 5
        or splits.get("n_repeats") != 5
        or tuple(splits.get("split_seeds", ())) != V2_SPLIT_SEEDS
        or splits.get("runtime_recompute") is not False
    ):
        raise ValueError("V2 formal configs require the frozen 5x5 participant registry")
    training = _strict_mapping(data["training"], "training")
    if training.get("execution_mode") != "formal":
        raise ValueError("formal V2 YAML must declare training.execution_mode=formal")
    if training.get("epoch_rule") != "fixed_epoch":
        raise ValueError("V2 reference/epoch ablations use fixed_epoch")
    epoch_profiles = {"ablation_7": 7, "default_10": 10, "ablation_15": 15}
    epoch_profile = training.get("epoch_profile")
    if epoch_profile not in epoch_profiles or training.get("fixed_epochs") != epoch_profiles[epoch_profile]:
        raise ValueError("V2 epoch_profile and fixed_epochs must be a matched 7/10/15 pair")
    if training.get("sampler") != "balance_line_weighted_v2":
        raise ValueError("V2 formal configs require sampler=balance_line_weighted_v2")
    if training.get("label_smoothing") != 0.0:
        raise ValueError("V2 reference requires explicit label_smoothing=0.0")
    gradient_clip_norm = training.get("gradient_clip_norm")
    if gradient_clip_norm is not None and (
        isinstance(gradient_clip_norm, bool) or float(gradient_clip_norm) <= 0.0
    ):
        raise ValueError("gradient_clip_norm must be null or a positive value")
    evaluation = _strict_mapping(data["evaluation"], "evaluation")
    statistics = _strict_mapping(evaluation.get("statistics"), "evaluation.statistics")
    expected_statistics = {
        "cluster_unit": "participant_with_all_five_repeat_oof_predictions",
        "bootstrap_replicates": 10000,
        "confidence_interval": "two_sided_95_percentile",
        "lcb95_percentile": 2.5,
        "lcb95_metrics": [
            "participant_level_mean_balanced_accuracy",
            "participant_level_mean_macro_f1",
        ],
        "paired_permutation_replicates": 100000,
        "paired_exchange_unit": "participant",
        "multiplicity_correction": "holm_within_comparison_family",
        "affects_automatic_selection": False,
    }
    if statistics != expected_statistics:
        raise ValueError("evaluation.statistics must match the confirmed V2 contract")
    ranking = _strict_mapping(evaluation.get("ranking"), "evaluation.ranking")
    expected_ranking = {
        "sort_key": "participant_level_mean_balanced_accuracy",
        "max_qualified_per_comparison_group": 10,
        "automatic_final_selection": False,
        "manual_multiple_final_versions_allowed": True,
        "preserve_ablation_provenance": True,
    }
    if ranking != expected_ranking:
        raise ValueError("evaluation.ranking must match the confirmed V2 contract")
    if evaluation.get("independent_test_available") is not False:
        raise ValueError("the 29-participant cohort is OOF validation, not an independent test")
    from .module_registry import validate_model_config

    validate_model_config(
        _strict_mapping(data["model"], "model"),
        str(data["representation_mode"]),
    )
    _validate_formal_ablation_materialization(data)


def _validate_formal_ablation_materialization(data: Mapping[str, Any]) -> None:
    """Validate one-factor provenance and reject hidden/cartesian profiles."""

    from .models import normalize_model_id
    from .models.time_scale import fixed_kernel_case
    from .signal.resample import validate_dl_resampling_config

    output = _strict_mapping(data["output"], "output")
    identity = output.get("formal_ablation_materialization")
    signal = _strict_mapping(data["signal"], "signal")
    training = _strict_mapping(data["training"], "training")
    model = _strict_mapping(data["model"], "model")
    dl = validate_dl_resampling_config(signal["dl_resampling"])
    filter_pair = (
        float(signal["ppg_filter"]["low_hz"]),
        float(signal["ppg_filter"]["high_hz"]),
    )
    gravity = str(signal["imu"]["gravity_method"])
    nonreference = {
        "epoch": (
            training["epoch_profile"], int(training["fixed_epochs"])
        ) != ("default_10", 10),
        "filter": filter_pair != (0.2, 8.0),
        "gravity": gravity != "quaternion_error_state_ekf",
        "fixed_kernel": (
            dl.get("case_id") is not None
            and not str(dl.get("case_id")).endswith("__reference")
        ),
    }
    if identity is None:
        if any(nonreference.values()):
            raise ValueError("non-reference profile requires materialization provenance")
        return
    identity = _strict_mapping(identity, "formal_ablation_materialization")
    _require_exact_keys(
        identity,
        {
            "schema_version", "family", "profile_id", "catalog_role",
            "base_config_path", "base_config_sha256", "profile_catalog_sha256",
            "single_factor_only", "automatic_execution",
            "scientific_execution_completed",
        },
        context="formal_ablation_materialization",
    )
    if (
        identity["schema_version"]
        != "ppg_frailty.formal_ablation_materialization.v2"
        or identity["family"] not in {
            "deep_fixed_epoch", "direct_filter", "imu_gravity",
            "fixed_kernel_samples",
        }
        or identity["single_factor_only"] is not True
        or identity["automatic_execution"] is not False
        or identity["scientific_execution_completed"] is not False
    ):
        raise ValueError("formal ablation materialization contract invalid")
    for name in ("base_config_sha256", "profile_catalog_sha256"):
        digest = str(identity[name])
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"formal ablation {name} must be SHA-256")
    family = str(identity["family"])
    active = {name for name, enabled in nonreference.items() if enabled}
    expected_active = {
        "deep_fixed_epoch": {"epoch"},
        "direct_filter": {"filter"},
        "imu_gravity": {"gravity"},
        "fixed_kernel_samples": {"fixed_kernel"},
    }[family]
    if str(identity["catalog_role"]) == "reference":
        expected_active = set()
    if active != expected_active:
        raise ValueError("formal ablation config is not a single-factor profile")
    profile_id = str(identity["profile_id"])
    if family == "deep_fixed_epoch":
        expected = {
            "epoch_7_ablation": ("ablation_7", 7),
            "default_epoch_10": ("default_10", 10),
            "epoch_15_ablation": ("ablation_15", 15),
        }.get(profile_id)
        _canonical, machine = normalize_model_id(str(model["model_id"]))
        if expected is None or machine in {
            "logistic_regression", "rbf_svm", "extra_trees",
            "rocket_numpy", "minirocket_ablation",
        } or (training["epoch_profile"], int(training["fixed_epochs"])) != expected:
            raise ValueError("deep epoch materialization identity drift")
    elif family == "direct_filter":
        expected = {
            "direct_filter_0p2_to_8hz": (0.2, 8.0),
            "direct_filter_0p5_to_5hz_ablation": (0.5, 5.0),
        }.get(profile_id)
        if expected is None or filter_pair != expected:
            raise ValueError("direct filter materialization identity drift")
    elif family == "imu_gravity":
        expected = {
            "calibrated_roll_pitch_ekf": "quaternion_error_state_ekf",
            "imu_lpf_0p3hz_ablation": "low_pass_0p3hz",
        }.get(profile_id)
        if expected is None or gravity != expected:
            raise ValueError("IMU gravity materialization identity drift")
    else:
        case = fixed_kernel_case(profile_id)
        _canonical, machine = normalize_model_id(str(model["model_id"]))
        expected_machine = (
            "compact_cnn" if case.model_name == "CompactCNN1D" else "inception_full"
        )
        if (
            data["representation_mode"] != "raw"
            or machine != expected_machine
            or dl.get("case_id") != case.case_id
            or float(data["windows"]["raw_dl"]["length_s"])
            != float(case.raw_window_seconds)
        ):
            raise ValueError("fixed-kernel materialization identity drift")


def validate_config_payload(payload: Mapping[str, Any], *, allow_legacy: bool = False) -> dict[str, Any]:
    """执行 fail-closed 配置验证 / Validate a formal V2 or explicit legacy config."""

    data = _strict_mapping(payload, "config")
    _validate_common_payload(data)
    schema = data["schema_version"]
    if schema == LEGACY_SCHEMA_VERSION:
        if not allow_legacy:
            raise ValueError("legacy V1 config is provenance-only; pass allow_legacy=True explicitly")
        aggregation = _strict_mapping(data["aggregation"], "aggregation")
        if aggregation.get("hierarchy") != ["window", "file", "role", "participant"]:
            raise ValueError("legacy V1 aggregation hierarchy drift")
        return data
    if schema != V2_SCHEMA_VERSION:
        raise ValueError("unsupported schema_version")
    if not str(data["config_id"]).endswith("_v2"):
        raise ValueError("formal V2 config_id must end with _v2")
    _validate_v2_quality(data)
    _validate_v2_balance(data)
    _validate_v2_protocol(data)
    return data


def load_config(path: str | Path, *, allow_legacy: bool = False) -> PipelineConfig:
    """加载正式 V2 或显式 legacy V1 / Load formal V2 or explicit legacy V1."""

    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    data = validate_config_payload(_strict_mapping(payload, "config"), allow_legacy=allow_legacy)
    digest = hashlib.sha256(canonical_json_bytes(data)).hexdigest()
    return PipelineConfig(data, source.as_posix(), digest)


def load_formal_experiment_catalog(path: str | Path) -> dict[str, Any]:
    """Load the declarative 13-candidate plus two-ensemble V2 catalogue."""

    source = Path(path)
    payload = _strict_mapping(
        yaml.safe_load(source.read_text(encoding="utf-8")),
        "formal_experiment_catalog",
    )
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "catalog_id",
            "pipeline_generation",
            "execution_policy",
            "entries",
        },
        context="formal_experiment_catalog",
    )
    if payload["schema_version"] != V2_FORMAL_CATALOG_SCHEMA:
        raise ValueError("unsupported formal experiment catalog schema")
    if payload["pipeline_generation"] != "final_pipeline_v2":
        raise ValueError("formal catalog must be bound to final_pipeline_v2")
    policy = _strict_mapping(payload["execution_policy"], "execution_policy")
    expected_policy = {
        "auto_run": False,
        "candidate_count": 13,
        "ensemble_comparison_count": 2,
        "default_balance_line": "line_a",
        "selectable_balance_lines": ["line_a", "line_b"],
        "materialization_only": True,
    }
    if policy != expected_policy:
        raise ValueError("formal catalog execution policy drifted")
    raw_entries = payload["entries"]
    if not isinstance(raw_entries, list) or len(raw_entries) != 15:
        raise ValueError("formal catalog requires 13 candidates plus two ensembles")
    from .module_registry import validate_model_config

    entries: list[dict[str, Any]] = []
    entry_ids: set[str] = set()
    stems: set[str] = set()
    for raw in raw_entries:
        entry = _strict_mapping(raw, "catalog_entry")
        _require_exact_keys(
            entry,
            {
                "entry_id",
                "config_stem",
                "representation_mode",
                "catalog_role",
                "model",
            },
            context="catalog_entry",
        )
        entry_id = str(entry["entry_id"])
        stem = str(entry["config_stem"])
        if (
            not entry_id
            or not stem
            or entry_id in entry_ids
            or stem in stems
        ):
            raise ValueError("catalog entry IDs/config stems must be non-empty and unique")
        entry_ids.add(entry_id)
        stems.add(stem)
        if entry["catalog_role"] not in {
            "reference_candidate",
            "ablation_candidate",
            "ensemble_comparison",
        }:
            raise ValueError("invalid formal catalog role")
        validate_model_config(
            _strict_mapping(entry["model"], f"{entry_id}.model"),
            str(entry["representation_mode"]),
        )
        entries.append(entry)
    ensemble_count = sum(
        entry["catalog_role"] == "ensemble_comparison" for entry in entries
    )
    if len(entries) - ensemble_count != 13 or ensemble_count != 2:
        raise ValueError("formal catalogue count contract drifted")
    payload["entries"] = entries
    payload["catalog_sha256"] = hashlib.sha256(
        canonical_json_bytes(
            {key: value for key, value in payload.items() if key != "catalog_sha256"}
        )
    ).hexdigest()
    return payload


def load_formal_ablation_profiles(path: str | Path) -> dict[str, Any]:
    """Load the materialization-only single-factor V2 profile catalogue."""

    source = Path(path)
    payload = _strict_mapping(
        yaml.safe_load(source.read_text(encoding="utf-8")),
        "formal_ablation_profiles",
    )
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "catalog_id",
            "pipeline_generation",
            "execution_policy",
            "families",
        },
        context="formal_ablation_profiles",
    )
    if payload["schema_version"] != V2_FORMAL_ABLATION_PROFILES_SCHEMA:
        raise ValueError("unsupported formal ablation-profile schema")
    if payload["pipeline_generation"] != "final_pipeline_v2":
        raise ValueError("ablation profiles must be bound to final_pipeline_v2")
    if payload["execution_policy"] != {
        "auto_run": False,
        "materialization_only": True,
        "allow_cartesian_product": False,
        "safe_suite_execution": False,
    }:
        raise ValueError("formal ablation execution policy drifted")
    families = _strict_mapping(payload["families"], "ablation_profile_families")
    _require_exact_keys(
        families,
        {"deep_fixed_epoch", "direct_filter", "imu_gravity", "fixed_kernel_samples"},
        context="ablation_profile_families",
    )
    expected_epoch = {
        "reference_profile_id": "default_epoch_10",
        "entries": [
            {"profile_id": "epoch_7_ablation", "fixed_epochs": 7, "catalog_role": "ablation", "auto_run": False},
            {"profile_id": "default_epoch_10", "fixed_epochs": 10, "catalog_role": "reference", "auto_run": False},
            {"profile_id": "epoch_15_ablation", "fixed_epochs": 15, "catalog_role": "ablation", "auto_run": False},
        ],
    }
    expected_filter = {
        "reference_profile_id": "direct_filter_0p2_to_8hz",
        "entries": [
            {"profile_id": "direct_filter_0p2_to_8hz", "low_hz": 0.2, "high_hz": 8.0, "catalog_role": "reference", "auto_run": False},
            {"profile_id": "direct_filter_0p5_to_5hz_ablation", "low_hz": 0.5, "high_hz": 5.0, "catalog_role": "ablation", "auto_run": False},
        ],
    }
    expected_imu = {
        "reference_profile_id": "calibrated_roll_pitch_ekf",
        "silent_fallback_forbidden": True,
        "entries": [
            {"profile_id": "calibrated_roll_pitch_ekf", "method": "calibrated_roll_pitch_ekf", "catalog_role": "reference", "auto_run": False},
            {"profile_id": "imu_lpf_0p3hz_ablation", "method": "low_pass_0p3hz", "catalog_role": "ablation", "auto_run": False},
        ],
    }
    if families["deep_fixed_epoch"] != expected_epoch:
        raise ValueError("fixed epoch profiles drifted")
    if families["direct_filter"] != expected_filter:
        raise ValueError("direct filter profiles drifted")
    if families["imu_gravity"] != expected_imu:
        raise ValueError("IMU gravity profiles drifted")
    from .models.time_scale import build_fixed_kernel_resampling_cases

    expected_cases = [
        {
            "case_id": case.case_id,
            "model_name": case.model_name,
            "dl_fs_hz": case.dl_fs_hz,
            "raw_window_seconds": case.raw_window_seconds,
            "sequence_length_samples": case.sequence_length_samples,
            "kernel_samples": list(case.kernel_samples),
            "dilation": case.dilation,
            "catalog_role": (
                "reference" if case.case_id.endswith("__reference") else "ablation"
            ),
            "auto_run": False,
        }
        for case in build_fixed_kernel_resampling_cases()
    ]
    expected_fixed = {
        "family_id": "fixed_kernel_samples_resampling_ablation",
        "eligible_models": ["CompactCNN1D", "InceptionTimeFull"],
        "physical_time_matched_claim": "forbidden",
        "cases": expected_cases,
    }
    if families["fixed_kernel_samples"] != expected_fixed:
        raise ValueError("fixed-kernel 12-case profiles drifted")
    payload["catalog_sha256"] = hashlib.sha256(
        canonical_json_bytes(
            {key: value for key, value in payload.items() if key != "catalog_sha256"}
        )
    ).hexdigest()
    return payload


def materialize_formal_ablation_config(
    base_config_path: str | Path,
    *,
    family: str,
    profile_id: str,
    output_path: str | Path,
    profiles_path: str | Path,
) -> PipelineConfig:
    """Materialize exactly one registered comparison factor; never execute it."""

    base_path = Path(base_config_path).resolve()
    target = Path(output_path).resolve()
    pipeline_root = Path(profiles_path).resolve().parent.parent
    base_relative = base_path.relative_to(pipeline_root).as_posix()
    target.relative_to(pipeline_root)
    if target.exists():
        raise FileExistsError(f"ablation config overwrite forbidden: {target}")
    base = load_config(base_path)
    catalog = load_formal_ablation_profiles(profiles_path)
    if family not in {
        "deep_fixed_epoch", "direct_filter", "imu_gravity",
        "fixed_kernel_samples",
    }:
        raise ValueError("unknown formal ablation family")
    payload = base.to_dict()
    from .models import normalize_model_id

    _canonical, machine_id = normalize_model_id(str(payload["model"]["model_id"]))
    estimator_ids = {
        "logistic_regression", "rbf_svm", "extra_trees",
        "rocket_numpy", "minirocket_ablation",
    }
    selected: dict[str, Any]
    if family == "fixed_kernel_samples":
        from .models.time_scale import fixed_kernel_case

        case = fixed_kernel_case(profile_id)
        expected_machine = (
            "compact_cnn" if case.model_name == "CompactCNN1D" else "inception_full"
        )
        if payload["representation_mode"] != "raw" or machine_id != expected_machine:
            raise ValueError(
                "fixed-kernel case requires the matching raw CompactCNN/Inception config"
            )
        payload["windows"]["raw_dl"]["length_s"] = float(case.raw_window_seconds)
        resampling = payload["signal"]["dl_resampling"]
        resampling["case_id"] = case.case_id
        resampling["enabled"] = float(case.dl_fs_hz) != 400.0
        resampling["target_fs_hz"] = float(case.dl_fs_hz)
        if machine_id == "compact_cnn":
            dilations = [int(case.dilation)] * 3
            payload["model"]["dilations"] = dilations
            payload["model"]["architecture_parameters"]["dilations"] = dilations
        else:
            payload["model"]["dilation"] = int(case.dilation)
            payload["model"]["architecture_parameters"]["dilation"] = int(case.dilation)
        selected = {
            "profile_id": case.case_id,
            "catalog_role": (
                "reference" if case.case_id.endswith("__reference") else "ablation"
            ),
        }
    else:
        entries = catalog["families"][family]["entries"]
        matches = [dict(row) for row in entries if row["profile_id"] == profile_id]
        if len(matches) != 1:
            raise ValueError(f"unknown profile_id for {family}: {profile_id}")
        selected = matches[0]
        if selected.get("auto_run") is not False:
            raise ValueError("formal ablation profiles must never auto-run")
        if family == "deep_fixed_epoch":
            if machine_id in estimator_ids:
                raise ValueError("epoch profiles are deep-model-only")
            fixed = int(selected["fixed_epochs"])
            payload["training"]["fixed_epochs"] = fixed
            payload["training"]["epoch_profile"] = {
                7: "ablation_7", 10: "default_10", 15: "ablation_15"
            }[fixed]
        elif family == "direct_filter":
            low = float(selected["low_hz"])
            high = float(selected["high_hz"])
            payload["signal"]["ppg_filter"]["low_hz"] = low
            payload["signal"]["ppg_filter"]["high_hz"] = high
            payload["signal"]["analysis_view"]["direct_source"] = (
                f"x_filter_{format(low, 'g').replace('.', 'p')}_to_"
                f"{format(high, 'g').replace('.', 'p')}hz"
            )
        else:
            payload["signal"]["imu"]["gravity_method"] = (
                "low_pass_0p3hz"
                if selected["profile_id"] == "imu_lpf_0p3hz_ablation"
                else "quaternion_error_state_ekf"
            )

    payload["config_id"] = (
        base.config_id.removesuffix("_v2")
        + "__" + str(profile_id).replace("-", "_") + "_v2"
    )
    payload["output"]["formal_ablation_materialization"] = {
        "schema_version": "ppg_frailty.formal_ablation_materialization.v2",
        "family": family,
        "profile_id": str(profile_id),
        "catalog_role": str(selected["catalog_role"]),
        "base_config_path": base_relative,
        "base_config_sha256": base.sha256,
        "profile_catalog_sha256": catalog["catalog_sha256"],
        "single_factor_only": True,
        "automatic_execution": False,
        "scientific_execution_completed": False,
    }
    validated = validate_config_payload(payload)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        temporary.write_text(
            yaml.safe_dump(validated, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return load_config(target)


def load_v2_decision_profile(path: str | Path) -> dict[str, Any]:
    """读取机器可审计默认/延期门 / Load machine-auditable defaults and gates."""

    source = Path(path)
    data = _strict_mapping(yaml.safe_load(source.read_text(encoding="utf-8")), "decision_profile")
    required = {
        "schema_version",
        "pipeline_generation",
        "profile_id",
        "authority",
        "confirmed_defaults",
        "comparison_profiles",
        "deferred_gates",
    }
    _require_exact_keys(data, required, context="decision_profile")
    if data["schema_version"] != V2_DECISION_PROFILE_SCHEMA:
        raise ValueError("unsupported V2 decision profile schema")
    if data["pipeline_generation"] != "final_pipeline_v2":
        raise ValueError("decision profile is not bound to final_pipeline_v2")
    for key in ("authority", "confirmed_defaults", "comparison_profiles", "deferred_gates"):
        _strict_mapping(data[key], key)
    return data


def load_dependency_profiles(
    profiles_path: str | Path,
    lock_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """验证六个分层依赖及未伪造 lock 状态 / Validate six dependency profiles."""

    profiles = _strict_mapping(
        json.loads(Path(profiles_path).read_text(encoding="utf-8")), "dependency_profiles"
    )
    locks = _strict_mapping(
        json.loads(Path(lock_path).read_text(encoding="utf-8")), "dependency_locks"
    )
    if profiles.get("schema_version") != V2_DEPENDENCY_PROFILE_SCHEMA:
        raise ValueError("unsupported dependency profile schema")
    if locks.get("schema_version") != V2_LOCK_MANIFEST_SCHEMA:
        raise ValueError("unsupported dependency lock schema")
    profile_rows = profiles.get("profiles")
    lock_rows = locks.get("profiles")
    if not isinstance(profile_rows, list) or not isinstance(lock_rows, list):
        raise ValueError("dependency profiles and locks must be lists")
    observed = {str(row.get("profile_id")) for row in profile_rows if isinstance(row, Mapping)}
    locked = {str(row.get("profile_id")) for row in lock_rows if isinstance(row, Mapping)}
    if observed != V2_DEPENDENCY_PROFILE_IDS or locked != V2_DEPENDENCY_PROFILE_IDS:
        raise ValueError("dependency manifests must contain exactly the six V2 profiles")
    for row in profile_rows:
        item = _strict_mapping(row, "dependency_profile")
        requirement_path = Path(profiles_path).parent / str(item.get("requirements_file"))
        if not requirement_path.is_file():
            raise ValueError(f"missing requirements profile: {requirement_path}")
    supplemental_profiles = profiles.get("supplemental_optional_inputs", [])
    supplemental_locks = locks.get("supplemental_optional_inputs", [])
    if not isinstance(supplemental_profiles, list) or not isinstance(supplemental_locks, list):
        raise ValueError("supplemental dependency inputs must be lists")
    if {row.get("input_id") for row in supplemental_profiles} != {"artifact_legacy_ablation"}:
        raise ValueError("the only supplemental input must be artifact_legacy_ablation")
    if {row.get("input_id") for row in supplemental_locks} != {"artifact_legacy_ablation"}:
        raise ValueError("supplemental lock identity mismatch")
    for row in supplemental_profiles:
        item = _strict_mapping(row, "supplemental_dependency_input")
        requirement_path = Path(profiles_path).parent / str(item.get("requirements_file"))
        if not requirement_path.is_file() or item.get("blocks_formal_benchmark") is not False:
            raise ValueError("legacy artifact input must exist and never block formal benchmark")
    allowed_lock_statuses = {
        "pending_profile_install_and_full_regression",
        "installed_safe_smoke_verified_not_full_regression",
        "compatibility_smoke_failed_upstream_nolds_import",
        "validated_exact_lock",
    }
    for row in [*lock_rows, *supplemental_locks]:
        item = _strict_mapping(row, "dependency_lock")
        status = item.get("status")
        resolved = item.get("resolved_packages")
        if status not in allowed_lock_statuses:
            raise ValueError("unknown dependency lock status")
        if status == "pending_profile_install_and_full_regression" and resolved:
            raise ValueError("pending lock must not claim resolved package versions")
        if status != "pending_profile_install_and_full_regression" and not resolved:
            raise ValueError("resolved or attempted profile must record exact packages")
    declared_by_id = {
        str(row["profile_id"]): _strict_mapping(row, "dependency_profile")
        for row in profile_rows
    }
    pipeline_root = Path(profiles_path).resolve().parent.parent
    inventory_cache: dict[Path, dict[str, Any]] = {}
    for raw in lock_rows:
        item = _strict_mapping(raw, "dependency_lock")
        if item.get("status") != "validated_exact_lock":
            continue
        profile_id = str(item["profile_id"])
        requirement_path = (
            Path(profiles_path).resolve().parent
            / str(declared_by_id[profile_id]["requirements_file"])
        )
        if (
            item.get("requirements_sha256")
            != hashlib.sha256(requirement_path.read_bytes()).hexdigest()
        ):
            raise ValueError(f"exact lock requirements hash drift: {profile_id}")
        raw_inventory_path = item.get("environment_inventory_path")
        expected_inventory_sha = item.get("environment_inventory_sha256")
        if not isinstance(raw_inventory_path, str) or not isinstance(
            expected_inventory_sha, str
        ):
            raise ValueError(f"exact lock inventory binding missing: {profile_id}")
        inventory_path = (pipeline_root / raw_inventory_path).resolve()
        inventory_path.relative_to(pipeline_root)
        if (
            not inventory_path.is_file()
            or hashlib.sha256(inventory_path.read_bytes()).hexdigest()
            != expected_inventory_sha
        ):
            raise ValueError(f"exact lock inventory hash drift: {profile_id}")
        if inventory_path not in inventory_cache:
            inventory = _strict_mapping(
                json.loads(inventory_path.read_text(encoding="utf-8")),
                "dependency_environment_inventory",
            )
            encoded = json.dumps(inventory, sort_keys=True, allow_nan=False)
            if (
                inventory.get("schema_version")
                != "ppg_frailty.environment_inventory.v2"
                or inventory.get("pipeline_generation") != "final_pipeline_v2"
                or inventory.get("environment_mutated_by_capture") is not False
                or inventory.get("validation", {}).get("pip_check")
                != "passed_no_broken_requirements"
                or inventory.get("validation", {}).get(
                    "scientific_training_executed"
                )
                is not False
                or "anaconda3/envs/ml" in encoded
                or "\\Users\\" in encoded
            ):
                raise ValueError("exact lock environment inventory contract drift")
            inventory_cache[inventory_path] = inventory
        inventory = inventory_cache[inventory_path]
        exact_distribution_set = inventory.get("exact_installed_distribution_set")
        if profile_id in {"onnx_winner_gate", "prv_aura_compare"}:
            exact_distribution_set = _strict_mapping(
                exact_distribution_set,
                "exact_installed_distribution_set",
            )
            _require_exact_keys(
                exact_distribution_set,
                {
                    "policy", "records_path", "record_count",
                    "records_sha256", "profile_closure_count",
                },
                context="exact_installed_distribution_set",
            )
            records_path = (
                pipeline_root / str(exact_distribution_set["records_path"])
            ).resolve()
            records_path.relative_to(pipeline_root)
            if not records_path.is_file():
                raise ValueError(
                    f"{profile_id} exact installed-distribution record file missing"
                )
            records_bytes = records_path.read_bytes()
            records = records_bytes.decode("utf-8").splitlines()
            if (
                exact_distribution_set["policy"]
                != "exact_records_no_unknown_distributions"
                or not isinstance(records, list)
                or records != sorted(records)
                or len(records) != len(set(records))
                or int(exact_distribution_set["record_count"]) != len(records)
                or hashlib.sha256(records_bytes).hexdigest()
                != exact_distribution_set["records_sha256"]
                or int(exact_distribution_set["profile_closure_count"])
                != len(item["resolved_packages"])
                or (
                    profile_id == "onnx_winner_gate"
                    and (
                        "onnxscript==0.5.7" not in item["resolved_packages"]
                        or "onnx-ir==0.1.13" not in item["resolved_packages"]
                    )
                )
                or (
                    profile_id == "prv_aura_compare"
                    and not {
                        "hrv-analysis==1.0.2", "nolds==0.6.2",
                        "astropy==5.2.2", "numpy==1.26.4",
                    } <= set(item["resolved_packages"])
                )
            ):
                raise ValueError(
                    f"{profile_id} exact installed-distribution policy drift"
                )
        bound_artifacts = item.get("bound_artifacts", [])
        if not isinstance(bound_artifacts, list):
            raise ValueError(f"exact lock bound_artifacts must be a list: {profile_id}")
        observed_bound_paths: set[str] = set()
        for raw_artifact in bound_artifacts:
            artifact = _strict_mapping(raw_artifact, "exact_lock_bound_artifact")
            _require_exact_keys(
                artifact,
                {"path", "sha256", "bytes"},
                context="exact_lock_bound_artifact",
            )
            artifact_relative = str(artifact["path"])
            if artifact_relative in observed_bound_paths:
                raise ValueError(f"duplicate exact lock bound artifact: {profile_id}")
            artifact_path = (pipeline_root / artifact_relative).resolve()
            artifact_path.relative_to(pipeline_root)
            if (
                not artifact_path.is_file()
                or artifact_path.stat().st_size != int(artifact["bytes"])
                or hashlib.sha256(artifact_path.read_bytes()).hexdigest()
                != str(artifact["sha256"])
            ):
                raise ValueError(
                    f"exact lock bound artifact drift: {profile_id}:{artifact_relative}"
                )
            observed_bound_paths.add(artifact_relative)
        if profile_id == "onnx_winner_gate" and observed_bound_paths != {
            "locks/onnx_winner_gate_isolated_probe_v2.json",
            "locks/onnx_winner_gate_installed_distributions_v2.txt",
            "locks/onnx_winner_gate_tiny_smoke_v2.json",
            "src/ppg_frailty/onnx_winner.py",
            "tools/validate_onnx_winner_profile.py",
        }:
            raise ValueError("ONNX exact lock source/smoke artifact binding drift")
        if profile_id == "prv_aura_compare" and observed_bound_paths != {
            "locks/prv_aura_hrv102_fixed_ppi_smoke_v2.json",
            "locks/prv_aura_hrv102_installed_distributions_v2.txt",
            "src/ppg_frailty/features/prv_backend_compare.py",
            "tools/validate_prv_aura_profile.py",
        }:
            raise ValueError("Aura exact lock source/smoke artifact binding drift")
        closures = _strict_mapping(
            inventory.get("profile_transitive_closures"),
            "profile_transitive_closures",
        )
        if list(closures.get(profile_id, ())) != list(item["resolved_packages"]):
            raise ValueError(f"exact lock transitive closure drift: {profile_id}")
        if (
            item.get("python_version") != inventory.get("python", {}).get("version")
            or item.get("platform") != inventory.get("platform")
        ):
            raise ValueError(f"exact lock runtime identity drift: {profile_id}")
    return profiles, locks


_DEEP_MODEL_IDS = frozenset(
    {
        "CompactCNN1D",
        "InceptionTimeFull",
        "InceptionTimeSmall",
        "InceptionTimeMatrix",
        "InceptionTimeFullFiveMemberEnsemble",
        "InceptionTimeMatrixFiveMemberEnsemble",
        "ShapeFormerChannelSpecificOSD",
        "ShapeFormerChannelSpecificScalarDistanceAblation",
        "ShapeFormerEffectSizeFixedV1",
        "FileBagFusionCompact",
        "FileBagFusionInception",
    }
)


def required_dependency_profile_ids(
    config: PipelineConfig,
    *,
    operation: str,
) -> tuple[str, ...]:
    """Resolve only the dependency profiles needed by one explicit operation."""

    allowed = {
        "preflight",
        "reduced_smoke",
        "formal_benchmark",
        "onnx_winner_gate",
        "prv_aura_compare",
        "prv_rhenan_legacy_compare",
    }
    if operation not in allowed:
        raise ValueError(f"unknown dependency-gate operation: {operation}")
    if operation in {"prv_aura_compare", "prv_rhenan_legacy_compare"}:
        return (operation,)
    if operation == "onnx_winner_gate":
        # The exact isolated post-selection runtime includes the protected base
        # closure plus both reviewed converter routes. Bundle loading separately
        # rechecks its training-runtime environment before deserialisation.
        return ("onnx_winner_gate",)
    required = ["core"]
    if str(config.section("model")["model_id"]) in _DEEP_MODEL_IDS:
        required.append("deep")
    if operation == "formal_benchmark":
        required.append("formal_benchmark")
    return tuple(required)


def _installed_distribution_records() -> tuple[str, ...]:
    """Return exact normalized installed records including duplicate metadata."""

    import importlib.metadata
    import re
    import sys

    prefix = Path(sys.prefix).resolve()
    base_prefix = Path(sys.base_prefix).resolve()
    rows: list[str] = []
    for distribution in importlib.metadata.distributions():
        location = Path(distribution.locate_file("")).resolve()
        try:
            location.relative_to(prefix)
            origin = "isolated_prefix"
        except ValueError:
            try:
                location.relative_to(base_prefix)
                origin = "protected_base_prefix"
            except ValueError:
                origin = "outside_declared_prefixes"
        name = distribution.metadata.get("Name") or distribution.name
        canonical_name = re.sub(r"[-_.]+", "-", str(name)).lower()
        rows.append(
            f"{canonical_name}=={distribution.version}@{origin}"
        )
    return tuple(sorted(rows))


_FORMAL_IMPORT_DISTRIBUTION_SURFACE = {
    "numpy": "numpy",
    "scipy": "scipy",
    "scikit-learn": "sklearn",
    "torch": "torch",
    "joblib": "joblib",
    "pyarrow": "pyarrow",
    "pyyaml": "yaml",
    "pandas": "pandas",
    "astropy": "astropy",
    "nolds": "nolds",
    "hrv-analysis": "hrvanalysis",
    "onnx": "onnx",
    "onnxruntime": "onnxruntime",
    "skl2onnx": "skl2onnx",
    "onnxscript": "onnxscript",
    "onnx-ir": "onnx_ir",
}


def _formal_import_origin_evidence(
    expected_packages: Mapping[str, str],
) -> tuple[bool, list[dict[str, Any]]]:
    """Bind imported modules to exact files owned by locked distributions."""

    import importlib
    import importlib.metadata
    import importlib.machinery
    import importlib.util
    import re

    normalized_expected = {
        re.sub(r"[-_.]+", "-", str(name)).lower(): str(name)
        for name in expected_packages
    }
    rows: list[dict[str, Any]] = []
    for canonical_name, module_name in sorted(
        _FORMAL_IMPORT_DISTRIBUTION_SURFACE.items()
    ):
        distribution_name = normalized_expected.get(canonical_name)
        if distribution_name is None:
            continue
        row: dict[str, Any] = {
            "distribution": distribution_name,
            "module": module_name,
            "module_origin": None,
            "distribution_record_owned": False,
            "status": "unverified",
        }
        try:
            distribution = importlib.metadata.distribution(distribution_name)
            files = distribution.files
            if files is None:
                raise ValueError("distribution_has_no_record_files")
            distribution_root = Path(
                distribution.locate_file("")
            ).resolve(strict=True)
            owned_files = {
                Path(distribution.locate_file(item)).resolve(strict=False)
                for item in files
            }
            trusted_spec = importlib.machinery.PathFinder.find_spec(
                module_name,
                [str(distribution_root)],
            )
            observed_spec = importlib.util.find_spec(module_name)
            if (
                trusted_spec is None
                or trusted_spec.origin is None
                or observed_spec is None
                or observed_spec.origin is None
            ):
                raise ValueError("module_has_no_regular_import_spec")
            trusted_lexical = Path(trusted_spec.origin).absolute()
            observed_lexical = Path(observed_spec.origin).absolute()
            trusted_origin = trusted_lexical.resolve(strict=True)
            observed_origin = observed_lexical.resolve(strict=True)
            if (
                trusted_origin != observed_origin
                or trusted_lexical != trusted_origin
                or observed_lexical != observed_origin
                or trusted_origin not in owned_files
                or not trusted_origin.is_file()
                or trusted_origin.is_symlink()
            ):
                raise ValueError("import_spec_origin_not_distribution_owned")
            module = importlib.import_module(module_name)
            module_origin = Path(
                str(getattr(module, "__file__", ""))
            ).absolute().resolve(strict=True)
            module_spec_origin = Path(
                str(getattr(getattr(module, "__spec__", None), "origin", ""))
            ).absolute().resolve(strict=True)
            owned = (
                module_origin == trusted_origin
                and module_spec_origin == trusted_origin
            )
            row.update(
                {
                    "module_origin": trusted_origin.as_posix(),
                    "distribution_record_owned": bool(owned),
                    "status": "matched" if owned else "origin_not_owned",
                }
            )
        except Exception as exc:
            row["status"] = (
                f"error:{type(exc).__name__}:{str(exc)[:200]}"
            )
        rows.append(row)
    return bool(rows) and all(
        row["distribution_record_owned"] is True for row in rows
    ), rows


def _live_exact_lock_evidence(
    lock: Mapping[str, Any],
    *,
    pipeline_root: Path,
    run_pip_check: bool,
) -> dict[str, Any]:
    """Compare one static exact lock with the interpreter executing the gate."""

    import importlib.metadata
    import platform
    import subprocess
    import sys

    inventory_path = (
        pipeline_root / str(lock["environment_inventory_path"])
    ).resolve()
    inventory_path.relative_to(pipeline_root)
    inventory = _strict_mapping(
        json.loads(inventory_path.read_text(encoding="utf-8")),
        "dependency_environment_inventory",
    )
    expected_packages: dict[str, str] = {}
    observed_packages: dict[str, str | None] = {}
    for frozen in lock.get("resolved_packages", ()):
        name, separator, version = str(frozen).partition("==")
        if not separator or not name or not version or name in expected_packages:
            raise ValueError("exact lock package row must be unique NAME==VERSION")
        expected_packages[name] = version
        try:
            observed_packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            observed_packages[name] = None
    python_inventory = _strict_mapping(
        inventory.get("python"),
        "dependency_environment_inventory.python",
    )
    executable_path_policy = str(
        python_inventory.get(
            "executable_path_policy",
            "resolved_relative_to_prefix",
        )
    )
    if executable_path_policy == "lexical_with_resolved_base_binding":
        try:
            executable_relative = (
                Path(sys.executable).absolute()
                .relative_to(Path(sys.prefix).absolute())
                .as_posix()
            )
            resolved_base_relative = (
                Path(sys.executable).resolve()
                .relative_to(Path(sys.base_prefix).resolve())
                .as_posix()
            )
        except ValueError:
            executable_relative = "outside_runtime_prefix"
            resolved_base_relative = "outside_base_prefix"
    elif executable_path_policy == "resolved_relative_to_prefix":
        try:
            executable_relative = (
                Path(sys.executable).resolve()
                .relative_to(Path(sys.prefix).resolve())
                .as_posix()
            )
        except ValueError:
            executable_relative = "outside_runtime_prefix"
        resolved_base_relative = None
    else:
        raise ValueError(
            "unknown exact-lock executable_path_policy: "
            + executable_path_policy
        )
    runtime_identity = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "environment_prefix_basename": Path(sys.prefix).resolve().name,
        "python_executable_relative_to_prefix": executable_relative,
    }
    if executable_path_policy == "lexical_with_resolved_base_binding":
        runtime_identity.update(
            {
                "base_prefix_basename": Path(sys.base_prefix).resolve().name,
                "python_executable_resolved_relative_to_base_prefix": (
                    resolved_base_relative
                ),
            }
        )
    expected_identity = {
        "python_version": str(lock.get("python_version")),
        "platform": str(lock.get("platform")),
        "environment_prefix_basename": str(
            inventory.get("environment_prefix_basename")
        ),
        "python_executable_relative_to_prefix": str(
            python_inventory.get("executable_relative_to_prefix")
        ),
    }
    if executable_path_policy == "lexical_with_resolved_base_binding":
        expected_identity.update(
            {
                "base_prefix_basename": str(
                    python_inventory.get("base_prefix_basename")
                ),
                "python_executable_resolved_relative_to_base_prefix": str(
                    python_inventory.get(
                        "executable_resolved_relative_to_base_prefix"
                    )
                ),
            }
        )
    package_match = observed_packages == expected_packages
    identity_match = runtime_identity == expected_identity
    import_origins_match, import_origin_evidence = (
        _formal_import_origin_evidence(expected_packages)
    )
    exact_distribution_set = inventory.get("exact_installed_distribution_set")
    distribution_set_match = True
    distribution_evidence: dict[str, Any] = {
        "policy": "required_packages_only",
        "expected_count": None,
        "observed_count": None,
        "expected_sha256": None,
        "observed_sha256": None,
        "missing_records": [],
        "unexpected_records": [],
    }
    if exact_distribution_set is not None:
        exact_distribution_set = _strict_mapping(
            exact_distribution_set,
            "exact_installed_distribution_set",
        )
        records_path = (
            pipeline_root / str(exact_distribution_set["records_path"])
        ).resolve()
        records_path.relative_to(pipeline_root)
        expected_records = tuple(
            records_path.read_text(encoding="utf-8").splitlines()
        )
        observed_records = _installed_distribution_records()
        expected_set = set(expected_records)
        observed_set = set(observed_records)
        distribution_set_match = observed_records == expected_records
        distribution_evidence = {
            "policy": str(exact_distribution_set["policy"]),
            "expected_count": len(expected_records),
            "observed_count": len(observed_records),
            "expected_sha256": hashlib.sha256(
                ("\n".join(expected_records) + "\n").encode("utf-8")
            ).hexdigest(),
            "observed_sha256": hashlib.sha256(
                ("\n".join(observed_records) + "\n").encode("utf-8")
            ).hexdigest(),
            "missing_records": sorted(expected_set - observed_set),
            "unexpected_records": sorted(observed_set - expected_set),
        }
    pip_check: dict[str, Any] = {
        "executed": False,
        "passed": None,
        "returncode": None,
    }
    if run_pip_check and package_match and identity_match:
        clean_environment = dict(__import__("os").environ)
        clean_environment.pop("PYTHONPATH", None)
        clean_environment.pop("PYTHONHOME", None)
        completed = subprocess.run(
            [sys.executable, "-I", "-m", "pip", "check"],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=Path(sys.prefix).resolve(),
            env=clean_environment,
        )
        pip_check = {
            "executed": True,
            "passed": completed.returncode == 0,
            "returncode": int(completed.returncode),
        }
    live_match = (
        package_match
        and identity_match
        and import_origins_match
        and distribution_set_match
        and (not run_pip_check or pip_check["passed"] is True)
    )
    return {
        "status": "matched" if live_match else "mismatch",
        "identity_match": identity_match,
        "package_versions_match": package_match,
        "import_origins_match": import_origins_match,
        "import_origin_evidence": import_origin_evidence,
        "installed_distribution_set_match": distribution_set_match,
        "installed_distribution_evidence": distribution_evidence,
        "expected_identity": expected_identity,
        "observed_identity": runtime_identity,
        "expected_packages": expected_packages,
        "observed_packages": observed_packages,
        "pip_check": pip_check,
        "live_exact_match": live_match,
    }


def dependency_gate_report_for_profiles(
    *,
    config_id: str,
    required_profile_ids: Sequence[str],
    operation: str,
    profiles_path: str | Path,
    lock_path: str | Path,
    require_exact_lock: bool,
) -> dict[str, Any]:
    """Report exact locks for an explicit, config-derived profile roster."""

    if not str(config_id).strip() or not str(operation).strip():
        raise ValueError("dependency gate config_id and operation must be explicit")
    required = tuple(str(value) for value in required_profile_ids)
    if (
        not required
        or len(required) != len(set(required))
        or not set(required).issubset(V2_DEPENDENCY_PROFILE_IDS)
    ):
        raise ValueError("dependency gate profile roster is empty, duplicate, or unknown")

    profiles, locks = load_dependency_profiles(profiles_path, lock_path)
    declared = {str(row["profile_id"]): dict(row) for row in profiles["profiles"]}
    locked = {str(row["profile_id"]): dict(row) for row in locks["profiles"]}
    pipeline_root = Path(profiles_path).resolve().parent.parent
    rows: list[dict[str, Any]] = []
    for profile_id in required:
        profile = declared[profile_id]
        lock = locked[profile_id]
        live = (
            _live_exact_lock_evidence(
                lock,
                pipeline_root=pipeline_root,
                run_pip_check=bool(require_exact_lock),
            )
            if lock.get("status") == "validated_exact_lock"
            else {
                "status": "not_checked_lock_not_exact",
                "live_exact_match": False,
            }
        )
        rows.append(
            {
                "profile_id": profile_id,
                "requirements_file": str(profile["requirements_file"]),
                "lock_status": str(lock["status"]),
                "python_version": lock.get("python_version"),
                "requirements_sha256": lock.get("requirements_sha256"),
                "resolved_packages": list(lock.get("resolved_packages", ())),
                "live_runtime": live,
                "exact_lock_ready": (
                    lock.get("status") == "validated_exact_lock"
                    and live["live_exact_match"] is True
                ),
            }
        )
    ready = all(row["exact_lock_ready"] for row in rows)
    report = {
        "schema_version": "ppg_frailty.dependency_gate.v2",
        "pipeline_generation": "final_pipeline_v2",
        "operation": operation,
        "config_id": str(config_id),
        "required_profile_ids": list(required),
        "require_exact_lock": bool(require_exact_lock),
        "all_required_exact_locks_ready": bool(ready),
        "profiles": rows,
    }
    if require_exact_lock and not ready:
        missing = [row["profile_id"] for row in rows if not row["exact_lock_ready"]]
        raise RuntimeError(
            "formal dependency exact-lock gate is closed for profiles: "
            + ",".join(missing)
        )
    return report


def dependency_gate_report(
    config: PipelineConfig,
    *,
    operation: str,
    profiles_path: str | Path,
    lock_path: str | Path,
    require_exact_lock: bool,
) -> dict[str, Any]:
    """Report a config-derived dependency gate; scientific use fails closed."""

    required = required_dependency_profile_ids(config, operation=operation)
    return dependency_gate_report_for_profiles(
        config_id=config.config_id,
        required_profile_ids=required,
        operation=operation,
        profiles_path=profiles_path,
        lock_path=lock_path,
        require_exact_lock=require_exact_lock,
    )


__all__ = [
    "LEGACY_SCHEMA_VERSION",
    "PipelineConfig",
    "TOP_LEVEL_KEYS",
    "V2_DECISION_PROFILE_SCHEMA",
    "V2_DEPENDENCY_PROFILE_IDS",
    "V2_FORMAL_CATALOG_SCHEMA",
    "V2_FORMAL_ABLATION_PROFILES_SCHEMA",
    "V2_SCHEMA_VERSION",
    "dependency_gate_report",
    "dependency_gate_report_for_profiles",
    "load_config",
    "load_dependency_profiles",
    "load_formal_experiment_catalog",
    "materialize_formal_ablation_config",
    "load_formal_ablation_profiles",
    "load_v2_decision_profile",
    "required_dependency_profile_ids",
    "validate_config_payload",
]
