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
from typing import Any, Mapping

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
V2_FORMAL_CATALOG_SCHEMA = "ppg_frailty.formal_experiment_catalog.v2"
V2_FORMAL_ABLATION_PROFILES_SCHEMA = "ppg_frailty.formal_ablation_profiles.v2"
V2_SPLIT_SEEDS = (42, 10042, 20042, 30042, 40042)
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
            "role",
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
    balance_pair = (
        str(training["training_balance"]),
        str(data["aggregation"]["balance_line"]),
    )
    nonreference = {
        "epoch": (
            training["epoch_profile"], int(training["fixed_epochs"])
        ) != ("default_10", 10),
        "filter": filter_pair != (0.2, 8.0),
        "gravity": gravity != "calibrated_roll_pitch_ekf",
        "aggregation": balance_pair
        != ("equal_role_families", "line_b_equal_role_families"),
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
            "fixed_kernel_samples", "aggregation_balance",
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
        "aggregation_balance": {"aggregation"},
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
            "calibrated_roll_pitch_ekf": "calibrated_roll_pitch_ekf",
            "imu_lpf_0p3hz_ablation": "low_pass_0p3hz",
        }.get(profile_id)
        if expected is None or gravity != expected:
            raise ValueError("IMU gravity materialization identity drift")
    elif family == "aggregation_balance":
        expected = {
            "role_aware_equal_roles": (
                "equal_role_families", "line_b_equal_role_families"
            ),
            "equal_files_line_a_ablation": (
                "equal_files", "line_a_equal_files"
            ),
        }.get(profile_id)
        if expected is None or balance_pair != expected:
            raise ValueError("aggregation-balance materialization identity drift")
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
    config_id = str(data["config_id"]).strip()
    base_config_id, separator, generated_case_id = config_id.partition("__")
    if not (
        config_id.endswith("_v2")
        or (
            separator
            and base_config_id.endswith("_v2")
            and bool(generated_case_id.strip())
        )
    ):
        raise ValueError(
            "V2 config_id must end with _v2 or be <base_v2>__<generated_case>"
        )
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
        "default_balance_line": "line_b",
        "selectable_balance_lines": ["line_b", "line_a"],
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
        {
            "aggregation_balance", "deep_fixed_epoch", "direct_filter",
            "imu_gravity", "fixed_kernel_samples",
        },
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
    expected_aggregation = {
        "reference_profile_id": "role_aware_equal_roles",
        "entries": [
            {
                "profile_id": "role_aware_equal_roles",
                "training_balance": "equal_role_families",
                "balance_line": "line_b_equal_role_families",
                "hierarchy": ["window", "file", "role", "participant"],
                "catalog_role": "reference",
                "auto_run": False,
            },
            {
                "profile_id": "equal_files_line_a_ablation",
                "training_balance": "equal_files",
                "balance_line": "line_a_equal_files",
                "hierarchy": ["window", "file", "participant"],
                "catalog_role": "ablation",
                "auto_run": False,
            },
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
    if families["aggregation_balance"] != expected_aggregation:
        raise ValueError("aggregation-balance profiles drifted")
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
        "fixed_kernel_samples", "aggregation_balance",
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
        elif family == "imu_gravity":
            payload["signal"]["imu"]["gravity_method"] = (
                "low_pass_0p3hz"
                if selected["profile_id"] == "imu_lpf_0p3hz_ablation"
                else "calibrated_roll_pitch_ekf"
            )
        else:
            is_line_b = selected["profile_id"] == "role_aware_equal_roles"
            payload["training"]["training_balance"] = str(
                selected["training_balance"]
            )
            payload["aggregation"].update(
                {
                    "balance_line": str(selected["balance_line"]),
                    "hierarchy": list(selected["hierarchy"]),
                    "window_to_file": "ordinary_mean",
                    "file_to_role": (
                        "ordinary_mean" if is_line_b else "not_applicable"
                    ),
                    "role_to_participant": (
                        "ordinary_mean" if is_line_b else "not_applicable"
                    ),
                    "missing_role_policy": (
                        "mean_available_roles" if is_line_b else "not_applicable"
                    ),
                    "quality_weighting": False,
                    "direct_all_window_participant_mean": False,
                }
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
_BASE_RUNTIME_MODULES = ("numpy", "scipy", "sklearn", "yaml", "pyarrow")


def required_runtime_modules(config: PipelineConfig) -> tuple[str, ...]:
    """Return import names needed for an ordinary run of this configuration."""

    modules = list(_BASE_RUNTIME_MODULES)
    if str(config.section("model")["model_id"]) in _DEEP_MODEL_IDS:
        modules.append("torch")
    return tuple(modules)


def dependency_availability_report(config: PipelineConfig) -> dict[str, Any]:
    """Report missing runtime imports without pinning versions or import origins."""

    import importlib.util

    modules = required_runtime_modules(config)
    rows = [
        {
            "module": module,
            "available": importlib.util.find_spec(module) is not None,
        }
        for module in modules
    ]
    missing = [row["module"] for row in rows if not row["available"]]
    return {
        "schema_version": "ppg_frailty.dependency_availability.v2",
        "pipeline_generation": "final_pipeline_v2",
        "config_id": config.config_id,
        "ready": not missing,
        "missing_modules": missing,
        "modules": rows,
        "policy": "ordinary_import_availability_no_version_or_origin_lock",
    }


def require_runtime_dependencies(config: PipelineConfig) -> dict[str, Any]:
    """Raise one actionable error when ordinary runtime imports are missing."""

    report = dependency_availability_report(config)
    if report["missing_modules"]:
        raise RuntimeError(
            "missing runtime dependencies: "
            + ", ".join(report["missing_modules"])
        )
    return report


__all__ = [
    "LEGACY_SCHEMA_VERSION",
    "PipelineConfig",
    "TOP_LEVEL_KEYS",
    "V2_DECISION_PROFILE_SCHEMA",
    "V2_FORMAL_CATALOG_SCHEMA",
    "V2_FORMAL_ABLATION_PROFILES_SCHEMA",
    "V2_SCHEMA_VERSION",
    "dependency_availability_report",
    "load_config",
    "load_formal_experiment_catalog",
    "materialize_formal_ablation_config",
    "load_formal_ablation_profiles",
    "load_v2_decision_profile",
    "require_runtime_dependencies",
    "required_runtime_modules",
    "validate_config_payload",
]
