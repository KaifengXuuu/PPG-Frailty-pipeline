"""配置、类型和规格锁回归 / Configuration, type, and specification-lock regressions."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import numpy as np

from ppg_frailty.config import TOP_LEVEL_KEYS, load_config, validate_config_payload
from ppg_frailty.contracts import (
    QualityComponent,
    QualityEndpoint,
    QualityResult,
    QualityState,
    SignalRoute,
    SignalViews,
    to_strict_json_value,
)


ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parents[1]


class ConfigContractTests(unittest.TestCase):
    """确认无隐藏默认值和外折隔离 / Check explicit config and outer isolation."""

    def test_all_reference_configs_load_and_have_unique_ids(self) -> None:
        """四份参考配置必须通过唯一加载器 / Load every reference config."""

        configs = [load_config(path) for path in sorted((ROOT / "configs").glob("*.yaml"))]
        self.assertEqual(len(configs), 4)
        self.assertEqual(len({config.config_id for config in configs}), 4)
        self.assertEqual(
            {config.representation_mode for config in configs},
            {"raw", "feature_vector", "feature_matrix"},
        )

    def test_config_has_exact_top_level_keys(self) -> None:
        """未知字段不得变成静默行为 / Unknown keys cannot become silent behavior."""

        config = load_config(ROOT / "configs/reference_static_v1.yaml")
        self.assertEqual(set(config.payload), TOP_LEVEL_KEYS)
        mutated = config.to_dict()
        mutated["hidden_default"] = True
        with self.assertRaisesRegex(ValueError, "unknown"):
            validate_config_payload(mutated)

    def test_outer_labels_are_never_visible(self) -> None:
        """所有参考训练配置关闭 outer labels / Outer labels stay hidden."""

        for path in sorted((ROOT / "configs").glob("*.yaml")):
            config = load_config(path)
            self.assertIs(config.payload["training"]["outer_labels_visible_to_trainer"], False)
            self.assertFalse(config.payload["splits"]["runtime_recompute"])


class SignalAndQualityContractTests(unittest.TestCase):
    """锁定 400 Hz 对齐与 rate-only 语义 / Lock alignment and rate-only semantics."""

    def test_signal_views_accept_aligned_400_hz(self) -> None:
        """合法 direct views 共享采样网格 / Valid direct views share a grid."""

        values = np.zeros((400, 2), dtype=np.float64)
        views = SignalViews(values.copy(), values.copy(), values.copy(), {}, {"fs_hz": 400.0})
        views.validate()

    def test_non_identity_requires_rate_only_flag(self) -> None:
        """非恒等 x_ar 不得伪装形态保真 / Non-identity output must be rate-only."""

        values = np.zeros((400, 2), dtype=np.float64)
        views = SignalViews(
            values.copy(),
            values.copy(),
            values.copy(),
            {},
            {"fs_hz": 400.0, "non_identity_artifact_reduction": True, "rate_only": False},
        )
        with self.assertRaisesRegex(ValueError, "rate-only"):
            views.validate()

    def test_q_morph_not_applicable_is_not_pass(self) -> None:
        """rate-only route 强制显式 NA / The rate-only route requires explicit NA."""

        component = QualityComponent(1.0, 1.0, QualityState.PASS, "synthetic_pass")
        q_rate = QualityEndpoint(1.0, QualityState.PASS, 0.5, {"synthetic": component}, (), 1.0)
        q_morph = QualityEndpoint(None, QualityState.NOT_APPLICABLE, None, {}, ("rate_only",), 0.0)
        result = QualityResult(q_rate, q_morph, "rate_only", {}, (), 1.0)
        result.validate_for_route(SignalRoute.ARTIFACT_RATE_ONLY)
        invalid = QualityResult(q_rate, q_rate, "invalid_claim", {}, (), 1.0)
        with self.assertRaisesRegex(ValueError, "not_applicable"):
            invalid.validate_for_route(SignalRoute.ARTIFACT_RATE_ONLY)

    def test_nonfinite_serializes_to_json_null(self) -> None:
        """缺失生理量必须是 null 而非 NaN token / Serialize unavailable values as null."""

        payload = to_strict_json_value({"nan": float("nan"), "inf": float("inf"), "ok": 1.0})
        encoded = json.dumps(payload, allow_nan=False)
        self.assertIn('"nan": null', encoded)
        self.assertIn('"inf": null', encoded)


class SpecificationLockTests(unittest.TestCase):
    """防止附件被悄然替换 / Prevent silent specification replacement."""

    def test_specification_byte_hash(self) -> None:
        """复算 41,122-byte 规格哈希 / Recompute the locked specification hash."""

        lock = json.loads((ROOT / "docs/spec/SPEC_LOCK.json").read_text(encoding="utf-8"))
        source = REPO / lock["source_path"]
        data = source.read_bytes()
        self.assertEqual(len(data), lock["source_bytes"])
        self.assertEqual(hashlib.sha256(data).hexdigest(), lock["source_sha256"])


if __name__ == "__main__":
    unittest.main()
