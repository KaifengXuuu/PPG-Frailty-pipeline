"""V2 quality-mode and decision-contract smoke tests / V2质量模式与决策合同测试。"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from ppg_frailty.contracts import SignalRoute
from ppg_frailty.experiment import (
    _ExperimentProtocolError,
    _RuntimeRecord,
    _quality_mode,
    _retain_without_quality_routing,
)
from ppg_frailty.v2_contract import resolve_balance_line, validate_quality_mode
from ppg_frailty.signal.sqi import SqiDiagnosticComponent, SqiDiagnostics


class _Config:
    """Minimal explicit config facade / 最小显式配置接口。"""

    def __init__(self, mode: str, ready: bool = False) -> None:
        self.mode = mode
        self.ready = ready

    def section(self, name: str) -> dict[str, object]:
        if name == "quality":
            return {"mode": self.mode, "supervised_route_ready": self.ready}
        if name == "artifact":
            return {"reducer": "identity"}
        raise KeyError(name)

    def to_dict(self) -> dict[str, object]:
        return {"quality": self.section("quality")}


class _SqiConfig:
    """Diagnostics-only fixture facade / 仅诊断测试桩。"""

    @staticmethod
    def from_resolved(_: object) -> object:
        return SimpleNamespace(calibrator="unresolved")


class V2QualityModeTest(unittest.TestCase):
    """Prove SQI off/diagnostics cannot route or drop / 证明两种模式不分流。"""

    def test_off_retains_without_evaluating_sqi(self) -> None:
        state = _RuntimeRecord(row=SimpleNamespace(record_id="P01_B"), views=object())
        with patch(
            "ppg_frailty.experiment._runtime_imports",
            return_value={"SignalRoute": SignalRoute},
        ):
            provenance = _retain_without_quality_routing(
                [state], _Config("off"), diagnostics_only=False
            )
        self.assertTrue(state.retained)
        self.assertIs(state.route, SignalRoute.DIRECT)
        self.assertEqual(state.route_status, "retained_direct_quality_off")
        self.assertEqual(provenance["classification_effect"], "none")

    def test_diagnostic_failure_never_drops_record(self) -> None:
        state = _RuntimeRecord(row=SimpleNamespace(record_id="P01_R1"), views=object())

        def fail_diagnostic(*_: object, **__: object) -> object:
            raise RuntimeError("diagnostic fixture failure")

        api = {
            "SignalRoute": SignalRoute,
            "SqiConfig": _SqiConfig,
            "evaluate_quality_diagnostics": fail_diagnostic,
        }
        with patch("ppg_frailty.experiment._runtime_imports", return_value=api):
            provenance = _retain_without_quality_routing(
                [state], _Config("diagnostics_only"), diagnostics_only=True
            )
        self.assertTrue(state.retained)
        self.assertIsNone(state.reason)
        self.assertIn("diagnostics_only_failed", state.diagnostic_reason or "")
        self.assertEqual(provenance["classification_effect"], "none")

    def test_diagnostics_archives_raw_values_without_decision_fields(self) -> None:
        state = _RuntimeRecord(row=SimpleNamespace(record_id="P01_R1"), views=object())
        observed = SqiDiagnostics(
            components={
                "snr_db": SqiDiagnosticComponent(12.5, True, "observed"),
            },
            coverage=1.0,
            route="direct",
            reasons=(),
        )
        api = {
            "SignalRoute": SignalRoute,
            "SqiConfig": _SqiConfig,
            "evaluate_quality_diagnostics": lambda *_args, **_kwargs: observed,
        }
        with patch("ppg_frailty.experiment._runtime_imports", return_value=api):
            provenance = _retain_without_quality_routing(
                [state],
                _Config("diagnostics_only"),
                diagnostics_only=True,
            )
        self.assertTrue(state.retained)
        self.assertIs(state.route, SignalRoute.DIRECT)
        self.assertEqual(
            state.diagnostic_components["components"]["snr_db"]["raw_value"],
            12.5,
        )
        for name in (
            "aggregation_performed",
            "weights_applied",
            "endpoint_thresholds_applied",
            "affects_classification",
        ):
            self.assertFalse(state.diagnostic_components[name])
        self.assertEqual(provenance["classification_effect"], "none")
        self.assertEqual(
            provenance["method"],
            "diagnostics_only_raw_components_no_weights_thresholds_or_fit",
        )

    def test_route_is_disabled_until_supervised(self) -> None:
        self.assertEqual(_quality_mode(_Config("off")), "off")
        self.assertEqual(_quality_mode(_Config("diagnostics_only")), "diagnostics_only")
        for ready in (False, True):
            with self.assertRaisesRegex(
                _ExperimentProtocolError, "no_frozen_supervised_artifact"
            ):
                _quality_mode(_Config("route", ready=ready))
            with self.assertRaisesRegex(ValueError, "frozen supervised artifact"):
                validate_quality_mode("route", supervised_route_ready=ready)

    def test_balance_lines_must_match_both_stages(self) -> None:
        resolved = resolve_balance_line(
            "line_a_equal_files",
            training_balance="equal_files",
            aggregation="equal_files_no_role_layer",
        )
        self.assertEqual(resolved.line_id, "line_a_equal_files")
        with self.assertRaisesRegex(ValueError, "balance line mismatch"):
            resolve_balance_line(
                "line_b_equal_role_families",
                training_balance="equal_files",
                aggregation="equal_role_families",
            )


if __name__ == "__main__":
    unittest.main()
