"""Synthetic execution contracts for the isolated L0--L7 bridge cell."""

from __future__ import annotations

import hashlib
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import ppg_frailty.experiment as experiment
from ppg_frailty.catalog import resolved_catalog_payloads
from ppg_frailty.config import PipelineConfig
from ppg_frailty.contracts import SignalRoute
from ppg_frailty.legacy_bridge import resolve_legacy_bridge_profile
from ppg_frailty.provenance import stable_payload_sha256
from ppg_frailty.representations import RawWindows


PIPELINE_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PIPELINE_ROOT.parents[1]
SOURCE_SPECIFICATION = (
    "AA_TODO/old_version_compare_V2/"
    "CODEX_LEGACY_V2_BRIDGE_REVISED_9_CASES_WITH_PHASE0.md"
)
def _compact_config() -> PipelineConfig:
    payload = next(
        value
        for value in resolved_catalog_payloads(
            pipeline_root=PIPELINE_ROOT,
            line="line_b",
        )
        if value["config_id"] == "formal_compact_cnn_line_b_v2"
    )
    return PipelineConfig(
        payload,
        "synthetic_legacy_bridge_config",
        stable_payload_sha256(payload),
    )


def _reserved_bridge_config(profile_id: str) -> PipelineConfig:
    payload = _compact_config().to_dict()
    payload["config_id"] = (
        f"formal_compact_cnn_line_b_v2__legacy_bridge_{profile_id.lower()}"
    )
    return PipelineConfig(
        payload,
        "synthetic_reserved_legacy_bridge_config",
        stable_payload_sha256(payload),
    )


def _raw_record(row: object, maximum: int | None) -> dict[str, object]:
    del maximum
    sample_count = 2_000
    time_axis = np.arange(sample_count, dtype=np.float64) / 400.0
    frequency = float(row.class_id + 1)
    return {
        "record_id": row.record_id,
        "fs_hz": 400.0,
        "ppg": np.column_stack(
            (
                np.sin(2.0 * np.pi * frequency * time_axis),
                np.cos(2.0 * np.pi * frequency * time_axis),
            )
        ),
        "acc": np.column_stack(
            (np.sin(time_axis), np.cos(time_axis), np.ones(sample_count))
        ),
        "gyro": np.column_stack(
            (
                np.sin(2.0 * time_axis),
                np.cos(2.0 * time_axis),
                np.sin(3.0 * time_axis),
            )
        ),
        "recording_qc": {"status": "pass"},
        "recording_qc_profile": {"profile_id": "synthetic"},
    }


def _runtime_rows() -> list[object]:
    identities = (
        ("train_0", 0),
        ("train_1", 1),
        ("train_2", 2),
        ("heldout_0", 0),
        ("heldout_1", 1),
        ("heldout_2", 2),
    )
    return [
        SimpleNamespace(
            participant_id=participant,
            record_id=f"{participant}_B",
            role="B",
            qc_status="pass",
            duration_s=5.0,
            n_samples=2_000,
            fs=400.0,
            class_id=label,
            manifest_version="synthetic_manifest_v1",
        )
        for participant, label in identities
    ]


class _Registry:
    def get_split(self, repeat: int, fold: int) -> dict[str, object]:
        return {
            "repeat_index": repeat,
            "fold_index": fold,
            "split_seed": 42,
            "training_seed": 42,
            "train_participant_ids": ("train_0", "train_1", "train_2"),
            "oof_participant_ids": ("heldout_0", "heldout_1", "heldout_2"),
        }


def _bridge_execution(profile_id: str, config: PipelineConfig) -> object:
    source_hash = hashlib.sha256(
        (REPOSITORY_ROOT / SOURCE_SPECIFICATION).read_bytes()
    ).hexdigest()
    return experiment._resolve_legacy_bridge_execution(
        SimpleNamespace(
            repository_root=REPOSITORY_ROOT,
            input_path=lambda relative: PIPELINE_ROOT / relative,
        ),
        config,
        profile_id=profile_id,
        source_specification=SOURCE_SPECIFICATION,
        source_specification_sha256=source_hash,
    )


class LegacyBridgeExperimentRuntimeTest(unittest.TestCase):
    def test_large_record_tables_are_externalized_with_resolvable_index_paths(
        self,
    ) -> None:
        summary = {
            "status": "passed",
            "repeat_index": 0,
            "fold_index": 3,
            "metrics": {"balanced_accuracy": 0.5},
            "physical_recording_qc": [{"record_id": "p0_B"}],
            "route_artifacts": [{"record_id": "p0_B", "retained": True}],
        }
        compact = experiment._artifact_index_cell_summary(
            summary,
            artifact_prefix="repeat_00_fold_03",
        )

        self.assertIn("physical_recording_qc", summary)
        self.assertIn("route_artifacts", summary)
        self.assertNotIn("physical_recording_qc", compact)
        self.assertNotIn("route_artifacts", compact)
        self.assertEqual(compact["physical_recording_qc_row_count"], 1)
        self.assertEqual(compact["route_artifacts_row_count"], 1)
        self.assertEqual(
            compact["physical_recording_qc_artifact"],
            "repeat_00_fold_03/physical_recording_qc.json",
        )
        self.assertEqual(
            compact["route_artifacts_artifact"],
            "repeat_00_fold_03/route_artifacts.json",
        )
        self.assertEqual(compact["metrics"], summary["metrics"])

    def test_source_specification_and_effective_config_are_byte_bound(self) -> None:
        config = _compact_config()
        first = _bridge_execution("L1", config)
        second = _bridge_execution("L1", config)
        self.assertEqual(first.effective_config_hash, second.effective_config_hash)
        self.assertNotEqual(first.effective_config_hash, config.sha256)
        self.assertEqual(first.profile.profile_id, "L1")

        dropout = experiment._resolved_legacy_bridge_dropout_comparison(
            resolve_legacy_bridge_profile("L7"),
            config.section("model"),
        )
        self.assertFalse(dropout["changed"])
        self.assertEqual(
            dropout["legacy_resolved"]["encoder_stage_dropouts"],
            [0.10, 0.15],
        )
        self.assertEqual(
            dropout["current_registered"]["classifier_head_dropout"],
            0.20,
        )

        with self.assertRaisesRegex(
            experiment._ExperimentProtocolError,
            "source_specification_sha256_mismatch",
        ):
            experiment._resolve_legacy_bridge_execution(
                SimpleNamespace(
                    pipeline_root=PIPELINE_ROOT,
                    repository_root=REPOSITORY_ROOT,
                    input_path=lambda relative: PIPELINE_ROOT / relative,
                ),
                config,
                profile_id="L1",
                source_specification=SOURCE_SPECIFICATION,
                source_specification_sha256="0" * 64,
            )

    def test_bridge_entrypoint_has_no_phase0_gate_input(self) -> None:
        parameters = inspect.signature(
            experiment.run_legacy_bridge_outer_cell
        ).parameters
        self.assertNotIn("phase0_gate_path", parameters)
        self.assertFalse(
            any(
                "phase0" in name.lower() or "gate" in name.lower()
                for name in parameters
            )
        )

    def test_effective_hash_binds_live_manifest_and_split_not_audit(self) -> None:
        config = _compact_config()
        source_hash = hashlib.sha256(
            (REPOSITORY_ROOT / SOURCE_SPECIFICATION).read_bytes()
        ).hexdigest()
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            manifest_relative = Path(config.section("manifest")["path"])
            split_relative = Path(config.section("splits")["path"])
            for relative in (manifest_relative, split_relative):
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((PIPELINE_ROOT / relative).read_bytes())
            paths = SimpleNamespace(
                repository_root=REPOSITORY_ROOT,
                input_path=lambda relative: root / relative,
            )
            first = experiment._resolve_legacy_bridge_execution(
                paths,
                config,
                profile_id="L1",
                source_specification=SOURCE_SPECIFICATION,
                source_specification_sha256=source_hash,
            )
            manifest_path = root / manifest_relative
            manifest_path.write_bytes(manifest_path.read_bytes() + b"\n")
            second = experiment._resolve_legacy_bridge_execution(
                paths,
                config,
                profile_id="L1",
                source_specification=SOURCE_SPECIFICATION,
                source_specification_sha256=source_hash,
            )
            self.assertNotEqual(first.manifest_sha256, second.manifest_sha256)
            self.assertEqual(first.split_sha256, second.split_sha256)
            self.assertNotEqual(
                first.effective_config_hash,
                second.effective_config_hash,
            )
            for resolved in (first, second):
                self.assertFalse(hasattr(resolved, "phase0_gate_sha256"))
                self.assertFalse(hasattr(resolved, "phase0_decision"))

    def test_canonical_outer_and_full_reject_reserved_bridge_config(self) -> None:
        config = _reserved_bridge_config("L1")
        paths = SimpleNamespace()
        api = {
            "PipelinePaths": SimpleNamespace(discover=lambda: paths),
            "preflight_pipeline": lambda *_args, **_kwargs: (
                SimpleNamespace(status="passed"),
                config,
                (),
                None,
            ),
        }
        invocations = (
            lambda: experiment.run_outer_cell(
                "unused.yaml", 0, 0, "must_not_be_created"
            ),
            lambda: experiment.run_full_experiment(
                "unused.yaml",
                output_dir="must_not_be_created",
                repeats=(0,),
                folds=(0,),
            ),
        )
        for invoke in invocations:
            with self.subTest(entrypoint=invoke):
                with (
                    patch.object(experiment, "_runtime_imports", return_value=api),
                    patch.object(
                        experiment,
                        "_resolve_output_directory",
                        side_effect=AssertionError(
                            "reserved config must fail before output/model work"
                        ),
                    ),
                ):
                    with self.assertRaisesRegex(
                        experiment._ExperimentProtocolError,
                        "reserved_config_requires_dedicated_entrypoint",
                    ):
                        invoke()

    def test_l0_uses_fresh_loader_and_historical_short_record_padding(self) -> None:
        row = _runtime_rows()[0]
        state = experiment._RuntimeRecord(row=row)
        calls: list[tuple[str, int | None]] = []

        def loader(current: object, maximum: int | None) -> dict[str, object]:
            calls.append((str(current.record_id), maximum))
            return _raw_record(current, maximum)

        provenance = experiment._preprocess_legacy_bridge_records(
            [state],
            resolve_legacy_bridge_profile("L0"),
            None,
            loader,
        )
        self.assertEqual(calls, [(row.record_id, None)])
        self.assertTrue(state.retained)
        self.assertEqual(state.route, SignalRoute.DIRECT)
        self.assertEqual(state.raw_windows.values.shape, (1, 8, 960))
        self.assertEqual(state.raw_windows.start_samples.tolist(), [0])
        self.assertFalse(provenance["historical_cache_used_for_training"])
        # Five seconds at 64 Hz are real values; the remaining ten seconds are
        # the exact historical right-zero padding behavior.
        self.assertTrue(np.all(state.raw_windows.values[:, :, 320:] == 0.0))
        representation = experiment._legacy_bridge_representation_artifacts(
            [state],
            resolve_legacy_bridge_profile("L0"),
            (str(row.participant_id),),
            (),
        )
        self.assertEqual(
            representation["legacy_bridge_window_materialization"]["padding"],
            "zero_right_only_if_source_shorter_than_one_window",
        )

    def test_l3_and_l4_apply_distinct_normalization_stages(self) -> None:
        sample_count = 2_000
        base = np.linspace(-3.0, 7.0, sample_count, dtype=np.float64)
        views = SimpleNamespace(
            x_filter=np.column_stack((base, base * 3.0 + 20.0)),
            imu_processed={
                "dynamic_acc_mps2": np.column_stack(
                    (base + 10.0, base * 2.0, base * 4.0 - 8.0)
                ),
                "gyro_rads": np.column_stack(
                    (base * 0.1, base * 0.2 + 4.0, base * 0.3 - 2.0)
                ),
                "imu_valid_mask": np.ones(sample_count, dtype=bool),
            },
            validate=lambda: None,
        )
        l3_state = experiment._RuntimeRecord(
            row=SimpleNamespace(participant_id="train_0", record_id="l3"),
            views=views,
            retained=True,
        )
        experiment._extract_l3_bridge_raw(
            l3_state,
            resolve_legacy_bridge_profile("L3"),
        )
        self.assertTrue(l3_state.retained)
        medians = np.median(l3_state.raw_windows.values[0], axis=1)
        self.assertTrue(np.allclose(medians, 0.0, atol=1e-5))
        l3_provenance = experiment._legacy_bridge_representation_artifacts(
            [l3_state],
            resolve_legacy_bridge_profile("L3"),
            ("train_0",),
            (),
        )
        self.assertEqual(
            l3_provenance["legacy_bridge_window_materialization"]["padding"],
            "none_complete_windows_only",
        )

        def state(participant: str, offset: float) -> object:
            values = np.zeros((1, 8, 32), dtype=np.float32)
            values[:, :2, :] = offset
            for channel in range(6):
                values[:, channel + 2, :] = (
                    offset + channel + np.linspace(0.0, 1.0, 32)
                )
            return experiment._RuntimeRecord(
                row=SimpleNamespace(
                    participant_id=participant,
                    record_id=f"{participant}_B",
                ),
                retained=True,
                raw_windows=RawWindows(
                    values=values,
                    valid_mask=np.ones((1, 32), dtype=bool),
                    start_samples=np.asarray([0], dtype=np.int64),
                    candidate_count=1,
                    dropped_invalid_count=0,
                ),
            )

        states = [state("train_0", 1.0), state("train_1", 3.0), state("heldout", 50.0)]
        original_ppg = [value.raw_windows.values[:, :2, :].copy() for value in states]
        provenance = experiment._legacy_bridge_representation_artifacts(
            states,
            resolve_legacy_bridge_profile("L4"),
            ("train_0", "train_1"),
            ("heldout",),
        )
        self.assertEqual(
            set(provenance["raw_imu"]["fitted_on_participant_ids"]),
            {"train_0", "train_1"},
        )
        for before, after in zip(original_ppg, states):
            self.assertTrue(np.array_equal(before, after.raw_windows.values[:, :2, :]))

    def test_l1_cell_executes_real_model_trainer_and_sampling_artifact(self) -> None:
        config = _compact_config()
        bridge = _bridge_execution("L1", config)
        report = SimpleNamespace(
            fold_hash="f" * 64,
            manifest_hash="a" * 64,
            window_profiles={},
        )
        cell = experiment._execute_cell_unchecked(
            report,
            config,
            _runtime_rows(),
            _Registry(),
            SimpleNamespace(),
            repeat_index=0,
            fold_index=0,
            maximum_seconds=None,
            record_cap=None,
            epoch_override=None,
            loader=_raw_record,
            legacy_bridge=bridge,
        )
        cell.summary["scientific_scope"] = "synthetic_contract_test"

        self.assertEqual(cell.summary["status"], "passed")
        self.assertEqual(cell.summary["training_seed"], 42)
        self.assertEqual(cell.summary["balance_line"], "line_a_equal_files")
        self.assertEqual(cell.summary["config_hash"], bridge.effective_config_hash)
        selected_window = cell.summary["frozen_model_run_provenance"][
            "window_plan"
        ]["selected"]
        self.assertEqual(selected_window["padding"], "none_complete_windows_only")
        self.assertEqual(selected_window["min_valid_fraction"], 1.0)
        self.assertEqual(len(cell.summary["sampling_diagnostics"]), 10)
        self.assertEqual(len(cell.window_rows), 3)
        self.assertEqual(len(cell.subject_rows), 3)
        self.assertTrue(
            any(
                "training_participant_balanced_accuracy" in row
                for row in cell.summary["training_history"]
            )
        )
        self.assertFalse(
            any(
                "sampling_diagnostics" in row
                for row in cell.summary["training_history"]
            )
        )
        first_sampling = cell.summary["sampling_diagnostics"][0]
        self.assertEqual(first_sampling["sampler_identity"], "exhaustive_shuffle_without_replacement")
        self.assertEqual(first_sampling["duplicate_draw_fraction"], 0.0)

        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw) / "cell"
            experiment._write_cell_artifacts(output, cell)
            manifest = json.loads(
                (output / "run_manifest.json").read_text(encoding="utf-8")
            )
            metrics_index = json.loads(
                (output / "metrics_per_fold_seed.json").read_text(encoding="utf-8")
            )
            diagnostics = json.loads(
                (output / "sampling_diagnostics.json").read_text(
                    encoding="utf-8"
                )
            )
            physical_qc = json.loads(
                (output / "physical_recording_qc.json").read_text(
                    encoding="utf-8"
                )
            )
            route_artifacts = json.loads(
                (output / "route_artifacts.json").read_text(encoding="utf-8")
            )
            self.assertIn(
                "sampling_diagnostics.json", manifest["mandatory_artifacts"]
            )
            self.assertIn(
                "physical_recording_qc.json", manifest["mandatory_artifacts"]
            )
            self.assertIn("route_artifacts.json", manifest["mandatory_artifacts"])
            self.assertEqual(diagnostics["profile_id"], "L1")
            self.assertEqual(len(diagnostics["rows"]), 10)
            self.assertNotIn(
                "sampling_diagnostics", manifest["cell"]
            )
            self.assertEqual(
                manifest["cell"]["sampling_diagnostics_row_count"], 10
            )
            for index_cell in (
                manifest["cell"],
                metrics_index["cells"][0],
            ):
                self.assertNotIn("physical_recording_qc", index_cell)
                self.assertNotIn("route_artifacts", index_cell)
                self.assertEqual(
                    index_cell["physical_recording_qc_artifact"],
                    "physical_recording_qc.json",
                )
                self.assertEqual(
                    index_cell["route_artifacts_artifact"],
                    "route_artifacts.json",
                )
                self.assertEqual(
                    index_cell["physical_recording_qc_row_count"],
                    len(cell.summary["physical_recording_qc"]),
                )
                self.assertEqual(
                    index_cell["route_artifacts_row_count"],
                    len(cell.summary["route_artifacts"]),
                )
            self.assertEqual(
                physical_qc["rows"], cell.summary["physical_recording_qc"]
            )
            self.assertEqual(
                route_artifacts["rows"], cell.summary["route_artifacts"]
            )


if __name__ == "__main__":
    unittest.main()
