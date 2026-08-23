"""Focused contracts for the 146-by-variable-K matrix and Small Inception."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from ppg_frailty.contracts import EngineeringFeatureSequence, PulseResult, SignalRoute
from ppg_frailty.data.windows import WindowPlan
from ppg_frailty.features.window_matrix import (
    WINDOW_FEATURE_SCHEMA_VERSION,
    WindowFeatureExtraction,
    build_ordered_window_matrix,
    build_route_eligible_rate_pulse,
    transform_window_features,
    window_feature_names,
    FoldWindowFeatureTransform,
    _rate_features,
)
from ppg_frailty.quality.routing_timeline import (
    RoutingEvidence,
    build_routing_timeline,
    build_routing_windows,
    resolve_routing_evidence,
)
from ppg_frailty.models.inception import InceptionTimeSingleNetwork
from ppg_frailty.training.datasets import (
    FeatureMatrixDataset,
    SampleIdentity,
    collate_samples,
)


def _extraction(k: int, *, valid_rows: np.ndarray | None = None) -> WindowFeatureExtraction:
    names = window_feature_names()
    mask = np.ones(k, dtype=bool) if valid_rows is None else valid_rows
    values = np.arange(k * len(names), dtype=np.float64).reshape(k, len(names))
    values[~mask] = np.nan
    return WindowFeatureExtraction(
        sequence=EngineeringFeatureSequence(
            values=values,
            start_samples=np.arange(k, dtype=np.int64) * 800,
            valid_row_mask=mask,
            channel_schema=names,
            schema_version=WINDOW_FEATURE_SCHEMA_VERSION,
        ),
        value_validity=np.isfinite(values),
        row_tiers=tuple("excellent" if value else "excluded" for value in mask),
        reasons=(),
    )


def _transformed(k: int, *, valid_rows: np.ndarray | None = None) -> WindowFeatureExtraction:
    source = _extraction(k, valid_rows=valid_rows)
    transform = FoldWindowFeatureTransform(
        center=np.zeros(146),
        scale=np.ones(146),
        valid_count=np.ones(146, dtype=np.int64),
        feature_names=window_feature_names(),
        fitted_on_participant_ids=("p1",),
    )
    return transform_window_features(source, transform)


class WindowMatrixV2Tests(unittest.TestCase):
    def test_300_second_complete_window_counts_are_variable(self) -> None:
        plan_2 = WindowPlan(
            source_record_id="r",
            window_seconds=10.0,
            hop_seconds=2.0,
            end_alignment="start",
            short_record_action="reject",
            include_padded_tail=False,
            max_windows=None,
            cap_policy="not_applicable",
        )
        plan_5 = WindowPlan(
            source_record_id="r",
            window_seconds=10.0,
            hop_seconds=5.0,
            end_alignment="start",
            short_record_action="reject",
            include_padded_tail=False,
            max_windows=None,
            cap_policy="not_applicable",
        )
        self.assertEqual(len(plan_2.plan(300 * 400, 400.0)), 146)
        self.assertEqual(len(plan_5.plan(300 * 400, 400.0)), 59)
        matrix_146 = build_ordered_window_matrix(
            _transformed(146), provenance={"record_id": "r"}
        )
        matrix_59 = build_ordered_window_matrix(
            _transformed(59), provenance={"record_id": "r"}
        )
        self.assertEqual(matrix_146.values.shape, (146, 146))
        self.assertEqual(matrix_59.values.shape, (146, 59))
        self.assertEqual(
            matrix_59.provenance["padding_policy"],
            "none_at_record_storage_batch_only",
        )
        self.assertNotIn("selected_source_rows", matrix_59.provenance)

    def test_batch_only_padding_and_padding_values_do_not_change_logits(self) -> None:
        matrices = [
            build_ordered_window_matrix(
                _transformed(5), provenance={"record_id": "r1"}
            ),
            build_ordered_window_matrix(
                _transformed(9), provenance={"record_id": "r2"}
            ),
        ]
        identities = (
            SampleIdentity("p1", "r1", "B", 0, "direct"),
            SampleIdentity("p2", "r2", "B", 1, "direct"),
        )
        dataset = FeatureMatrixDataset.from_contracts(matrices, identities)
        self.assertEqual(dataset.sequence_lengths, (5, 9))
        batch = collate_samples([dataset[0], dataset[1]])
        self.assertEqual(tuple(batch["x"].shape), (2, 146, 9))
        self.assertEqual(batch["mask"].sum(dim=1).tolist(), [5, 9])

        model = InceptionTimeSingleNetwork(146, 3, variant="small").eval()
        self.assertEqual(
            sum(parameter.numel() for parameter in model.parameters()), 70_275
        )
        with torch.no_grad():
            baseline = model(batch["x"], batch["mask"])
            changed = batch["x"].clone()
            changed[0, :, 5:] = 999.0
            observed = model(changed, batch["mask"])
        self.assertEqual(tuple(baseline.shape), (2, 3))
        self.assertTrue(torch.equal(baseline, observed))

    def test_schema_has_only_146_physiological_window_predictors(self) -> None:
        names = window_feature_names()
        self.assertEqual(len(names), 146)
        self.assertEqual(len(set(names)), 146)
        for forbidden in ("sqi", "motion", "route", "coverage", "metadata"):
            self.assertFalse(any(forbidden in name.lower() for name in names))

    def test_successive_ppi_pair_cannot_cross_routing_cell_boundary(self) -> None:
        windows = build_routing_windows("r", 10 * 400)
        rows = tuple(
            resolve_routing_evidence(
                RoutingEvidence(
                    window=window,
                    sqi_mode="route",
                    sqi_assessed=True,
                    direct_q_rate_state="pass",
                    direct_q_morph_state="pass",
                ),
                role="B",
            )
            for window in windows
        )
        timeline = build_routing_timeline(
            record_id="r",
            participant_id="p1",
            role="B",
            n_samples=10 * 400,
            evidence=rows,
            config_sha256="a" * 64,
        )
        peak_times = np.asarray((3.0, 4.0, 5.0, 6.0, 7.0))
        pulse = PulseResult(
            peaks=np.rint(peak_times * 400).astype(np.int64),
            peak_timestamps_s=peak_times,
            accepted_peak_mask=np.ones(5, dtype=bool),
            interval_start_peak_indices=np.arange(4, dtype=np.int64),
            interval_stop_peak_indices=np.arange(1, 5, dtype=np.int64),
            ppi_s=np.asarray((1.0, 0.9, 1.1, 1.0)),
            valid_interval_mask=np.ones(4, dtype=bool),
            adjacency_mask=np.ones(4, dtype=bool),
            wavelength="RED",
            detector_version="fixture",
            confidence=np.ones(5),
            source_route=SignalRoute.DIRECT,
            detection_run_id="global-direct",
            interval_run_ids=np.asarray(["global-direct"] * 4),
        )
        values, validity = _rate_features((pulse,), timeline, 0, 10 * 400)
        self.assertTrue(validity[:9].all())
        self.assertFalse(validity[9:].any())
        self.assertTrue(np.isnan(values[9:]).all())

    def test_composite_rate_pulse_preserves_source_and_inserts_invalid_boundary(self) -> None:
        windows = build_routing_windows("r", 10 * 400)
        first = resolve_routing_evidence(
            RoutingEvidence(
                window=windows[0],
                sqi_mode="route",
                sqi_assessed=True,
                direct_q_rate_state="pass",
                direct_q_morph_state="pass",
            ),
            role="B",
        )
        second = resolve_routing_evidence(
            RoutingEvidence(
                window=windows[1],
                sqi_mode="route",
                sqi_assessed=True,
                direct_q_rate_state="fail",
                direct_q_morph_state="fail",
                motion_detector_enabled=True,
                motion_state="high",
                denoiser_enabled=True,
                denoiser_status="success",
                post_q_rate_state="pass",
            ),
            role="B",
        )
        timeline = build_routing_timeline(
            record_id="r",
            participant_id="p1",
            role="B",
            n_samples=10 * 400,
            evidence=(first, second),
            config_sha256="a" * 64,
        )

        def source_pulse(route: SignalRoute, times: tuple[float, ...], run: str):
            timestamps = np.asarray(times)
            count = timestamps.size - 1
            return PulseResult(
                peaks=np.rint(timestamps * 400).astype(np.int64),
                peak_timestamps_s=timestamps,
                accepted_peak_mask=np.ones(timestamps.size, dtype=bool),
                interval_start_peak_indices=np.arange(count),
                interval_stop_peak_indices=np.arange(1, count + 1),
                ppi_s=np.diff(timestamps),
                valid_interval_mask=np.ones(count, dtype=bool),
                adjacency_mask=np.ones(count, dtype=bool),
                wavelength="RED",
                detector_version="fixture",
                confidence=np.ones(timestamps.size),
                source_route=route,
                detection_run_id=run,
                interval_run_ids=np.asarray([run] * count),
            )

        composite = build_route_eligible_rate_pulse(
            timeline,
            source_pulse(SignalRoute.DIRECT, (3.0, 4.0, 5.0), "direct"),
            source_pulse(
                SignalRoute.ARTIFACT_RATE_ONLY,
                (5.0, 6.0, 7.0),
                "processed",
            ),
        )
        composite.validate_identity()
        routes = np.asarray(composite.interval_source_routes).astype(str)
        self.assertEqual(np.count_nonzero(routes == "routing_boundary"), 1)
        boundary = routes == "routing_boundary"
        self.assertFalse(np.asarray(composite.valid_interval_mask)[boundary].any())
        self.assertEqual(
            set(routes) - {"routing_boundary"},
            {SignalRoute.DIRECT.value, SignalRoute.ARTIFACT_RATE_ONLY.value},
        )


if __name__ == "__main__":
    unittest.main()
