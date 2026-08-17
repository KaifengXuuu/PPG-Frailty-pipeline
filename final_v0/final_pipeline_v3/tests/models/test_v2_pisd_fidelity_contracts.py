"""Safe-suite upstream PISD endpoint and information-gain fidelity locks."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from ppg_frailty.models import normalize_model_id
from ppg_frailty.models.pisd_port import (
    DISCOVERY_POSITION_SEARCH_BOUNDARY_RULE,
    INFORMATION_GAIN_SPLIT_RULE,
    PIP_SELECTION_RULE,
    _order_selected_candidate_rows_upstream,
    _information_gain,
    _insertion_stage_three_pip_intervals,
    _pisd_min_distance,
    _rank_candidate_rows_upstream,
)
from ppg_frailty.models.shapeformer_literature import (
    _ChunkedShapeFormerAttention,
    _upstream_observed_position_width,
)


class PisdUpstreamFidelityContracts(unittest.TestCase):
    """No-training numerical contracts included by the standard-library safe gate."""

    def test_insertion_stage_endpoints_equal_frozen_upstream_fixture(self) -> None:
        signal = np.asarray(
            (0.0, 4.0, 1.0, 5.0, 0.0, 3.0, 1.0, 2.0, 0.0, 1.0)
        )
        self.assertEqual(
            _insertion_stage_three_pip_intervals(signal, 5),
            ((0, 9), (3, 9), (0, 4), (4, 9), (3, 5)),
        )

    def test_perpendicular_pip_endpoints_equal_upstream_counterexample(self) -> None:
        signal = np.random.default_rng(1).normal(size=12)
        self.assertEqual(
            PIP_SELECTION_RULE,
            "upstream_zscored_time_index_perpendicular_distance_first_max",
        )
        self.assertEqual(
            _insertion_stage_three_pip_intervals(signal, 6),
            (
                (0, 11),
                (3, 11),
                (0, 4),
                (4, 11),
                (3, 6),
                (6, 11),
                (4, 7),
            ),
        )

    def test_pcs_left_boundary_equals_frozen_upstream_counterexample(self) -> None:
        rng = np.random.default_rng(10)
        source = rng.normal(size=12)
        target = rng.normal(size=12)
        self.assertEqual(
            DISCOVERY_POSITION_SEARCH_BOUNDARY_RULE,
            "upstream_pcs_start_minus_w_plus_1_end_plus_w_half_open",
        )
        self.assertAlmostEqual(
            _pisd_min_distance(
                target,
                source[3:7],
                source_start=3,
                source_end=7,
                position_search_neighbourhood_samples=2,
                position_chunk_size=256,
            ),
            4.90973464426476,
            places=14,
        )

    def test_upstream_equal_ig_ties_and_final_start_order_are_frozen(self) -> None:
        rows = [
            (-1.0, 10, 0, 8, 10, 5),
            (-1.0, 11, 0, 1, 3, 5),
            (-1.0, 12, 1, 5, 7, 5),
            (-1.0, 13, 1, 3, 4, 5),
        ]
        ranked = _rank_candidate_rows_upstream(rows)
        self.assertEqual([row[1] for row in ranked], [13, 12, 11, 10])
        ordered = _order_selected_candidate_rows_upstream(ranked[:3])
        self.assertEqual(
            [(row[3], row[1]) for row in ordered],
            [(1, 11), (3, 13), (5, 12)],
        )

    def test_attention_matches_upstream_formula_without_probability_dropout(self) -> None:
        torch.manual_seed(7)
        attention = _ChunkedShapeFormerAttention(4, 2, 0.95, 2)
        attention.train()
        tokens = torch.randn(2, 3, 4)
        actual = attention(tokens)
        keys = attention.key(tokens).reshape(2, 3, 2, 2).permute(0, 2, 3, 1)
        values = attention.value(tokens).reshape(2, 3, 2, 2).transpose(1, 2)
        queries = attention.query(tokens).reshape(2, 3, 2, 2).transpose(1, 2)
        weights = torch.softmax(torch.matmul(queries, keys) * (4.0 ** -0.5), dim=-1)
        expected = torch.matmul(weights, values).transpose(1, 2).reshape(2, 3, 4)
        expected = attention.output_norm(expected)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual, attention(tokens))

    def test_position_one_hot_width_is_observed_max_plus_one(self) -> None:
        self.assertEqual(
            _upstream_observed_position_width(torch.tensor((0, 3, 2))),
            4,
        )
        self.assertEqual(
            _upstream_observed_position_width(torch.tensor((5, 9, 7))),
            10,
        )

    def test_information_gain_is_numerically_equal_to_local_upstream(self) -> None:
        distances = np.arange(1, 21, dtype=np.float64) / 100.0
        target_class = np.asarray(
            (1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1),
            dtype=bool,
        )
        self.assertEqual(INFORMATION_GAIN_SPLIT_RULE, "upstream_positive_recall_grid_0p2")
        self.assertEqual(
            _information_gain(distances, target_class),
            0.10803154614559995,
        )
        self.assertEqual(
            _information_gain(
                np.asarray((0.1, 0.2, 0.3)),
                np.asarray((True, False, False)),
            ),
            -1.0,
        )

    def test_information_gain_without_target_class_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "no positive samples"):
            _information_gain(
                np.asarray((0.1, 0.2, 0.3)),
                np.asarray((False, False, False)),
            )

    def test_unretained_multichannel_route_has_no_pseudo_ablation_identity(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported model_id"):
            normalize_model_id("shapeformer_multichannel_pip_centered_ig")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
