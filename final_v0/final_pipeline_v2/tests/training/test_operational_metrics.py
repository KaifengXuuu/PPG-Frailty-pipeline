"""Safe synthetic tests for environment-scoped operational measurements."""

from __future__ import annotations

import unittest

import numpy as np
import torch
from torch import nn

from ppg_frailty.training import (
    CPU_BATCH1_MEASURED_RUNS,
    CPU_BATCH1_WARMUP_RUNS,
    measure_cpu_batch1_operational_metrics,
)


class OperationalMetricsTests(unittest.TestCase):
    def test_tiny_cpu_model_reports_fixed_batch1_latency_and_parameter_count(self) -> None:
        model = nn.Linear(2, 3).eval()
        metrics = measure_cpu_batch1_operational_metrics(
            model,
            torch.zeros((1, 2), dtype=torch.float32),
            preprocessing=lambda values: np.asarray(values) * 2.0,
            preprocessing_input=np.zeros((1, 2), dtype=np.float32),
        )
        self.assertEqual(metrics.parameter_count, 9)
        self.assertEqual(metrics.parameter_count_definition, "torch_trainable_parameter_elements")
        self.assertEqual(metrics.warmup_runs, CPU_BATCH1_WARMUP_RUNS)
        self.assertEqual(metrics.measured_runs, CPU_BATCH1_MEASURED_RUNS)
        self.assertEqual(metrics.batch_size, 1)
        self.assertEqual(metrics.device, "cpu")
        self.assertFalse(metrics.preprocessing_included_in_model_latency)
        self.assertEqual(metrics.conda_environment, "ml")
        self.assertEqual(metrics.torch_intraop_threads, 1)
        self.assertGreaterEqual(metrics.model_latency_p95_ms, metrics.model_latency_p50_ms)
        self.assertIsNotNone(metrics.preprocessing_latency_p50_ms)
        self.assertEqual(
            set(metrics.inference_cost),
            {
                "cpu_batch1_model_only_p50_ms",
                "cpu_batch1_model_only_p95_ms",
            },
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
