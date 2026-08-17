"""§8.2/§8.3 ECG fixture 与统计守卫 / ECG fixture and regression guards.

中文：冻结的 ECG R-peak 时间轴用于验证一对一事件匹配和三个策略的对称
量化 schema；它只证明指标实现与接口，不声称是真实 pulse-transit-time-ppg
性能。另用确定性标签打乱证明分类器不会从无关标签产生虚假高 BA。

English: frozen ECG R-peak timestamps exercise one-to-one matching and a symmetric
three-policy metric schema. This proves metric/interface behavior only; it is not an
external pulse-transit-time-ppg performance claim. A deterministic label-shuffle
challenge separately guards against implausibly high balanced accuracy.
"""

from __future__ import annotations

import unittest

import numpy as np
from sklearn.metrics import balanced_accuracy_score

from ppg_frailty.models.factory import ModelInputSpec, create_model
from ppg_frailty.peaks.pairing import match_events


# 中文：模拟独立 ECG R peaks 的冻结秒时间；English: frozen ECG-like R-peak times.
FROZEN_ECG_R_PEAKS_S = np.array(
    [0.80, 1.60, 2.40, 3.20, 4.00, 4.80, 5.60, 6.40, 7.20, 8.00],
    dtype=np.float64,
)
POLICY_SCHEMA = {
    "policy",
    "fixture_scope",
    "event_tolerance_s",
    "true_positive",
    "false_positive",
    "false_negative",
    "event_precision",
    "event_recall",
    "event_f1",
    "timing_mae_s",
    "hr_mae_bpm",
    "hr_rmse_bpm",
    "hr_bias_bpm",
    "ppi_mae_s",
    "coverage",
    "q_morph_state",
}


def _matched_pairs(reference: np.ndarray, predicted: np.ndarray, tolerance_s: float) -> list[tuple[int, float]]:
    """重建唯一匹配用于 HR/PPI 误差 / Reconstruct unique pairs for interval errors."""

    used = np.zeros(predicted.size, dtype=bool)
    pairs: list[tuple[int, float]] = []
    for reference_index, event in enumerate(reference):
        candidates = np.flatnonzero((~used) & (np.abs(predicted - event) <= tolerance_s))
        if candidates.size:
            chosen = int(candidates[np.argmin(np.abs(predicted[candidates] - event))])
            used[chosen] = True
            pairs.append((reference_index, float(predicted[chosen])))
    return pairs


def _policy_metrics(policy: str, predicted: np.ndarray, *, q_morph_state: str) -> dict[str, object]:
    """形成三个策略共用的量化 schema / Build one symmetric policy result."""

    tolerance_s = 0.15
    reference = FROZEN_ECG_R_PEAKS_S
    predicted = np.asarray(predicted, dtype=np.float64)
    match = match_events(reference, predicted, tolerance_s=tolerance_s)
    pairs = _matched_pairs(reference, predicted, tolerance_s)
    ppi_errors = []
    hr_errors = []
    for (left_index, left_predicted), (right_index, right_predicted) in zip(pairs, pairs[1:]):
        # 中文：跨缺失 beat 的间隔不是一拍 PPI；English: skip intervals across a missed beat.
        if right_index != left_index + 1:
            continue
        reference_ppi = float(reference[right_index] - reference[left_index])
        predicted_ppi = float(right_predicted - left_predicted)
        ppi_errors.append(predicted_ppi - reference_ppi)
        hr_errors.append(60.0 / predicted_ppi - 60.0 / reference_ppi)
    if not ppi_errors:
        raise AssertionError("fixture policy yielded no adjacent matched intervals")
    hr_array = np.asarray(hr_errors)
    return {
        "policy": policy,
        "fixture_scope": "frozen_synthetic_ecg_contract_not_external_dataset_benchmark",
        "event_tolerance_s": tolerance_s,
        "true_positive": match.true_positive,
        "false_positive": match.false_positive,
        "false_negative": match.false_negative,
        "event_precision": match.precision,
        "event_recall": match.recall,
        "event_f1": match.f1,
        "timing_mae_s": match.timing_mae_s,
        "hr_mae_bpm": float(np.mean(np.abs(hr_array))),
        "hr_rmse_bpm": float(np.sqrt(np.mean(np.square(hr_array)))),
        "hr_bias_bpm": float(np.mean(hr_array)),
        "ppi_mae_s": float(np.mean(np.abs(ppi_errors))),
        "coverage": match.true_positive / reference.size,
        "q_morph_state": q_morph_state,
    }


class ExternalEcgFixtureContractTest(unittest.TestCase):
    """验证 ECG 对齐指标和三策略对称性 / Verify ECG metrics and policy symmetry."""

    def test_one_to_one_match_rejects_duplicate_prediction(self) -> None:
        """同一参考事件只能消耗一个预测 / One reference consumes one prediction."""

        result = match_events(
            np.array([1.0]),
            np.array([0.98, 1.02]),
            tolerance_s=0.05,
        )
        self.assertEqual((result.true_positive, result.false_positive, result.false_negative), (1, 1, 0))

    def test_raw_quality_reducer_have_symmetric_quantitative_schema(self) -> None:
        """raw/quality/reducer 输出完全同构 / All three policies emit the same schema."""

        jitter = np.array([0.010, -0.012, 0.008, -0.006, 0.011, -0.009, 0.004, 0.007, -0.010, 0.005])
        raw = np.sort(np.concatenate((FROZEN_ECG_R_PEAKS_S + jitter, np.array([4.37]))))
        quality_only = FROZEN_ECG_R_PEAKS_S + jitter
        reducer_rate_only = np.delete(FROZEN_ECG_R_PEAKS_S + 0.5 * jitter, 5)
        rows = [
            _policy_metrics("raw_no_denoise", raw, q_morph_state="available"),
            _policy_metrics("quality_only", quality_only, q_morph_state="available"),
            _policy_metrics("non_identity_reducer", reducer_rate_only, q_morph_state="not_applicable"),
        ]
        self.assertTrue(all(set(row) == POLICY_SCHEMA for row in rows))
        self.assertEqual({row["policy"] for row in rows}, {"raw_no_denoise", "quality_only", "non_identity_reducer"})
        self.assertGreater(rows[1]["event_precision"], rows[0]["event_precision"])
        self.assertEqual(rows[2]["q_morph_state"], "not_applicable")
        for row in rows:
            self.assertGreaterEqual(row["coverage"], 0.0)
            self.assertLessEqual(row["coverage"], 1.0)
            self.assertTrue(np.isfinite(row["hr_mae_bpm"]))
            self.assertTrue(np.isfinite(row["ppi_mae_s"]))


class LabelShuffleRegressionTest(unittest.TestCase):
    """标签打乱性能 sanity check / Label-shuffle performance sanity check."""

    def test_label_shuffle_destroys_separable_feature_signal(self) -> None:
        """打乱训练标签后 BA 必须回到机会附近 / Shuffling labels destroys high BA."""

        rng = np.random.default_rng(20260815)
        train_labels = np.tile(np.arange(3), 120)
        test_labels = np.tile(np.arange(3), 60)
        class_centers = np.eye(3, 6) * 4.0
        train_values = class_centers[train_labels] + rng.normal(0.0, 0.25, (train_labels.size, 6))
        test_values = class_centers[test_labels] + rng.normal(0.0, 0.25, (test_labels.size, 6))
        names = tuple(f"signal_{index}" for index in range(6))
        spec = ModelInputSpec(
            "feature_vector",
            n_classes=3,
            n_file_features=6,
            feature_names=names,
        )
        participants = tuple(f"train_{index:03d}" for index in range(train_labels.size))
        reference = create_model({"model_id": "LogisticRegressionL2", "seed": 42}, spec)
        reference.fit(train_values, train_labels, participant_ids=participants)
        reference_ba = balanced_accuracy_score(test_labels, reference.predict_proba(test_values).argmax(axis=1))

        shuffled_labels = rng.permutation(train_labels)
        shuffled = create_model({"model_id": "LogisticRegressionL2", "seed": 42}, spec)
        shuffled.fit(train_values, shuffled_labels, participant_ids=participants)
        shuffled_ba = balanced_accuracy_score(test_labels, shuffled.predict_proba(test_values).argmax(axis=1))
        self.assertGreater(reference_ba, 0.95)
        self.assertLess(shuffled_ba, 0.55)
        self.assertGreater(reference_ba - shuffled_ba, 0.40)


if __name__ == "__main__":
    unittest.main()

