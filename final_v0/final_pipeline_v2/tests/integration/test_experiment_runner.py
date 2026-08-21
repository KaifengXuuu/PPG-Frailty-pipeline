'''真实 outer-fold runner 集成测试 / Frozen outer-fold runner integration tests.

The synthetic fixture exercises the same preprocessing, direct SQI, empirical
outer-train calibration, feature extraction, unified trainer and OOF path as a
real run. No scientific gate is mocked or relaxed.

合成 fixture 与真实执行共用预处理、direct SQI、仅 outer-train 拟合的经验校准、
特征提取、统一训练器和 OOF 路径；不 mock、不放宽任何科学门禁。
'''

from __future__ import annotations

import ast
import copy
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq

import ppg_frailty.experiment as experiment
from ppg_frailty.config import PipelineConfig
from ppg_frailty.models.feature_baselines import FeatureVectorBaseline
from ppg_frailty.pipeline import PipelinePaths, preflight_pipeline
from ppg_frailty.provenance import stable_payload_sha256


CONFIG_PATH = 'configs/motion_benchmark_v1.yaml'


class _SyntheticRegistry:
    '''Minimal immutable registry facade / 最小不可变 registry facade。'''

    def __init__(self, split: dict[str, object]) -> None:
        self._split = copy.deepcopy(split)

    def get_split(self, repeat_index: int, fold_index: int) -> dict[str, object]:
        if (repeat_index, fold_index) != (0, 0):
            raise KeyError((repeat_index, fold_index))
        return copy.deepcopy(self._split)


def _synthetic_record(row: object, maximum: int | None) -> dict[str, object]:
    '''Build deterministic dual-wavelength PPG and six-axis IMU.

    构建确定性双波长 PPG 与六轴 IMU；类别只改变生理可解释的心搏频率，
    不向训练器额外传递 outer 标签。
    '''
    samples = min(int(row.n_samples), maximum or int(row.n_samples))
    time_axis = np.arange(samples, dtype=np.float64) / float(row.fs)
    # Keep SQI morphology identical across classes so empirical quantiles test
    # leakage isolation, not synthetic quality imbalance. 类别不改变 SQI 形态。
    frequency_hz = 1.15
    phase = 0.0
    fundamental = np.sin(2.0 * np.pi * frequency_hz * time_axis + phase)
    harmonic = 0.28 * np.sin(
        4.0 * np.pi * frequency_hz * time_axis + phase + 0.35
    )
    pulse = fundamental + harmonic
    ppg = np.column_stack(
        (
            1000.0 + 28.0 * pulse,
            1200.0 + 21.0 * np.sin(
                2.0 * np.pi * frequency_hz * time_axis + phase + 0.08
            ) + 5.0 * harmonic,
        )
    )
    acceleration = np.column_stack(
        (
            0.01 * np.sin(2.0 * np.pi * 0.25 * time_axis),
            0.01 * np.cos(2.0 * np.pi * 0.25 * time_axis),
            np.full(samples, 1.0),
        )
    )
    gyroscope = np.zeros((samples, 3), dtype=np.float64)
    return {
        'record_id': str(row.record_id),
        'ppg': ppg,
        'acc': acceleration,
        'gyro': gyroscope,
        'acc_unit': 'g',
        'gyro_unit': 'deg/s',
        'fs_hz': float(row.fs),
        'timestamps_s': time_axis,
    }


def _synthetic_contract() -> tuple[object, PipelineConfig, list[object], object, object]:
    '''Derive a direct-route test config from the validated formal config.

    从已通过 preflight 的正式配置派生 direct-route 测试配置；仅改变合成数据
    角色与 run-lock drop 策略，经验 SQI 校准及训练协议保持正式实现。
    '''
    paths = PipelinePaths.discover()
    report, formal, _, _ = preflight_pipeline(
        CONFIG_PATH,
        mode='full',
        paths=paths,
    )
    payload = formal.to_dict()
    payload['config_id'] = 'synthetic_experiment_runner_contract_v1'
    payload['roles'] = ['B']
    payload['artifact']['motion_detector_enabled'] = False
    payload['artifact']['degraded_policy'] = 'drop'
    payload['artifact']['reducer'] = 'identity'
    payload['artifact']['reducer_version'] = 'identity_v1'
    config = PipelineConfig(
        payload=payload,
        source_path='synthetic_in_memory',
        sha256=stable_payload_sha256(payload),
    )
    train_ids = tuple(
        f'train_{class_id}_{replicate}'
        for replicate in range(2)
        for class_id in range(3)
    )
    oof_ids = tuple(f'oof_{class_id}_0' for class_id in range(3))
    rows = []
    for participant_id in (*train_ids, *oof_ids):
        class_id = int(participant_id.split('_')[1])
        rows.append(
            SimpleNamespace(
                participant_id=participant_id,
                record_id=f'{participant_id}_B',
                role='B',
                class_id=class_id,
                qc_status='pass',
                duration_s=24.0,
                n_samples=9600,
                fs=400.0,
            )
        )
    split = {
        'repeat_index': 0,
        'fold_index': 0,
        'split_seed': 42,
        'training_seed': 42,
        'train_participant_ids': train_ids,
        'oof_participant_ids': oof_ids,
    }
    return report, config, rows, _SyntheticRegistry(split), paths


class ExperimentRunnerTest(unittest.TestCase):
    '''Runner contract tests / Runner 合同测试。'''

    def test_public_entry_points_are_unique(self) -> None:
        '''Prevent stale placeholder definitions / 防止旧骨架重复覆盖真实入口。'''
        source_path = Path(experiment.__file__).resolve()
        module = ast.parse(source_path.read_text(encoding='utf-8'))
        names = [
            node.name
            for node in module.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        self.assertEqual(names.count('run_reduced_fold_experiment'), 1)
        self.assertEqual(names.count('run_full_experiment'), 1)
        self.assertNotIn('experiment_execution_contract_not_yet_satisfied', source_path.read_text())

    def test_reduced_smoke_epoch_override_is_backend_specific(self) -> None:
        estimator = SimpleNamespace(
            section=lambda name: {
                'model_id': 'logistic_regression'
            } if name == 'model' else {}
        )
        deep = SimpleNamespace(
            section=lambda name: {
                'model_id': 'compact_cnn'
            } if name == 'model' else {}
        )

        self.assertIsNone(experiment._epoch_override_for_backend(estimator, 1))
        self.assertEqual(experiment._epoch_override_for_backend(deep, 1), 1)
        self.assertIsNone(experiment._epoch_override_for_backend(deep, None))

    def test_all_missing_train_column_preserves_frozen_width_without_warning(self) -> None:
        '''Keep all registered columns through each allow-listed imputer.

        三个白名单 baseline 均不得因 outer-train 全缺失列而告警或改变冻结列宽。
        '''
        values = np.asarray(
            [
                [0.0, np.nan, 1.0],
                [0.2, np.nan, 0.8],
                [1.0, np.nan, 0.0],
                [1.2, np.nan, 0.2],
                [2.0, np.nan, -1.0],
                [2.2, np.nan, -0.8],
            ],
            dtype=np.float64,
        )
        labels = np.asarray([0, 0, 1, 1, 2, 2])
        participants = tuple(f'train_{index}' for index in range(values.shape[0]))
        options = {
            'logistic_regression': {
                'logistic_c': 1.0,
                'logistic_max_iter': 5000,
                'logistic_solver': 'lbfgs',
            },
            'rbf_svm': {
                'svm_kernel': 'rbf',
                'svm_probability': True,
                'svm_c': 1.0,
                'svm_gamma': 'scale',
            },
            'extra_trees': {
                'extra_trees_n_estimators': 500,
                'extra_trees_n_jobs': 1,
                'extra_trees_max_features': 'sqrt',
                'extra_trees_min_samples_leaf': 1,
            },
        }
        for model_id in ('logistic_regression', 'rbf_svm', 'extra_trees'):
            with self.subTest(model_id=model_id), warnings.catch_warnings():
                warnings.simplefilter('error')
                model = FeatureVectorBaseline(
                    model_id,
                    ('observed_a', 'route_specific_empty', 'observed_b'),
                    seed=42,
                    **options[model_id],
                )
                model.fit(values, labels, participant_ids=participants)
                transformed = model.pipeline.named_steps['imputer'].transform(values)
                self.assertEqual(transformed.shape, values.shape)
                self.assertTrue(np.isfinite(transformed).all())

    def test_synthetic_three_class_same_route_end_to_end(self) -> None:
        '''Train and create nonempty three-class OOF through the production route.'''
        report, config, rows, registry, paths = _synthetic_contract()
        cell = experiment._execute_vector_cell(
            report,
            config,
            rows,
            registry,
            paths,
            repeat_index=0,
            fold_index=0,
            maximum_seconds=24.0,
            record_cap=1,
            epoch_override=1,
            loader=_synthetic_record,
        )
        self.assertEqual(cell.summary['status'], 'passed')
        fitted = set(
            cell.summary['sqi_calibrator_provenance']['fitted_on_participant_ids']
        )
        self.assertEqual(fitted, {row.participant_id for row in rows if row.participant_id.startswith('train_')})
        self.assertFalse(any(value.startswith('oof_') for value in fitted))
        self.assertEqual({row.label for row in cell.subject_rows}, {0, 1, 2})
        self.assertTrue(all(row.retained for row in cell.subject_rows))
        self.assertTrue(all(len(row.probabilities) == 3 for row in cell.subject_rows))

    def test_real_frozen_fold_train_only_nonempty_oof(self) -> None:
        '''Run the real 60-second endpoint in a clean, self-deleting directory.'''
        paths = PipelinePaths.discover()
        artifact_root = paths.pipeline_root / 'artifacts'
        with tempfile.TemporaryDirectory(
            prefix='experiment_real_integration_',
            dir=artifact_root,
        ) as temporary:
            result = experiment.run_reduced_fold_experiment(
                CONFIG_PATH,
                output_dir=Path(temporary).resolve() / 'result',
            )
            self.assertEqual(result.status, 'passed')
            self.assertEqual(
                result.scientific_scope,
                'smoke_not_scientific_benchmark',
            )
            self.assertTrue(result.provenance['frozen_outer_split'])
            self.assertTrue(result.provenance['outer_train_only_calibrator'])
            output = Path(result.output_dir)
            subject = pq.read_table(
                output / 'oof_subject_predictions.parquet'
            ).to_pydict()
            self.assertGreater(len(subject['participant_id']), 0)
            self.assertTrue(any(subject['retained']))


if __name__ == '__main__':
    unittest.main()
