'''Physical-time CNN/Inception ablation tests / 物理时间消融测试。'''

from __future__ import annotations

import unittest

import torch

from ppg_frailty.models.compact_cnn import CompactCNN1D, trainable_parameter_count
from ppg_frailty.models.inception import InceptionTimeSingleNetwork
from ppg_frailty.models.time_scale import (
    REFERENCE_INCEPTION_KERNEL_DURATIONS_S,
    build_physical_time_cases,
    create_time_scaled_model,
    inception_local_receptive_field,
    realize_kernel_durations,
)


class PhysicalTimeAblationTests(unittest.TestCase):
    '''Keep declared seconds, odd samples and default snapshots auditable.

    中文：同时冻结声明秒数、实际奇数样本与默认参数快照。
    '''

    def test_reference_400_hz_kernels_and_receptive_field(self) -> None:
        '''Reproduce 39/19/9 and 229 samples / 复现规范数值。'''

        realized = realize_kernel_durations(
            REFERENCE_INCEPTION_KERNEL_DURATIONS_S, fs_hz=400.0
        )
        self.assertEqual(realized.sample_counts, (39, 19, 9))
        self.assertEqual(
            inception_local_receptive_field(realized.sample_counts, depth=6), 229
        )
        self.assertAlmostEqual(229 / 400.0, 0.5725)

    def test_grid_covers_fs_window_dilation_and_four_modes(self) -> None:
        '''All mandated comparison axes exist / 所有强制比较轴均已物化。'''

        cases = build_physical_time_cases()
        self.assertEqual(len(cases), 4 * 2 * 2 * 4)
        self.assertEqual({case.dl_fs_hz for case in cases}, {100.0, 160.0, 200.0, 400.0})
        self.assertEqual({case.raw_window_s for case in cases}, {5.0, 10.0})
        self.assertEqual({case.dilation for case in cases}, {1, 2})
        self.assertEqual(
            {case.representation_mode for case in cases},
            {'raw', 'feature_vector', 'feature_matrix', 'fusion'},
        )
        self.assertTrue(all(
            value % 2 == 1
            for case in cases for value in case.inception_kernels.sample_counts
        ))

    def test_optional_controls_preserve_default_parameter_snapshots(self) -> None:
        '''Default models remain byte-architecture compatible / 默认架构不漂移。'''

        compact = CompactCNN1D(8, 3)
        full = InceptionTimeSingleNetwork(8, 3, variant='full')
        small = InceptionTimeSingleNetwork(8, 3, variant='small')
        self.assertEqual(trainable_parameter_count(compact), 79_139)
        self.assertEqual(trainable_parameter_count(full), 456_579)
        self.assertEqual(trainable_parameter_count(small), 57_027)

    def test_seconds_derived_kernel_and_dilation_execute(self) -> None:
        '''A generated case reaches a real forward pass / 生成条件能真实前向。'''

        compact = create_time_scaled_model(
            'CompactCNN1D', n_channels=8, n_classes=3,
            dl_fs_hz=160.0, dilation=2, seed=42,
        )
        compact_output = compact(torch.randn(2, 8, 5 * 160))
        self.assertEqual(tuple(compact_output.shape), (2, 3))
        self.assertEqual(compact.dilations, (2, 2, 2))
        inception = create_time_scaled_model(
            'InceptionTimeFull', n_channels=8, n_classes=3,
            dl_fs_hz=100.0, dilation=2, seed=42,
        )
        inception_output = inception(torch.randn(2, 8, 5 * 100))
        self.assertEqual(tuple(inception_output.shape), (2, 3))
        self.assertEqual(inception.dilation, 2)
        self.assertEqual(
            inception.physical_time_provenance['acquisition_and_feature_grid_hz'], 400.0
        )


if __name__ == '__main__':
    unittest.main()
