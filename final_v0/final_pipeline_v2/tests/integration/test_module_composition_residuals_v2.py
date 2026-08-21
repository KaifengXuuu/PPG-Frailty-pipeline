"""Runtime composition checks for configurable V2 modules."""

from __future__ import annotations

import copy
import hashlib
import unittest
from pathlib import Path

import numpy as np

from ppg_frailty.config import (
    PipelineConfig,
    canonical_json_bytes,
    load_config,
    validate_config_payload,
)
from ppg_frailty.contracts import SignalRoute
from ppg_frailty.experiment import (
    _model_input_sampling_rate_hz,
    _prepare_dl_input_dataset,
)
from ppg_frailty.module_registry import ARTIFACT_MODULES
from ppg_frailty.training.datasets import FileBagDataset, SampleIdentity


ROOT = Path(__file__).resolve().parents[2]


def _effective_config(payload: dict[str, object]) -> PipelineConfig:
    resolved = validate_config_payload(copy.deepcopy(payload))
    digest = hashlib.sha256(canonical_json_bytes(resolved)).hexdigest()
    return PipelineConfig(resolved, "memory://composition-test", digest)


class V2ModuleCompositionResidualTests(unittest.TestCase):
    """Ensure exposed module combinations have real runtime consumers."""

    def test_generic_dl_resampling_is_configurable_for_fusion(self) -> None:
        payload = load_config(
            ROOT / "configs/reference_static_fusion_v2.yaml"
        ).to_dict()
        payload["signal"]["dl_resampling"] = {
            "enabled": True,
            "target_fs_hz": 200.0,
        }
        config = _effective_config(payload)
        self.assertEqual(_model_input_sampling_rate_hz(config), 200.0)

        identities = (
            SampleIdentity("p1", "f1", "B", 0, "direct"),
            SampleIdentity("p2", "f2", "R", 1, "direct"),
        )
        bags = (
            np.arange(2 * 8 * 20, dtype=np.float32).reshape(2, 8, 20),
            np.arange(8 * 20, dtype=np.float32).reshape(1, 8, 20),
        )
        masks = (
            np.ones((2, 20), dtype=bool),
            np.ones((1, 20), dtype=bool),
        )
        features = np.asarray(((1.0, 0.0), (0.0, 1.0)), dtype=np.float32)
        dataset = FileBagDataset(bags, features, identities, masks)
        transformed, provenance_key, profile = _prepare_dl_input_dataset(
            dataset,
            "fusion",
            config.section("signal")["dl_resampling"],
        )

        self.assertEqual(provenance_key, "dl_input_resampling")
        self.assertEqual(
            [bag.shape for bag in transformed.window_bags],
            [(2, 8, 10), (1, 8, 10)],
        )
        self.assertEqual(
            [mask.shape for mask in transformed.sample_masks],
            [(2, 10), (1, 10)],
        )
        self.assertEqual(transformed.identities, identities)
        np.testing.assert_array_equal(transformed.file_features, features)
        self.assertEqual(profile["source_fs_hz"], 400.0)
        self.assertEqual(profile["target_fs_hz"], 200.0)

    def test_dl_resampling_rejects_only_inapplicable_combinations(self) -> None:
        feature_payload = load_config(
            ROOT / "configs/reference_static_feature_vector_v2.yaml"
        ).to_dict()
        feature_payload["signal"]["dl_resampling"] = {
            "enabled": True,
            "target_fs_hz": 200.0,
        }
        with self.assertRaisesRegex(ValueError, "only for raw or fusion"):
            validate_config_payload(feature_payload)

        fusion_payload = load_config(
            ROOT / "configs/reference_static_fusion_v2.yaml"
        ).to_dict()
        fusion_payload["signal"]["dl_resampling"] = {
            "enabled": True,
            "target_fs_hz": 100.0,
            "case_id": "compactcnn1d__fs_100",
        }
        with self.assertRaisesRegex(ValueError, "fixed-kernel.*requires raw"):
            validate_config_payload(fusion_payload)

    def test_inactive_window_profiles_cannot_be_hash_only_axes(self) -> None:
        raw = load_config(
            ROOT / "configs/reference_static_role_aware_v2.yaml"
        ).to_dict()
        raw["windows"]["engineering"]["hop_s"] = 4.0
        with self.assertRaisesRegex(ValueError, "windows.engineering is inactive"):
            validate_config_payload(raw)

        vector = load_config(
            ROOT / "configs/reference_static_feature_vector_v2.yaml"
        ).to_dict()
        vector["windows"]["raw_dl"]["hop_s"] = 1.0
        with self.assertRaisesRegex(ValueError, "windows.raw_dl is inactive"):
            validate_config_payload(vector)

        fusion = load_config(
            ROOT / "configs/reference_static_fusion_v2.yaml"
        ).to_dict()
        fusion["windows"]["engineering"]["hop_s"] = 4.0
        fusion["windows"]["raw_dl"]["hop_s"] = 1.0
        resolved = validate_config_payload(fusion)
        self.assertEqual(resolved["windows"]["engineering"]["hop_s"], 4.0)
        self.assertEqual(resolved["windows"]["raw_dl"]["hop_s"], 1.0)

    def test_engineering_window_requires_a_predictor_consumer(self) -> None:
        for filename in (
            "reference_static_feature_vector_v2.yaml",
            "reference_static_fusion_v2.yaml",
        ):
            with self.subTest(filename=filename):
                payload = load_config(ROOT / "configs" / filename).to_dict()
                payload["features"]["enabled_groups"].remove(
                    "engineering_summary"
                )
                payload["windows"]["engineering"]["hop_s"] = 4.0
                with self.assertRaisesRegex(
                    ValueError,
                    "windows.engineering is inactive",
                ):
                    validate_config_payload(payload)

        matrix = load_config(
            ROOT / "configs/reference_static_feature_matrix_v2.yaml"
        ).to_dict()
        matrix["windows"]["engineering"]["hop_s"] = 4.0
        with self.assertRaisesRegex(ValueError, "fixed 10 s/2 s"):
            validate_config_payload(matrix)

    def test_rate_only_artifact_route_is_not_a_feature_matrix_fallback(self) -> None:
        descriptors = {item.module_id: item for item in ARTIFACT_MODULES}
        for module_id, descriptor in descriptors.items():
            if module_id != "identity":
                self.assertEqual(
                    descriptor.representation_modes,
                    ("feature_vector",),
                )

        payload = load_config(
            ROOT / "configs/reference_static_feature_matrix_v2.yaml"
        ).to_dict()
        payload["quality"]["mode"] = "route"
        payload["quality"]["rate_threshold"] = 0.5
        payload["quality"]["morph_threshold"] = 0.5
        payload["artifact"].update(
            {
                "denoiser_enabled": True,
                "reducer": "spectral_mask",
                "reducer_version": "spectral_mask_v1",
                "degraded_policy": "denoise_then_extract_rate_features",
                "parameters": {},
            }
        )
        with self.assertRaisesRegex(
            ValueError,
            "only with representation_mode='feature_vector'",
        ):
            validate_config_payload(payload)


if __name__ == "__main__":
    unittest.main()
