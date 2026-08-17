"""Generated model-card identity tests / 自动生成模型卡身份测试。"""

from __future__ import annotations

from pathlib import Path
import unittest

from ppg_frailty.models.factory import CANONICAL_MODEL_REGISTRY


V1_ROOT = Path(__file__).resolve().parents[2]


class ModelCardTests(unittest.TestCase):
    """Cards must cover the exact model registry / 模型卡必须精确覆盖唯一注册表。"""

    def test_every_registered_model_has_one_card(self) -> None:
        """Reject missing or orphan cards / 拒绝缺失卡片和孤儿卡片。"""

        expected = {f'{machine_id}.md' for machine_id in CANONICAL_MODEL_REGISTRY.values()}
        observed = {
            path.name for path in (V1_ROOT / 'model_cards').glob('*.md')
            if path.name != 'README.md'
        }
        self.assertEqual(observed, expected)

    def test_shapeformer_card_locks_experimental_patch_identity(self) -> None:
        """Lock patch-first/no-parity wording / 锁定 patch-first 与无 parity 声明。"""

        text = (V1_ROOT / "model_cards" / "shapeformer_effect_size.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("experimental_ineligible_for_parity_claim", text)
        self.assertIn("patch embedding before mask-aware generic self-attention", text)
        self.assertIn("not PISD/original parity", text)
        self.assertIn("samples/seconds are mandatory provenance", text)
        self.assertIn("raw sample-token attention is structurally rejected", text)

    def test_cards_state_evaluation_and_claim_limits(self) -> None:
        """No card may imply an independent test / 不得暗示已完成独立测试。"""

        for canonical_name, machine_id in CANONICAL_MODEL_REGISTRY.items():
            text = (V1_ROOT / 'model_cards' / f'{machine_id}.md').read_text(
                encoding='utf-8'
            )
            with self.subTest(canonical_name=canonical_name):
                self.assertIn(f'# {canonical_name}', text)
                self.assertIn('`independent_test=false`', text)
                self.assertIn('participant after window→file→role-aware aggregation', text)
                self.assertIn('Limitations / 限制', text)


if __name__ == '__main__':
    unittest.main()
