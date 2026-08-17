#!/usr/bin/env python3
"""Generate auditable model cards from the frozen V1 model registry.

English: Cards are generated from one explicit table so model names, supported
representations, signal routes, and scientific limitations cannot drift independently.

中文：模型卡由一个显式表生成，使模型名称、支持的表征、信号路线和科学限制
保持同步；生成内容不得被当作尚未运行的性能声明。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CardDefinition:
    """One immutable model-card source row / 一条不可变模型卡源记录。"""

    canonical_name: str
    machine_id: str
    representation_modes: tuple[str, ...]
    signal_routes: tuple[str, ...]
    scientific_status: str
    deviation: str
    limitations: tuple[str, ...]


COMMON_LIMITS = (
    "No independent frailty test set is available; formal scores must be named oof_validation_*.",
    "No V1 performance is claimed until the same frozen 5×5 participant protocol is run.",
)


CARDS = (
    CardDefinition(
        "CompactCNN1D", "compact_cnn", ("raw",),
        ("direct_x_filter", "identity_direct"), "reference_single_network",
        "Preserves the reviewed project CNN (32/64/128, kernels 9/9/7); it is not Wang-FCN.",
        COMMON_LIMITS + ("Reference receptive field is local and requires the physical-time ablation.",),
    ),
    CardDefinition(
        "InceptionTimeFull", "inception_full", ("raw",),
        ("direct_x_filter", "identity_direct"), "reference_single_network",
        "A single reviewed Inception port; it is not the original five-network ensemble.",
        COMMON_LIMITS + ("The 39-sample longest kernel is 97.5 ms at 400 Hz.",),
    ),
    CardDefinition(
        "InceptionTimeSmall", "inception_small", ("raw",),
        ("direct_x_filter", "identity_direct"), "reference_single_network",
        "A reduced single-network project port, not the original ensemble.",
        COMMON_LIMITS + ("Lower capacity is an explicit project variant.",),
    ),
    CardDefinition(
        "InceptionTimeMatrix", "inception_matrix", ("feature_matrix",),
        ("direct_x_filter", "identity_direct"), "reference_single_network_mask_aware",
        "Uses the reviewed Inception body on OrderedFeatureMatrixV1 with mask-aware pooling.",
        COMMON_LIMITS + ("Requires a complete fold-local D×32 matrix schema and validity mask.",),
    ),
    CardDefinition(
        "InceptionTimeFiveMemberEnsemble", "inception_five_member_ensemble",
        ("raw", "feature_matrix"), ("direct_x_filter", "identity_direct"),
        "optional_five_member_probability_ensemble",
        "Five independently initialised members are averaged arithmetically at probability level.",
        COMMON_LIMITS + ("The reduced CPU test is not evidence that the full 5×5×5 budget ran.",),
    ),
    CardDefinition(
        "ROCKET", "rocket_numpy", ("feature_matrix",),
        ("direct_x_filter", "identity_direct"), "self_contained_project_rocket",
        "A deterministic NumPy/SciPy project implementation; not an aeon/sktime parity claim.",
        COMMON_LIMITS + ("Primary formal configuration requires 10,000 kernels and fold-local ridge.",),
    ),
    CardDefinition(
        "MiniROCKET", "minirocket_ablation", ("feature_matrix",),
        ("direct_x_filter", "identity_direct"), "named_engineering_ablation",
        "A low-cost project ablation; it is not an exact MiniROCKET port.",
        COMMON_LIMITS,
    ),
    CardDefinition(
        "LogisticRegressionL2", "logistic_regression", ("feature_vector",),
        ("direct_x_filter", "identity_direct", "non_identity_x_ar_rate_only"),
        "reference_feature_baseline", "Fold-local imputation/scaling precede an L2 logistic model.",
        COMMON_LIMITS + ("Only the frozen allowlist and explicit validity fields are eligible.",),
    ),
    CardDefinition(
        "RBFSVM", "rbf_svm", ("feature_vector",),
        ("direct_x_filter", "identity_direct", "non_identity_x_ar_rate_only"),
        "reference_feature_baseline", "Fold-local imputation/scaling precede an RBF SVM.",
        COMMON_LIMITS + ("Only outer-training data may determine fitted transforms.",),
    ),
    CardDefinition(
        "ExtraTrees", "extra_trees", ("feature_vector",),
        ("direct_x_filter", "identity_direct", "non_identity_x_ar_rate_only"),
        "reference_feature_baseline", "A project tree baseline under the common frozen protocol.",
        COMMON_LIMITS + ("Missing physiology is represented by validity, never valid zero.",),
    ),
    CardDefinition(
        "ShapeFormerEffectSize", "shapeformer_effect_size", ("raw", "feature_matrix"),
        ("direct_x_filter", "identity_direct"), "experimental_ineligible_for_parity_claim",
        (
            "Self-contained outer-fold-bound effect-size discovery plus non-overlapping "
            "patch embedding before mask-aware generic self-attention and trainable "
            "shapelet distances; not PISD/original parity."
        ),
        COMMON_LIMITS + (
            "Discovery method is required and effect_size_shapelets_v1 never substitutes for PISD.",
            "Input sampling rate and shapelet length in samples/seconds are mandatory provenance.",
            "Patch size is at least two samples; raw sample-token attention is structurally rejected.",
        ),
    ),
    CardDefinition(
        "FileBagFusionCompact", "fusion_compact", ("fusion",),
        ("direct_x_filter", "identity_direct"), "reference_file_level_fusion",
        "Pools raw windows to one file embedding, then concatenates one file vector exactly once.",
        COMMON_LIMITS + ("Per-window repeated file features are forbidden.",),
    ),
    CardDefinition(
        "FileBagFusionInception", "fusion_inception", ("fusion",),
        ("direct_x_filter", "identity_direct"), "reference_file_level_fusion",
        "Uses an Inception file encoder and concatenates the file vector only after pooling.",
        COMMON_LIMITS + ("The signal member remains a project single-network port.",),
    ),
)


def _render(card: CardDefinition) -> str:
    """Render one bilingual Markdown card / 渲染一份双语 Markdown 模型卡。"""

    representations = ", ".join(f"`{item}`" for item in card.representation_modes)
    routes = ", ".join(f"`{item}`" for item in card.signal_routes)
    limits = "\n".join(f"- {item}" for item in card.limitations)
    return f"""# {card.canonical_name}

- Machine ID / 机器 ID：`{card.machine_id}`
- Scientific status / 科学状态：`{card.scientific_status}`
- Representation mode / 表征：{representations}
- Eligible signal routes / 可用信号路线：{routes}
- Evaluation unit / 评估单位：participant after window→file→role-aware aggregation
- Independent test / 独立测试：absent; `independent_test=false`

## Identity and deviation / 身份与偏离

{card.deviation}

中文：本卡只描述已实现接口和命名边界，不把架构存在、单元测试通过或历史分数
解释为 corrected V1 性能证据。

## Limitations / 限制

{limits}

## Required provenance / 必需追溯字段

Every formal result must bind participant/file/role, repeat/fold/seed, config hash,
manifest and fold hashes, preprocessing and feature hashes, signal route, aggregation,
model state, environment, and coverage.  正式结果必须绑定上述全部身份字段。
"""


def generate(output: Path) -> tuple[Path, ...]:
    """Write every card and a deterministic index / 写全部模型卡和确定性索引。"""

    output.mkdir(parents=True, exist_ok=True)
    expected = {f"{card.machine_id}.md" for card in CARDS} | {"README.md"}
    unexpected = {path.name for path in output.glob("*.md")} - expected
    if unexpected:
        raise RuntimeError(f"unexpected pre-existing model cards: {sorted(unexpected)}")
    written: list[Path] = []
    for card in CARDS:
        path = output / f"{card.machine_id}.md"
        path.write_text(_render(card), encoding="utf-8")
        written.append(path)
    rows = "\n".join(
        f"- [{card.canonical_name}]({card.machine_id}.md): `{card.scientific_status}`"
        for card in CARDS
    )
    index = output / "README.md"
    index.write_text(
        "# Generated model cards / 自动生成模型卡\n\n"
        "Generated by `tools/generate_model_cards.py`; do not hand-edit individual cards.\n\n"
        "由唯一注册表生成；单份卡片不应手工修改。\n\n" + rows + "\n",
        encoding="utf-8",
    )
    written.append(index)
    return tuple(written)


def main() -> int:
    """CLI entry / 命令行入口。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "model_cards")
    arguments = parser.parse_args()
    paths = generate(arguments.output)
    print(f"generated_model_cards={len(paths) - 1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
