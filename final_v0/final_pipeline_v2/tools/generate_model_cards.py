#!/usr/bin/env python3
"""Generate auditable model cards from the frozen V2 model registry.

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
    "No V2 performance is claimed until the same frozen 5×5 participant protocol is run.",
    "The implementation is registered/constructible; the scientific benchmark has not been run.",
)


CARDS = (
    CardDefinition(
        "CompactCNN1D", "compact_cnn", ("raw",),
        ("direct_x_filter", "identity_direct"), "reference_single_network",
        "Preserves the reviewed project CNN (32/64/128, kernels 9/9/7); it is not Wang-FCN.",
        COMMON_LIMITS + (
            "V2-019 changes fs/context/dilation one factor at a time while kernel sample counts stay fixed; kernels are not converted to physical time.",
        ),
    ),
    CardDefinition(
        "InceptionTimeFull", "inception_full", ("raw",),
        ("direct_x_filter", "identity_direct"), "reference_single_network",
        "A single reviewed Inception port; it is not the original five-network ensemble.",
        COMMON_LIMITS + (
            "The 39-sample longest kernel is 97.5 ms at 400 Hz.",
            "V2-019 keeps 39/19/9 samples fixed across registered fs/context/dilation cases; it does not physical-time match kernels.",
        ),
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
        COMMON_LIMITS + ("Requires a complete fold-local registry-derived D×K matrix schema and validity mask.",),
    ),
    CardDefinition(
        "InceptionTimeFullFiveMemberEnsemble", "inception_full_five_member_ensemble",
        ("raw",), ("direct_x_filter", "identity_direct"),
        "optional_five_member_probability_ensemble",
        (
            "The historical model ID is retained as a compatibility alias. Runtime cardinality is "
            "derived from the explicit unique `member_seeds` roster (one or more members), and all "
            "declared member probabilities are averaged exactly. The checked-in formal comparison "
            "preset remains a five-member experiment; it does not constrain ordinary V2 configs."
        ),
        COMMON_LIMITS + ("The reduced CPU test is not evidence that the full 5×5×5 budget ran.",),
    ),
    CardDefinition(
        "InceptionTimeMatrixFiveMemberEnsemble", "inception_matrix_five_member_ensemble",
        ("feature_matrix",), ("direct_x_filter", "identity_direct"),
        "optional_five_member_probability_ensemble",
        (
            "The historical model ID is retained as a compatibility alias. Runtime cardinality is "
            "derived from the explicit unique `member_seeds` roster (one or more members), and all "
            "declared member probabilities are averaged exactly. The checked-in formal comparison "
            "preset remains a five-member experiment; it does not constrain ordinary V2 configs."
        ),
        COMMON_LIMITS + ("The reduced CPU test is not evidence that the full 5×5×5 budget ran.",),
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
        "ShapeFormerChannelSpecificOSD", "shapeformer_channel_specific_osd", ("raw",),
        ("direct_x_filter", "identity_direct"), "implemented_not_benchmarked_high_compute",
        (
            "Fold-local channel-specific OSD/PISD uses floor(0.20*T) PIPs from actual T "
            "(minimum five), the upstream z-scored time-index perpendicular-distance "
            "PIP selector, exhaustive insertion-stage variable candidates bounded by "
            "three consecutive PIPs, and no fixed candidate length/stride/cap. Discovery "
            "uses the upstream PCS [start-w+1,end+w) boundary. Candidates are enumerated "
            "class/channel/source/candidate, ranked with default NumPy argsort then reverse, "
            "and each selected class bank is finally ordered by start sample. Each "
            "candidate is ranked with the reviewed upstream target-positive recall "
            "0.2-grid information-gain rule (including its -1 no-grid sentinel). Each "
            "ShapeBlock searches raw segments only on its source channel within "
            "source start/end +/-128 samples, emits l1(selected)-l2(shapelet), adds "
            "channel/start/end embeddings whose widths are observed max+1 and IG weighting, "
            "then fuses shape attention (without probability dropout, matching upstream "
            "forward) with the full eight-channel generic branch."
        ),
        COMMON_LIMITS + (
            "Three shapelets per class and the participant/file-balanced cap of 180 windows are project capacity controls.",
            "Every shapelet archives source channel name/index, sample/second endpoints, length, and discovery-window identity.",
            "The persisted information_gain_split_rule is upstream_positive_recall_grid_0p2; exhaustive all-threshold IG is not the reference.",
            "Discovery PCS uses start-w+1 while downstream ShapeBlock uses start-w, matching the two distinct upstream implementations.",
            "Candidate enumeration, default-argsort tie ranking, final start ordering, observed-max+1 embedding widths, and unused attention dropout are persisted architecture identities.",
            "A PISD failure is explicit and cannot silently select another discovery method.",
            "At 5 s x 400 Hz, T=2000 and the derived PIP count is 400; exhaustive discovery is intentionally high-compute and has not been benchmarked.",
        ),
    ),
    CardDefinition(
        "ShapeFormerChannelSpecificScalarDistanceAblation",
        "shapeformer_channel_specific_scalar_distance_ablation",
        ("raw",),
        ("direct_x_filter", "identity_direct"),
        "named_optional_ablation_not_literature_shapeformer",
        (
            "Uses the same fold-local channel-specific variable OSD bank, but reduces "
            "each shapelet to one global z-normalised minimum-distance scalar and "
            "concatenates those scalars with patch attention. This is not PISDPort, "
            "not the literature ShapeBlock/token-fusion architecture, and is never a fallback."
        ),
        COMMON_LIMITS + (
            "This optional downstream ablation is constructible but is not part of the 13 default candidate slots.",
        ),
    ),
    CardDefinition(
        "ShapeFormerEffectSizeFixedV1", "shapeformer_effect_size_fixed_v1", ("raw",),
        ("direct_x_filter", "identity_direct"), "experimental_ineligible_for_parity_claim",
        (
            "Outer-fold fixed-length effect-size discovery defaults to 128 samples and "
            "stride 64; both controls are runtime-selectable and provenance-bound. It "
            "uses non-overlapping patch embedding before mask-aware generic self-attention "
            "and trainable shapelet distances; not PISD/original parity."
        ),
        COMMON_LIMITS + (
            "Discovery method is effect_size_fixed_v1 and never substitutes for channel_specific_osd.",
            "Input sampling rate and shapelet length in samples/seconds are mandatory provenance.",
            "Patch size is at least two samples; raw sample-token attention is structurally rejected.",
        ),
    ),
    CardDefinition(
        "ShapeFormerLegacyEffectSizePort",
        "shapeformer_legacy_effect_size_port",
        ("raw",),
        ("direct_x_filter", "identity_direct"),
        "legacy_parallel_ablation_not_osd_parity",
        (
            "Preserves the historical channel-wise class-v-rest effect-map "
            "discovery and its functional local-convolution plus source-position "
            "shape-token downstream. Defaults map to three shapelets per class, "
            "128-sample shapelets, stride 64, a 180-window class-balanced cap, "
            "eight candidates per class/channel, 48/128 embeddings, 256 FFN, "
            "four heads, dropout 0.30, and a 64-sample shapelet search span."
        ),
        COMMON_LIMITS + (
            "This is an isolated historical comparison module, not channel-specific OSD/PISD parity.",
            "Discovery is fitted only on the exact verified outer-training dataset and repeated on the exact all-29 final-refit scope.",
            "Complete unpadded windows are required by the historical downstream.",
            "The historical len_w=64 bookkeeping did not affect forward; the real local convolution width is 8 and no dead len_w option is exposed.",
            "Historical processes/verbose controls affected execution or console output only and are deliberately not model inputs or hash-only fields.",
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
    CardDefinition(
        "FileBagFusion", "file_bag_fusion", ("fusion",),
        ("direct_x_filter", "identity_direct"),
        "optional_composable_signal_encoder",
        (
            "Composes one registered raw forward_features encoder with a file-level "
            "feature vector after window pooling. Compact, Inception, faithful "
            "channel-specific OSD ShapeFormer, its scalar-distance ablation, the "
            "newer effect-size model, and the isolated legacy effect-size port "
            "are selectable signal modules."
        ),
        COMMON_LIMITS + (
            "Shapelet discovery is derived only from verified outer-training file bags.",
            "File features are never repeated per window and never enter shapelet discovery.",
            "This optional composer is not an additional default catalogue candidate.",
        ),
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
- Evaluation unit / 评估单位：participant after window→file, then config-dependent Line A equal-files or Line B equal-role-families aggregation
- Current classifier role scope / 当前分类 role 范围：SQI off; only B and R are admitted
- Execution status / 执行状态：registered/constructible; scientific benchmark not run
- Independent test / 独立测试：absent; `independent_test=false`

## Identity and deviation / 身份与偏离

{card.deviation}

中文：本卡只描述已实现接口和命名边界，不把架构存在、单元测试通过或历史分数
解释为 V2 性能证据。

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
