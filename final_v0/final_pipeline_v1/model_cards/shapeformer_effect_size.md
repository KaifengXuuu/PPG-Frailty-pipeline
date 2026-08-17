# ShapeFormerEffectSize

- Machine ID / 机器 ID：`shapeformer_effect_size`
- Scientific status / 科学状态：`experimental_ineligible_for_parity_claim`
- Representation mode / 表征：`raw`, `feature_matrix`
- Eligible signal routes / 可用信号路线：`direct_x_filter`, `identity_direct`
- Evaluation unit / 评估单位：participant after window→file→role-aware aggregation
- Independent test / 独立测试：absent; `independent_test=false`

## Identity and deviation / 身份与偏离

Self-contained outer-fold-bound effect-size discovery plus non-overlapping patch embedding before mask-aware generic self-attention and trainable shapelet distances; not PISD/original parity.

中文：本卡只描述已实现接口和命名边界，不把架构存在、单元测试通过或历史分数
解释为 corrected V1 性能证据。

## Limitations / 限制

- No independent frailty test set is available; formal scores must be named oof_validation_*.
- No V1 performance is claimed until the same frozen 5×5 participant protocol is run.
- Discovery method is required and effect_size_shapelets_v1 never substitutes for PISD.
- Input sampling rate and shapelet length in samples/seconds are mandatory provenance.
- Patch size is at least two samples; raw sample-token attention is structurally rejected.

## Required provenance / 必需追溯字段

Every formal result must bind participant/file/role, repeat/fold/seed, config hash,
manifest and fold hashes, preprocessing and feature hashes, signal route, aggregation,
model state, environment, and coverage.  正式结果必须绑定上述全部身份字段。
