# ShapeFormerEffectSizeFixedV1

- Machine ID / 机器 ID：`shapeformer_effect_size_fixed_v1`
- Scientific status / 科学状态：`experimental_ineligible_for_parity_claim`
- Representation mode / 表征：`raw`
- Eligible signal routes / 可用信号路线：`direct_x_filter`, `identity_direct`
- Evaluation unit / 评估单位：participant after window→file, then config-dependent Line A equal-files or Line B equal-role-families aggregation
- Current classifier role scope / 当前分类 role 范围：SQI off; only B and R are admitted
- Execution status / 执行状态：registered/constructible; scientific benchmark not run
- Independent test / 独立测试：absent; `independent_test=false`

## Identity and deviation / 身份与偏离

Fixed 128-sample/stride-64 outer-fold effect-size discovery plus non-overlapping patch embedding before mask-aware generic self-attention and trainable shapelet distances; not PISD/original parity.

中文：本卡只描述已实现接口和命名边界，不把架构存在、单元测试通过或历史分数
解释为 V2 性能证据。

## Limitations / 限制

- No independent frailty test set is available; formal scores must be named oof_validation_*.
- No V2 performance is claimed until the same frozen 5×5 participant protocol is run.
- The implementation is registered/constructible; the scientific benchmark has not been run.
- Discovery method is effect_size_fixed_v1 and never substitutes for channel_specific_osd.
- Input sampling rate and shapelet length in samples/seconds are mandatory provenance.
- Patch size is at least two samples; raw sample-token attention is structurally rejected.

## Required provenance / 必需追溯字段

Every formal result must bind participant/file/role, repeat/fold/seed, config hash,
manifest and fold hashes, preprocessing and feature hashes, signal route, aggregation,
model state, environment, and coverage.  正式结果必须绑定上述全部身份字段。
