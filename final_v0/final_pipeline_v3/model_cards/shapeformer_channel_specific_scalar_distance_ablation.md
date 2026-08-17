# ShapeFormerChannelSpecificScalarDistanceAblation

- Machine ID / 机器 ID：`shapeformer_channel_specific_scalar_distance_ablation`
- Scientific status / 科学状态：`named_optional_ablation_not_literature_shapeformer`
- Representation mode / 表征：`raw`
- Eligible signal routes / 可用信号路线：`direct_x_filter`, `identity_direct`
- Evaluation unit / 评估单位：participant after window→file, then config-dependent Line A equal-files or Line B equal-role-families aggregation
- Current classifier role scope / 当前分类 role 范围：SQI off; only B and R are admitted
- Execution status / 执行状态：registered/constructible; scientific benchmark not run
- Independent test / 独立测试：absent; `independent_test=false`

## Identity and deviation / 身份与偏离

Uses the same fold-local channel-specific variable OSD bank, but reduces each shapelet to one global z-normalised minimum-distance scalar and concatenates those scalars with patch attention. This is not PISDPort, not the literature ShapeBlock/token-fusion architecture, and is never a fallback.

中文：本卡只描述已实现接口和命名边界，不把架构存在、单元测试通过或历史分数
解释为 V2 性能证据。

## Limitations / 限制

- No independent frailty test set is available; formal scores must be named oof_validation_*.
- No V2 performance is claimed until the same frozen 5×5 participant protocol is run.
- The implementation is registered/constructible; the scientific benchmark has not been run.
- This optional downstream ablation is constructible but is not part of the 13 default candidate slots.

## Required provenance / 必需追溯字段

Every formal result must bind participant/file/role, repeat/fold/seed, config hash,
manifest and fold hashes, preprocessing and feature hashes, signal route, aggregation,
model state, environment, and coverage.  正式结果必须绑定上述全部身份字段。
