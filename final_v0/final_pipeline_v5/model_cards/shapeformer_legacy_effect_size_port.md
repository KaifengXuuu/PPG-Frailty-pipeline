# ShapeFormerLegacyEffectSizePort

- Machine ID / 机器 ID：`shapeformer_legacy_effect_size_port`
- Scientific status / 科学状态：`legacy_parallel_ablation_not_osd_parity`
- Representation mode / 表征：`raw`
- Eligible signal routes / 可用信号路线：`direct_x_filter`, `identity_direct`
- Evaluation unit / 评估单位：participant after window→file, then config-dependent Line A equal-files or Line B equal-role-families aggregation
- Current classifier role scope / 当前分类 role 范围：SQI off; only B and R are admitted
- Execution status / 执行状态：registered/constructible; scientific benchmark not run
- Independent test / 独立测试：absent; `independent_test=false`

## Identity and deviation / 身份与偏离

Preserves the historical channel-wise class-v-rest effect-map discovery and its functional local-convolution plus source-position shape-token downstream. Defaults map to three shapelets per class, 128-sample shapelets, stride 64, a 180-window class-balanced cap, eight candidates per class/channel, 48/128 embeddings, 256 FFN, four heads, dropout 0.30, and a 64-sample shapelet search span.

中文：本卡只描述已实现接口和命名边界，不把架构存在、单元测试通过或历史分数
解释为 V2 性能证据。

## Limitations / 限制

- No independent frailty test set is available; formal scores must be named oof_validation_*.
- No V2 performance is claimed until the same frozen 5×5 participant protocol is run.
- The implementation is registered/constructible; the scientific benchmark has not been run.
- This is an isolated historical comparison module, not channel-specific OSD/PISD parity.
- Discovery is fitted only on the exact verified outer-training dataset and repeated on the exact all-29 final-refit scope.
- Complete unpadded windows are required by the historical downstream.
- The historical len_w=64 bookkeeping did not affect forward; the real local convolution width is 8 and no dead len_w option is exposed.
- Historical processes/verbose controls affected execution or console output only and are deliberately not model inputs or hash-only fields.

## Required provenance / 必需追溯字段

Every formal result must bind participant/file/role, repeat/fold/seed, config hash,
manifest and fold hashes, preprocessing and feature hashes, signal route, aggregation,
model state, environment, and coverage.  正式结果必须绑定上述全部身份字段。
