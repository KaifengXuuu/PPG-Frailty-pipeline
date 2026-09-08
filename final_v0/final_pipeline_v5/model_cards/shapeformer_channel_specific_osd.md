# ShapeFormerChannelSpecificOSD

- Machine ID / 机器 ID：`shapeformer_channel_specific_osd`
- Scientific status / 科学状态：`implemented_not_benchmarked_high_compute`
- Representation mode / 表征：`raw`
- Eligible signal routes / 可用信号路线：`direct_x_filter`, `identity_direct`
- Evaluation unit / 评估单位：participant after window→file, then config-dependent Line A equal-files or Line B equal-role-families aggregation
- Current classifier role scope / 当前分类 role 范围：SQI off; only B and R are admitted
- Execution status / 执行状态：registered/constructible; scientific benchmark not run
- Independent test / 独立测试：absent; `independent_test=false`

## Identity and deviation / 身份与偏离

Fold-local channel-specific OSD/PISD uses floor(0.20*T) PIPs from actual T (minimum five), the upstream z-scored time-index perpendicular-distance PIP selector, exhaustive insertion-stage variable candidates bounded by three consecutive PIPs, and no fixed candidate length/stride/cap. Discovery uses the upstream PCS [start-w+1,end+w) boundary. Candidates are enumerated class/channel/source/candidate, ranked with default NumPy argsort then reverse, and each selected class bank is finally ordered by start sample. Each candidate is ranked with the reviewed upstream target-positive recall 0.2-grid information-gain rule (including its -1 no-grid sentinel). Each ShapeBlock searches raw segments only on its source channel within source start/end +/-128 samples, emits l1(selected)-l2(shapelet), adds channel/start/end embeddings whose widths are observed max+1 and IG weighting, then fuses shape attention (without probability dropout, matching upstream forward) with the full eight-channel generic branch.

中文：本卡只描述已实现接口和命名边界，不把架构存在、单元测试通过或历史分数
解释为 V2 性能证据。

## Limitations / 限制

- No independent frailty test set is available; formal scores must be named oof_validation_*.
- No V2 performance is claimed until the same frozen 5×5 participant protocol is run.
- The implementation is registered/constructible; the scientific benchmark has not been run.
- Three shapelets per class and the participant/file-balanced cap of 180 windows are project capacity controls.
- Every shapelet archives source channel name/index, sample/second endpoints, length, and discovery-window identity.
- The persisted information_gain_split_rule is upstream_positive_recall_grid_0p2; exhaustive all-threshold IG is not the reference.
- Discovery PCS uses start-w+1 while downstream ShapeBlock uses start-w, matching the two distinct upstream implementations.
- Candidate enumeration, default-argsort tie ranking, final start ordering, observed-max+1 embedding widths, and unused attention dropout are persisted architecture identities.
- A PISD failure is explicit and cannot silently select another discovery method.
- At 5 s x 400 Hz, T=2000 and the derived PIP count is 400; exhaustive discovery is intentionally high-compute and has not been benchmarked.

## Required provenance / 必需追溯字段

Every formal result must bind participant/file/role, repeat/fold/seed, config hash,
manifest and fold hashes, preprocessing and feature hashes, signal route, aggregation,
model state, environment, and coverage.  正式结果必须绑定上述全部身份字段。
