# ROCKET

- Machine ID / 机器 ID：`rocket_numpy`
- Scientific status / 科学状态：`self_contained_project_rocket`
- Representation mode / 表征：`feature_matrix`
- Eligible signal routes / 可用信号路线：`direct_x_filter`, `identity_direct`
- Evaluation unit / 评估单位：participant after window→file→role-aware aggregation
- Independent test / 独立测试：absent; `independent_test=false`

## Identity and deviation / 身份与偏离

A deterministic NumPy/SciPy project implementation; not an aeon/sktime parity claim.

中文：本卡只描述已实现接口和命名边界，不把架构存在、单元测试通过或历史分数
解释为 corrected V1 性能证据。

## Limitations / 限制

- No independent frailty test set is available; formal scores must be named oof_validation_*.
- No V1 performance is claimed until the same frozen 5×5 participant protocol is run.
- Primary formal configuration requires 10,000 kernels and fold-local ridge.

## Required provenance / 必需追溯字段

Every formal result must bind participant/file/role, repeat/fold/seed, config hash,
manifest and fold hashes, preprocessing and feature hashes, signal route, aggregation,
model state, environment, and coverage.  正式结果必须绑定上述全部身份字段。
