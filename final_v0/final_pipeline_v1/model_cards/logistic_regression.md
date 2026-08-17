# LogisticRegressionL2

- Machine ID / 机器 ID：`logistic_regression`
- Scientific status / 科学状态：`reference_feature_baseline`
- Representation mode / 表征：`feature_vector`
- Eligible signal routes / 可用信号路线：`direct_x_filter`, `identity_direct`, `non_identity_x_ar_rate_only`
- Evaluation unit / 评估单位：participant after window→file→role-aware aggregation
- Independent test / 独立测试：absent; `independent_test=false`

## Identity and deviation / 身份与偏离

Fold-local imputation/scaling precede an L2 logistic model.

中文：本卡只描述已实现接口和命名边界，不把架构存在、单元测试通过或历史分数
解释为 corrected V1 性能证据。

## Limitations / 限制

- No independent frailty test set is available; formal scores must be named oof_validation_*.
- No V1 performance is claimed until the same frozen 5×5 participant protocol is run.
- Only the frozen allowlist and explicit validity fields are eligible.

## Required provenance / 必需追溯字段

Every formal result must bind participant/file/role, repeat/fold/seed, config hash,
manifest and fold hashes, preprocessing and feature hashes, signal route, aggregation,
model state, environment, and coverage.  正式结果必须绑定上述全部身份字段。
