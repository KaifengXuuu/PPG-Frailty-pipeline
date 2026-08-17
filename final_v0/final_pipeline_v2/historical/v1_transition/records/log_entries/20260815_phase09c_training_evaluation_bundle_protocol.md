# Phase 09c — training, evaluation, OOF and bundle protocol / 训练评估与部署协议

- Status / 状态：implemented; the isolated training suite passes 28/28 tests.
- Scope / 范围：training implementation, training tests, and this immutable phase entry;
  signal, artifact, feature, data and CLI implementations were not changed by this phase.
- Process / 流程：aggregation identity was expanded before any filtering; rejected rows
  remain explicit coverage denominators; final fits require the exact frozen train roster;
  optional content binding detects data hidden under false participant IDs.
- Algorithm / 算法：inner epoch selection now applies equal-weight
  window→file→role→participant aggregation. Evaluation reports per-class metrics,
  worst-class metrics, multiclass Brier, top-label equal-width ECE, coverage, repeat
  population/sample SD, Student-t 95% CI, and paired repeat/fold/seed deltas.
- OOF / 折外预测：the formal validator checks the exact frozen
  repeat×fold×seed×config×participant Cartesian product, complete trace fields, explicit
  rejected rows, and exact ensemble member indices 0..4.
- Bundle / 部署包：metadata is fail-closed against the complete §5.14 schema. State,
  transforms, golden arrays and the optional raw-record adapter are hashed in a
  same-filesystem staging directory; golden parity is checked before one atomic directory
  rename exposes the target. Load rejects stale expected schemas and unverified entries.
- Result / 结果：28/28 training tests pass, including identity-mix, held-out mutation,
  exact-roster, drop coverage, metric formula, failed-staging cleanup, raw adapter,
  stale-schema, and 10,000 lightweight load/predict rounds without repeated save.
- 中文说明：本阶段关闭了 §5.12–§5.14 的主要协议缺口；任何 drop 均不再从
  coverage 分母中消失，任何不同实验身份均不得混合，正式 OOF 与 bundle 均采用
  显式完整性门禁。本结果是软件协议测试，不是 29-subject 科学 benchmark。
