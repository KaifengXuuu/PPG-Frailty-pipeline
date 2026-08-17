# Phase 09a — executable baseline regression gate / 可执行基线回归门

- Status / 状态：implemented; validation pending the next full-suite run.
- Scope / 范围：only `final_pipeline_v1/tests/audit/`; no historical source was changed.
- Process / 流程：the test reloads both audit JSON files, streams every registered root
  source as bytes, recomputes size/SHA-256, checks the 261-record/29-participant cohort,
  and verifies all historical scores remain explicitly ineligible for V1 ranking.
- Algorithm / 算法：content identity uses SHA-256 over ordered byte blocks; scientific
  eligibility is a fail-closed Boolean invariant rather than a score threshold.
- Result / 结果：four deterministic tests were added.  They will be included in the
  final all-suite report; no historical metric was reinterpreted as a V1 result.
- 中文说明：新增四项确定性测试，确保历史源码指纹、M2 roster、模型参数快照和
  历史结果 `eligible=false` 均不可静默漂移；本批次没有改动根目录历史代码。
