# Phase 09c — executable physical-time ablation / 可执行物理时间消融

- Status / 状态：implemented and tested.
- Scope / 范围：`models/time_scale.py`, optional CompactCNN/Inception constructor
  controls, and four model tests; default model identities remain unchanged.
- Process / 流程：declared physical kernel durations are converted to nearest odd
  samples for DL-only grids 100/160/200/400 Hz; 5/10 s windows, dilation 1/2 and all
  four representation labels are materialized as 64 auditable candidate conditions.
- Algorithm / 算法：ties in odd-sample conversion choose the larger kernel; the longest
  Inception branch uses `1 + depth*dilation*(max_kernel-1)` samples.  At 400 Hz the
  reference remains 39/19/9 samples and 229 samples = 0.5725 s receptive field.
- Result / 结果：model suite 16/16 passed.  Real CompactCNN and Inception forward calls
  succeeded for seconds-derived kernels with dilation 2; reference parameter counts stayed
  79,139 / 456,579 / 57,027.
- 中文说明：物理时间消融不改变 400 Hz acquisition/feature 网格，也不把 64 个候选
  视为已完成全量训练；正式比较仍须复用相同 folds/seeds 并报告指标、覆盖率、时间和内存。
