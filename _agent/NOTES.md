# NOTES

状态：confirmed（含明确标注的观察、风险和待验证假设）
来源：用户评价、`_agent/arc/PROJECT_HANDOFF.md`、代码与结果复核
最后手动更新时间：2026-07-26

## 2026-07-26：Frailty3 协议与结果解释边界

- 类型：风险/已核准事实
- 涉及文件/模块：`frailty_3class_overfitting_sweep.py`, `analyze_sweep.py`
- 内容摘要：当前主协议是 subject-level 5-fold `StratifiedGroupKFold`、
  fixed epoch、no early stopping。CV 之外没有额外独立 test set；5 folds 合并后是
  OOF validation。历史报告中的部分 `test_*` 字段实际代表 OOF validation。
- 证据或线索：当前训练/汇总代码和 2026-06-25、2026-06-30 reports。
- 可能下一步：统一命名为 `oof_validation_*`，并用 protocol registry 阻止
  holdout、early-stopping CV、fixed-epoch CV 和泄漏实验进入同一排名。

## 2026-07-26：当前最佳分数仍不稳定

- 类型：观察/风险
- 涉及文件/模块：`results_frailty3/_overfitting_sweep/`
- 内容摘要：同一 nominal `s1_122` 在 2026-06-25 为
  BA `0.610 +/- 0.061`，在 2026-06-30 reference rerun 为
  `0.623 +/- 0.014`。平均分相近，但 repeat stability 未复现；现有
  train-validation BA gap 约 0.46，过拟合仍明显。
- 待验证问题：差异来自 folds/seeds、reference 参数解析、window realization，
  还是训练非确定性。
- 可能下一步：保存并复用 fold registry，做 paired-seed rerun 和 prediction-level diff。

## 2026-07-26：最新 class bottleneck 修正

- 类型：对旧观察的修正
- 涉及文件/模块：`analyze_sweep.py`, 2026-06-30 generalization sweep
- 内容摘要：旧记录强调 Pre-Frail vs Robust 是唯一主要瓶颈；最新最佳 reference
  aggregate confusion 中三类 recall 约为 Pre-Frail 77.8%、Robust 56.7%、
  Young 52.5%，Young 反而最低。因此后续必须同时报告三类结果，不能预设 Young 易分。
- 可能下一步：按 subject、role、SQI 和窗口数检查 Young -> Robust 的误分来源。

## 2026-07-26：SQI gating 的解释风险

- 类型：方法风险
- 涉及文件/模块：`frailty_3class_classifier.py`,
  `frailty_3class_overfitting_sweep.py`
- 内容摘要：SQI 是 label-independent quality filter；它在 training 中删除
  file 内低质量 windows，在 evaluation/aggregation 中选择高质量 windows 或加权。
  这不等于人为挑对样本，但会改变每个 subject/class 的评估 coverage。
- 待验证问题：低 SQI 是否与 class、age、motion 或 recording role 系统相关。
- 可能下一步：所有 SQI 实验同步报告每类/每 subject 保留窗口数，并做
  no-gating paired comparison。

## 2026-07-26：Hierarchical classifier 的错误传播

- 类型：待验证假设/风险
- 涉及文件/模块：计划中的 `frailty_3class_hierarchical.py`
- 内容摘要：Young-vs-Old 上层可能简化两个老年类的下层边界，但上层把 old
  错分为 Young 后，下层无法修正。hierarchy 不是天然优于 flat three-class。
- 可能下一步：同时报告 top-level、old-only oracle bottom-level 和 end-to-end
  three-class metrics，并使用概率乘法而不是硬路由作为初版。

## 2026-07-26：外部 ECG supervision 的 domain shift

- 类型：风险
- 涉及文件/模块：`ppg_peak_hr_gating_train.py`
- 内容摘要：开放同步 ECG/PPG 数据可监督 motion peak/PPI/HR，但设备波长、
  placement、activity、subject 和 sampling pipeline 与 frailty 数据不同。
  外部 benchmark 成功不等于目标域 peak extraction 已解决。
- 可能下一步：严格 subject/dataset split、LODO、per-activity scorecard，并在
  frailty data 的高质量段做无 ECG consistency checks。

## 2026-07-26：跨 Sweep 参数“影响系数”不能直接当因果效应

- 类型：统计风险
- 涉及文件/模块：计划中的 `analyze_all_frailty_experiments.py`
- 内容摘要：历史 grids 不平衡，部分参数只与特定 protocol/model/config 同时出现。
  描述性均值或普通回归系数会混入交互和 selection bias。
- 可能下一步：只在可比子集中做 paired contrasts/standardized associations，
  明确写“关联”而不是“因果主效应”；必要的主效应用专门消融验证。

## 2026-07-26：异步人工特征融合风险

- 类型：数据泄漏/隐式加权风险
- 涉及文件/模块：`frailty_3class_classifier.py`
- 内容摘要：当前 file-level PPI/HRV/morphology 可被复制到多个 windows。
  这会按 window 数隐式重复同一向量；若推理目标是实时 window，还可能使用
  当前 window 之后的整文件信息。
- 可能下一步：明确 prediction unit 和时间范围，比较 file/subject late fusion
  与严格 OOF stacking。

## 2026-07-26：Scaler 可能删除 morphology 信息

- 类型：待验证风险
- 涉及文件/模块：frailty3 raw window preprocessing
- 内容摘要：每个 window 单独 median/IQR scaling 有利于稳健训练，但可能删除
  pulse amplitude、IR/Red amplitude ratio、stage recovery amplitude 等信号。
- 可能下一步：在相同 folds/seeds 下做 amplitude-preserving scaler ablation。

## 2026-06-23：关键观察与风险

- 类型：用户偏好
- 内容摘要：用户优先关注成功率、可解释性、泛化能力和持续迭代能力；重要实验需要 scorecard、图表、分层评估和可复现输出。
- 涉及文件/模块：全部实验模块
- 可能下一步：所有主线实验输出应至少包含 summary CSV、confusion matrix、learning curve、scorecard 或 markdown report。

## 2026-06-23：dynamic denoising 失败原因

- 类型：观察/风险
- 涉及文件/模块：`pttppg_denoiser_hybrid_*`, `pttppg_stage2_denoiser.py`
- 内容摘要：Denoiser A 接近复制 raw PPG；Denoiser B 出现双倍伪峰、峰谷错位和非生理形态；static/sit 段表现好不能证明 motion 段去噪有效。
- 证据或线索：`PROJECT_HANDOFF.md` 第 7.1 节和用户明确评价。
- 待验证问题：是否仍值得把 coarse-denoised dynamic signal 作为 frailty classifier 探索输入。
- 可能下一步：仅作为 high-risk exploratory input，不作为可靠去噪模块。

## 2026-06-23：frailty3 当前风险

- 类型：风险
- 涉及文件/模块：`frailty_3class_classifier.py`, `analyze_sweep.py`, `frailty_3class_holdout_eval.py`
- 内容摘要：当时结果的主要瓶颈是 Pre-Frail 与 Robust/Non-Frail 混淆；
  subject 数少导致 fold/repeat 方差大；overlap windows 会增加样本相关性。
- 后续修正：2026-06-30 最佳 reference 中 Young recall 最低，见 2026-07-26 记录。
- 可能下一步：按 subject、role、静/动态状态、HRV、窗口质量分层分析错误。

## 2026-06-23：数据规范风险

- 类型：待确认
- 涉及文件/模块：`funcs.py`, `frailty_3class_classifier.py`, `ppg_analyse4_calib.ipynb`
- 内容摘要：采样率、时间戳、重采样策略必须明确，否则会影响 peak timing、HRV 和 window 模型。
- 后续核准：当前活动 frailty3 流程保持原始 400 Hz、无 resampling；其他数据源
  的时间轴和设备量纲仍需分别审计。
- 可能下一步：建立统一 manifest，注明原始采样率、有效时间轴和处理规则。
