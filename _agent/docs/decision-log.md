# Decision Log

## 2026-06-23：`PROJECT_HANDOFF.md` 第 0 节优先于旧内容

- 状态：draft
- 背景：`PROJECT_HANDOFF.md` 旧章节中存在“frailty3 脚本未审阅”等过期描述。
- 决策：若旧内容与第 0 节冲突，以第 0 节为准。
- 决策原因：第 0 节是较新的 confirmed 补充，已纠正旧交接内容。
- 影响范围：`MODULES.md`, `TODO.md`, `ROADMAP.md`, 后续 handoff 归档。

## 2026-06-23：放弃 dynamic clean waveform reconstruction 作为主路线

- 状态：draft
- 背景：旧 denoiser 在 motion 段泛化失败，缺少真实 motion clean PPG 监督。
- 决策：dynamic denoising 标记为 deprecated/experimental，不作为主线。
- 决策原因：A 近似复制输入；B 引入伪峰和相位错误；static 段表现不能证明动态有效。
- 后续追踪：保留 gating/motion state 价值和 ONNX 部署经验。

## 2026-06-23：动态段主线改为 direct heartbeat / IBI extraction

- 状态：draft
- 背景：frailty pipeline 真正需要可靠 HR/HRV，而不是视觉上 clean 的动态 PPG。
- 决策：优先推进 `ppg_peak_hr_gating_train.py`，直接预测 peak、IBI、HR/HRV。
- 决策原因：该路线更贴近最终生理特征需求，也避开 clean waveform 监督不可得问题。
- 后续追踪：正式训练、scorecard、LODO、extra-holdout、delay analysis、ONNX/CPU-only。

## 2026-06-23：frailty3 评估按 subject-level 与 config-level 聚合

- 状态：draft
- 背景：window overlap 和同一 subject 多窗口可能导致泄漏或过度乐观。
- 决策：主 CV 使用 `StratifiedGroupKFold`；leaderboard 按 config group mean/std/CI，而不是单 run。
- 决策原因：避免 subject leakage，减少偶然 repeat 对模型选择的影响。
- 后续追踪：final config 选定后重新训练部署模型，不从 CV fold 中挑最好模型。

## 2026-06-23：PPI/HRV 作为表格特征 fusion，而不是 raw 时序通道

- 状态：draft
- 背景：frailty3 已支持 `extra_input=0/PPI/HRV`。
- 决策：PPI/HRV 经 fold 内标准化后作为 extra tabular features，通过 MLP 与深度特征融合。
- 决策原因：保留手工生理特征的解释性，同时避免直接混入 `[N,8,T]` raw window。
- 后续追踪：是否保留需基于 group-level sweep，而不是单 run。

## 2026-06-23：ASA 只作为旁支实验

- 状态：draft
- 背景：`asa_classifier.py` 是 VitalDB ASA 1/2/3 三分类实验。
- 决策：ASA 不纳入 frailty pipeline 主线结论。
- 决策原因：任务、标签和数据来源均不同，不能直接等同 frailty 分类。
- 后续追踪：在 `MODULES.md` 中标注为 side experiment。
