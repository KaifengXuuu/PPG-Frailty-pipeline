# Decision Log

## 2026-07-26：`_agent` 任务更新采用增补、合并和状态迁移

- 状态：confirmed
- 背景：新增研究想法与旧 TODO 有重叠，直接追加会重复，直接覆盖会丢失历史。
- 决策：新方向作为 P0/P1 增补；语义相同的旧任务合并到新任务；
  已完成或被后续实验替代的事项迁移到对应状态区，不删除历史事实。
- 决策原因：同时保持活动文档可执行和更新过程可追溯。
- 影响范围：不覆盖 archive handoff；不把 planned 文件写成 implemented；
  修改旧结论时保留原日期并增加后续修正。

## 2026-07-26：先统一 Benchmark，再选择最终 Frailty3 模型

- 状态：confirmed
- 背景：历史实验混有 holdout、early stopping、fixed-epoch CV 和 data leakage，
  同名 reference 的稳定性也未完全复现。
- 决策：优先建立统一 manifest、fold registry、protocol registry、指标和输出格式；
  只在严格可比协议中按 config-level repeat aggregation 选择 Top 5。
- 决策原因：避免把 split、seed、命名或协议差异误判为模型改进。
- 影响范围：`frailty_3class_classifier.py`,
  `frailty_3class_overfitting_sweep.py`, `frailty_3class_holdout_eval.py`,
  `analyze_sweep.py` 和计划中的 benchmark/meta-analysis 脚本。

## 2026-07-26：当前主比较协议为 5-fold Fixed-Epoch Subject CV

- 状态：confirmed
- 背景：subject 数较少，用户要求不使用 early stopping，并要求公平比较。
- 决策：当前模型选择实验统一使用 5-fold `StratifiedGroupKFold`、固定 epoch、
  no early stopping、相同 folds/seeds/repeats。
- 决策原因：同一 subject 不跨 fold，且不为每个 fold 引入不同的
  inner-validation selection noise。
- 影响范围：当前 CV 没有额外独立 test；fold 汇总必须称为 OOF validation。
  holdout/early-stopping 历史结果保留，但不得与本协议直接排名。

## 2026-07-26：Frailty3 活动流程保持 400 Hz 且 Raw Data 只读

- 状态：confirmed
- 背景：需要防止重采样改变 peak timing/形态，并防止训练脚本修改原始数据。
- 决策：活动 frailty3 pipeline 全程使用原始 400 Hz，不执行 resampling；
  `PPG_Testing_05_01_2026/` 与 `physionet.org/` 保持只读。
- 决策原因：统一时间基线并保护数据来源。
- 影响范围：`datasets/` 只作为生成/读取 cache；archive 中旧 resample 代码
  不代表当前算法。

## 2026-07-26：新增两条高优先级候选路线

- 状态：confirmed
- 背景：flat InceptionTime 在多个 sweep 中仍明显过拟合，BA 未达到 0.73。
- 决策：在同一 benchmark 中优先验证：
  1. Young-vs-Old、再 Pre-Frail-vs-Robust 的 hierarchical InceptionTime；
  2. Base/Motion/Relax 的 HR/HRV/IMU/recovery features 加弱模型。
- 决策原因：前者重新表达标签结构，后者减少 raw deep model 的样本复杂度并增强解释性。
- 影响范围：两条路线都必须使用 subject-level split，并进入统一消融；
  目前只是 approved plan，不得记录为已实现。

## 2026-07-26：SQI Gating 必须同时报告 Coverage

- 状态：confirmed
- 背景：SQI 不使用 class label，但会丢弃或降低低质量 windows 的权重。
- 决策：允许把 SQI 作为质量控制因素，但每次实验必须同时报告每 class/subject
  的保留窗口数，并与 no-gating 做 paired comparison。
- 决策原因：防止 coverage 改变被误解为纯模型泛化提升。
- 影响范围：训练采样、subject aggregation、报告和消融。

## 2026-07-26：历史 Grid 的参数统计只解释为关联

- 状态：confirmed
- 背景：历史 grids 不平衡，参数组合和实验协议存在混杂。
- 决策：参数均值、回归系数和交互图只能表述为描述性关联；
  因果主效应必须由同 folds/seeds、单因素变化的消融支持。
- 决策原因：避免从非正交实验设计中得出过强结论。
- 影响范围：跨 sweep 报告、论文文字和下一轮实验设计。

## 2026-06-23：`PROJECT_HANDOFF.md` 第 0 节优先于旧内容

- 状态：confirmed
- 背景：`PROJECT_HANDOFF.md` 旧章节中存在“frailty3 脚本未审阅”等过期描述。
- 决策：若旧内容与第 0 节冲突，以第 0 节为准。
- 决策原因：第 0 节是较新的 confirmed 补充，已纠正旧交接内容。
- 影响范围：`MODULES.md`, `TODO.md`, `ROADMAP.md`, 后续 handoff 归档。

## 2026-06-23：放弃 dynamic clean waveform reconstruction 作为主路线

- 状态：confirmed
- 背景：旧 denoiser 在 motion 段泛化失败，缺少真实 motion clean PPG 监督。
- 决策：dynamic denoising 标记为 deprecated/experimental，不作为主线。
- 决策原因：A 近似复制输入；B 引入伪峰和相位错误；static 段表现不能证明动态有效。
- 后续追踪：保留 gating/motion state 价值和 ONNX 部署经验。

## 2026-06-23：动态段主线改为 direct heartbeat / IBI extraction

- 状态：confirmed
- 背景：frailty pipeline 真正需要可靠 HR/HRV，而不是视觉上 clean 的动态 PPG。
- 决策：优先推进 `ppg_peak_hr_gating_train.py`，直接预测 peak、IBI、HR/HRV。
- 决策原因：该路线更贴近最终生理特征需求，也避开 clean waveform 监督不可得问题。
- 后续追踪：正式训练、scorecard、LODO、extra-holdout、delay analysis、ONNX/CPU-only。

## 2026-06-23：frailty3 评估按 subject-level 与 config-level 聚合

- 状态：confirmed
- 背景：window overlap 和同一 subject 多窗口可能导致泄漏或过度乐观。
- 决策：主 CV 使用 `StratifiedGroupKFold`；leaderboard 按 config group mean/std/CI，而不是单 run。
- 决策原因：避免 subject leakage，减少偶然 repeat 对模型选择的影响。
- 后续追踪：final config 选定后重新训练部署模型，不从 CV fold 中挑最好模型。

## 2026-06-23：PPI/HRV 作为表格特征 fusion，而不是 raw 时序通道

- 状态：confirmed
- 背景：frailty3 已支持 `extra_input=0/PPI/HRV`。
- 决策：PPI/HRV 经 fold 内标准化后作为 extra tabular features，通过 MLP 与深度特征融合。
- 决策原因：保留手工生理特征的解释性，同时避免直接混入 `[N,8,T]` raw window。
- 后续追踪：是否保留需基于 group-level sweep，而不是单 run。

## 2026-06-23：ASA 只作为旁支实验

- 状态：confirmed
- 背景：`asa_classifier.py` 是 VitalDB ASA 1/2/3 三分类实验。
- 决策：ASA 不纳入 frailty pipeline 主线结论。
- 决策原因：任务、标签和数据来源均不同，不能直接等同 frailty 分类。
- 后续追踪：在 `MODULES.md` 中标注为 side experiment。
