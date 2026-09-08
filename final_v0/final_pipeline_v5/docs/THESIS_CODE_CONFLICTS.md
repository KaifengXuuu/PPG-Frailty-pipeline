# 论文与代码实现冲突

只读论文来源为仓库根目录 `Kaifeng_Masterarbeit_draft_v1_0.docx`。本文按对预测、
统计结论或现实推理的潜在影响从大到小排序。V5 对冲突的统一处理是：当前运行继续
遵循已冻结的 V2 数值语义；需要修改数学计算、模型架构、采样率、split 或 workflow
时，先建立明确的新方案并获得用户决定，不用文档解释静默改变代码。

`finalcase` 是用户选定的 Rank 2 `tuned_all_roles_small_no_gravity`。截至当前，V5
尚未完成正式 5×5 finalcase，因此这里只说明执行合同，不能声称 25-fold 已与 V2
数值一致。等价判据是相同输入/split/GPU/CUDA/PyTorch/依赖、离散值精确相同、
浮点 `atol=1e-6, rtol=0`。

## 1. 最终实验身份不唯一

**影响：关键；会同时改变训练数据、模型和优化。**

论文 draft 的一般 baseline 是 B/R roles、400 Hz、CompactCNN、Adam/batch 64；
Appendix E 的最终候选接近 all-role、64 Hz、Small Inception、AdamW/batch 16。它们
不是同一个实验，结果不能互称复现。

V5 保留具名 `baseline`，并把用户选择的 Rank 2 单独命名为 `finalcase`。运行必须
显式选择 preset、完整 YAML 或纯 CLI；正式 study 使用
`configs/studies/finalcase.yaml`。论文结果、代码输出和模型文件必须记录 case ID，
不能只写“final model”。

## 2. Rank 2 的重力处理不同于 draft 的首选候选

**影响：关键；直接改变三个加速度通道和模型输入分布。**

论文排名首位候选使用 Profile A 重力估计/移除；用户选定的 Rank 2 使用
`sensor_filter_only_no_gravity_removal`。Rank 2 仍做传感器低通、同 participant 静态
校准和 SI 单位转换，只是不减去估计重力。

V5 的 `finalcase` 原样实现 Rank 2，不把它改回 Rank 1。两者应作为不同 comparison
cases 报告，不能把重力差异当成展示参数。

## 3. IMU 单位、静态校准和滤波描述不一致

**影响：高；改变幅值、动态加速度、jerk 和所有相关特征。**

论文一般方法文字可读作保留 g/deg/s、部分 gravity profile 不需要轴校准，并写出与
实现不同的 gravity filter 阶数。V2 相关路径实际使用同 participant B-role 5–100 s
静态校准/bias removal，随后转为 m/s² 和 rad/s；Profile A 使用实现中冻结的滤波
合同。

V5 沿用 V2 行为。改单位、校准窗口或阶数会使既有权重、特征和比较失效，必须另建
ablation。

## 4. participant 聚合 estimand 不一致

**影响：高；改变概率、BA/F1/AUC、混淆矩阵和 P 值。**

论文一处描述把 participant 的全部 windows 直接平均，这会让长 recording 权重更大；
V2 最终 Line B 是 `window → file → role family → participant`，每层普通均值，可用
role families 等权。

`finalcase` 固定 Line B。V5 同时保存每 fold 的 window/file/role/participant 概率，
使其他聚合或不同统计单位可以在训练后研究，但这些结果是新的 post-hoc estimand，
不能覆盖声明主结果。

## 5. 深度输入采样与尾窗规则不一致

**影响：高；改变序列长度、窗口数、边界样本和实际感受野。**

论文一般 workflow 容易被理解为全程 400 Hz，并提到 short-record zero padding；
Appendix E/V2 final 候选实际在 canonical 400 Hz 预处理后，以 polyphase anti-alias
转为 64 Hz，使用 5 s 窗、2.5 s hop、distinct right-aligned 完整尾窗、无 padding、
每 file 最多 128 窗。

V5 按每个 resolved config 执行；仅 finalcase 使用上述 64 Hz 合同，不能把它写成所有
模块的全局默认。

## 6. outer-CV 权重与部署 refit 容易混淆

**影响：高；不会改变 OOF 值，但会改变模型可代表的训练总体和可作出的结论。**

论文性能来自 participant-grouped outer OOF，而现场演示需要一份可加载权重。任意
fold 模型只见过该 fold 的 outer-training participants；从 folds 中按 OOF 指标选权重
本身也利用了 held-out 表现，因此不能叫无偏“最终部署模型”。

V5 保存全部 fold bundles，并默认发布按
`(balanced_accuracy, repeat, fold)` 排序的中位 fold，标作 research/Dash trial。
runtime `refit` 默认 false；仅加 `--refit` 时，outer folds 之后对该 run 的**每个
case**执行现有 all-29 refit。配置中的 refit-related 训练字段不单独触发这一步。

all-29 权重没有内部 self-evaluation，仍只能引用 refit 前 outer OOF 的性能。中位
fold 和 all-29 bundle 必须在论文/演示中清楚区分。

## 7. 现实动态输入可能缺少静态 B 校准

**影响：高但部署局部；缺失策略会系统性改变 IMU 分布。**

训练 workflow 的动态 R/S/W recording 依赖同 participant 静态 B recording。现实
使用可能只得到动态记录；静默用零 bias、其他 participant 或近似重力都不等价于
训练预处理。

当前 CLI/Dash inference 要求动态输入同时提供同 participant B；B-only 输入可独立
处理。missing-B 静默校准是 **V5 TODO**，以后必须以单独 ablation 测试，当前不会
作为隐式 fallback。

## 8. 历史 ShapeFormer epoch selection 使用 outer-held-out 信息

**影响：高但只影响历史模块；会产生 optimistic bias。**

论文已承认部分历史 ShapeFormer early stopping 参考 outer-held-out fold。这与
fixed-epoch 或 outer-train-only inner grouped selection 不可公平比较。

V5 保留 faithful/history 与修正模块用于复查，但不会重写它们的数学算法；报告必须
标明 leakage 状态，历史结果不得默认进入正式无泄漏 leaderboard。

## 9. PRV eligibility 门槛不一致

**影响：中高；改变 feature 值、missingness 和 feature-model 输入。**

论文正文简写 spectral PRV 为至少 60 s/少量 intervals，表格又给出 5 min 和 SampEn
200 intervals。V2 区分 rate、time、spectral、nonlinear：time 至少 60 s、30
intervals、coverage 0.8；spectral 至少 300 s、200 intervals、coverage 0.8；
SampEn 至少 200 intervals。

V5 保留这些分层门槛。finalcase 是 raw 模型，冲突主要影响并联 feature 分析与诊断，
而非其主输入张量。

## 10. PTT 数据版本、采样率和用途表述不一致

**影响：中；主要影响外部 motion/peak 分支。**

论文写 PTT-PPG v1.0.0/500 Hz；V2 使用本地 1.1 数据，并在项目处理合同中规范到
400 Hz，同时引用 v1.0 页面作为波长/单位证据。PTT 用于 beat/motion 开发和
transfer，不是独立 frailty test cohort。

V5 保留 V2 的版本、单位和 resampling 语义。报告不得把 PTT transfer 标成独立
frailty test。

## 11. 默认 peak detector 的文字残留

**影响：中；改变 peak/PPI、PRV、morphology 与 SQI。**

论文 overall workflow 一处仍写 Aboy++，结果和 Appendix E 则采用 MSPTDfast v2.3。
V2/V5 finalcase 使用 MSPTDfast Python peak-only port；Aboy/project variants 继续
作为显式历史/ablation 模块。失败时不静默换 detector。

## 12. bootstrap、permutation 次数和检验适用性不一致

**影响：中低；预测不变，但 CI/P 值精度和可报告结论改变。**

论文概述 participant-cluster permutation 为 10,000 次。V2/V5 声明配置使用
participant bootstrap 10,000、paired permutation 100,000，并在 comparison family
内 Holm 校正；部分 ROC-AUC P 值因前提不足为 N/A。

report 必须记录 resamples、seed、cluster/exchange unit 和 multiplicity family。基于
保存的 file/fold 预测另做不同单位检验时，应作为独立分析记录。

## 13. feature-matrix 维度存在历史漂移

**影响：低至中；错误采用旧维度会造成模型输入不兼容。**

旧材料曾写 `115×150`；当前代码/config 与论文后段使用
`146×variable-K`，仅 batch 内 padding，并用 mask 排除 padding。V5 以执行配置和
registry 为准，不迁移旧维度描述。

## 14. motion transfer 与“独立测试”措辞混淆

**影响：数值低、结论风险高。**

PTT22 内部 OOF 与 PTT→Frailty transfer 是两类证据；内部 frailty 数据目前只有
participant outer OOF，没有 roster-disjoint independent frailty test。report 的
`test` mode 只接受明确独立 test evidence，不能把 outer OOF 改名当 test。

## 15. V2 将训练与展示耦合

**影响：科学数值低、维护和算力影响高。**

V2 部分入口在训练后立即生成 plots/HTML，重复 comparison 会重复展示开销。V5 的
pipeline 只写数据、pipeline Excel 和 weights；`analyse_report.py` 从已有结果任意
组合通用 classification figures/tables，并为 specialized 分支生成各自注册的完整
套件、HTML 和 report Excel。这个结构变化不应改变训练或预测，且允许同一 run
重复分析而不重训。

## 论文定位

- 3.1 Overall workflow：Word 页 42
- 3.2 Datasets：页 46–47
- 3.3 Signal preparation：页 48–59
- 3.4 Feature engineering：页 59–70
- 3.5 Models：页 70–77
- 3.6 Evaluation：页 77–80
- 4.1 Dynamic experiments：页 81 起
- 4.2 Frailty classification：页 88 起
- Appendix E：页 141 起
