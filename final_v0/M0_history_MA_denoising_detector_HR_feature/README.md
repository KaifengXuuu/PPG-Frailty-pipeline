# M0 历史 Motion Artifact、降噪、检测器与动态 HR 归档

## 1. 归档身份

- 用户指定显示名称：`M0_**history\_MA\_denoising\_detector\_HR\_feature**`
- 实际安全目录名：`M0_history_MA_denoising_detector_HR_feature`
- 建立日期：2026-08-03
- TODO 归属：仍为 **M0 扩展收尾**，没有进入 M1
- 写入边界：仅 `final_v0/`
- 源代码、原始数据、历史输出、`AGENTS.md` 与 `_agent/`：只读，未修改
- 外部联网：未使用；TROIKA/JOSS 只做本地代码存在性审计和原理级路线映射，未冒充文献复现

本目录把本轮 M0 的人工可读结论、候选脚本、五类方法审计、可实现接口、测试与 benchmark 契约集中保存。原始扫描得到的大型逐文件 manifest 继续保留在 `final_v0/records/generated/`；本目录通过快照清单、SHA-256 和路径索引引用这些证据，避免复制出第二套可能漂移的超大事实源。

## 2. 结论先行

1. 当前没有任何 motion detector、动态 PPG 去噪或 heartbeat/HR 路线达到 `strictly_validated`。
2. 当前最有希望的 motion 工程候选是 `ppg_peak_hr_gating_train.py` 中独立的 10-channel PPG+IMU A/B benchmark；external SIM 的 Light CNN 为 F1 `0.7634`、BA `0.7802`、AUC `0.8642`，但仍缺 subject-level CI 和真实 PPG artifact 标签。
3. 当前 waveform denoising 均缺真实 clean reference 与独立生理终点；hybrid 只有 proxy validation objective，v7.2 独立 holdout 明确为负，v8/Stage-2 没有产物。
4. 动态 HR 的首选新路线应是“STFT/谱证据抑制 → 候选峰 → Viterbi/Kalman/Particle 轨迹 → SQI/拒绝输出”，而不是继续强求完整波形恢复；该完整路线在仓库中尚未实现。
5. 现有双波长数据足以试验 PCA/FastICA/STFT-NMF，但仓库中的 PCA 仅用于 SVM 特征降维，不是盲源分离；ICA/NMF 均不存在。
6. 当前 SQI 在 frailty generalization sweep 中有轻微正信号：top50 平均 subject BA `0.51079`，none 为 `0.49790`；但现有公式缺用户点名的大部分指标，且存在标准化、P5/P95 跨测试分布和训练/部署不一致风险。
7. 用户已确认后续串行主线：`SQI-v2 → 29-subject Motion threshold/CV → Motion 融入 SQI → 谱域轨迹+SQI → 双波长 BSS → 非平稳分解对照 → 自适应滤波风险对照 → PTT HR/PPI 监督测试 → frailty 路线特征选择`。这是一项路线决定，不表示任何模块已实现或已验证。

## 3. 文件导航

| 文件 | 用途 |
|---|---|
| `01_M0_COMPLETE_RESULTS_AND_DECISIONS.md` | 完整 M0 扫描、路线、数值结果、证据等级、淘汰/保留结论 |
| `02_CANDIDATE_SCRIPTS_MOTION_DENOISING_DYNAMIC_HR.md` | 用户指定的新文档：三个问题的全部有希望脚本、算法、输入、输出、结果、状态和未来方向 |
| `03_FIVE_METHOD_FAMILIES_CODE_TEST_IMPLEMENTATION_AUDIT.md` | 五点逐项源码位置、理论假设、已有测试、缺陷、可实现方案和测试矩阵 |
| `04_UNIFIED_IMPLEMENTATION_AND_BENCHMARK_CONTRACT.md` | 后续实现的统一接口、数据切分、指标、输出 schema、对照和验收门 |
| `05_EVIDENCE_INDEX_AND_PROVENANCE.md` | 源代码、输入、输出、机器证据及快照的可追溯索引 |
| `06_M0_PACKAGE_TREE.md` | 保留的 v1 历史包树；不因本次路线更新而覆盖 |
| `07_CONFIRMED_MADENOISER_FOLLOWUP_ROADMAP.md` | 用户确认的 MAdenoiser 后续顺序、29 人 Motion/SQI、PTT HR/PPI 与 frailty 5-fold 选择合同 |
| `08_M0_PACKAGE_TREE_V2.md` | 当前追加式 v2 包的完整树、大小、SHA-256 和内容说明 |
| `M0_SOURCE_SNAPSHOT_MANIFEST.json` | 保留的 v1 快照清单 |
| `M0_PACKAGE_VERIFICATION.json` | 保留的 v1 包验证 |
| `M0_SOURCE_SNAPSHOT_MANIFEST_V2.json` | 本次路线的追加式 v2 快照源、目标、字节与 SHA-256 |
| `M0_PACKAGE_VERIFICATION_V2.json` | 本次路线 v2 的必需文档、快照与算法图验证 |
| `snapshots/records/` | M0 核心报告、registry、crosswalk、风险和扫描说明的只读快照 |
| `snapshots/algorithm_diagrams/` | M0 算法图快照 |
| `snapshots/verification/` | 精选机器验证与输入/输出 summary；大型逐文件 manifest 由索引指向原位 |

## 4. 状态词汇

| 状态 | 定义 |
|---|---|
| `strictly_validated` | subject-disjoint 或 external 协议、无评价集调参、核心生理指标和可复现 runtime 均通过 |
| `implemented_unverified` | 有可调用实现，但证据不足或协议不完整 |
| `promising_but_not_implemented` | 数据与设计条件具备，仓库尚无完整实现 |
| `smoke_only` | 仅证明能运行/画图/导出，不证明效果 |
| `failed_or_deprecated` | 已有负结果、确定性阻断、泄漏或目标不成立，不应作为主线恢复 |
| `baseline_only` | 只保留作对照、风险证明或工程消融 |

## 5. 使用规则

1. 每个数值必须同时带脚本、真实输出目录、split、目标定义和证据等级。
2. `activity detector` 不得写成 `PPG artifact ground-truth detector`。
3. ECG R-peak timing 未做 ECG→PPG delay 校正时，不得写成 PPG pulse-peak accuracy。
4. 模型、ONNX、图片或目录存在，不等于有效性验证完成。
5. 所有未来路线必须与 raw/no-denoising、bandpass 和 high-quality-only/SQI 基线比较，并报告 coverage/拒绝率。
6. 本归档是 M0 历史快照；刷新 `snapshots/` 必须有用户明确确认，不能在后续 TODO 中静默改写历史结论。

## 6. 下一人工决策门

开始实现前的当前首要阻塞项，是明确“29-subject Motion detector”的监督目标：窗口级光学伪影人工标签、`B/R/S/W` 活动代理，或独立定义的 peak/HR 不可用性标签。三者会改变标签构造、阈值解释、CV 指标和论文表述，不能由 agent 擅自代选。

本目录完成的是本地代码与结果审计、可实现方案和测试合同，不包含外部论文下载。若后续要把 TROIKA/JOSS 写成精确、带原始论文引用的复现规范，仍需用户另行确认允许外部联网并指定是否只读原始论文/官方来源。
