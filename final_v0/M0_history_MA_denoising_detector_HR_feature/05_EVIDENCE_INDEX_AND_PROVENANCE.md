# M0 证据索引与来源链

## 1. 证据优先级

冲突时按以下顺序处理：

1. 当前源代码的实际执行分支、参数与写出语句；
2. 实际存在的 JSON/CSV/Markdown 文本结果；
3. 二进制 artifact 的文件名、大小和伴随 meta；
4. 输入文件 header/schema；
5. 历史 `_agent` 记录与 notebook 输出；
6. 注释、README 或预期输出结构。

目录存在、文件名、模型/ONNX 存在只能证明产物存在，不能替代 split、target、metric 和有效性验证。

## 2. 本归档的直接来源

### 2.1 核心 M0 报告

| 权威源 | 内容 |
|---|---|
| `final_v0/records/M0_EXECUTIVE_REPORT.md` | M0 完整流程、扫描统计、路线结论和验收映射 |
| `final_v0/records/M0_METHOD_REGISTRY.md` | F01–P02 共 17 路方法的实现、参数、结果和状态 |
| `final_v0/records/M0_CODE_OUTPUT_CROSSWALK.md` | 代码声明路径、实际输出、schema 与对应状态 |
| `final_v0/records/M0_PAPER_EVIDENCE.md` | 定量结果、证据等级与论文表述边界 |
| `final_v0/records/M0_RISK_REGISTER.md` | 泄漏、运行错误、目标与部署风险 |
| `final_v0/records/M0_ARCHIVED_LINEAGE_EVIDENCE.md` | 归档代码/历史版本的谱系与证据 |
| `final_v0/records/PROJECT_WIDE_SCAN_FINDINGS.md` | workspace 范围、输入/输出统计和结构发现 |
| `final_v0/records/CODE_IO_MASTER_INDEX.md` | 52 份代码/notebook 的输入输出主索引 |
| `final_v0/records/ROOT_FILE_IO_INVENTORY.md` | 根目录逐文件角色、路径与结果摘要 |
| `final_v0/records/ARCHIVED_CODE_IO_INVENTORY.md` | Arc/notebook 的算法和 I/O 清单 |

本目录 `snapshots/records/` 保存上述核心文件的 M0 历史快照；`M0_SOURCE_SNAPSHOT_MANIFEST.json` 记录源/快照 SHA-256 相等性。

### 2.2 算法图

- `final_v0/algorithm_diagrams/00_PROJECT_HISTORICAL_SIGNAL_FLOW.md`
- `final_v0/algorithm_diagrams/m0/01_FOUNDATION_FUNCS_PPG.md`
- `final_v0/algorithm_diagrams/m0/02_V7_TO_STAGE2_EVOLUTION.md`
- `final_v0/algorithm_diagrams/m0/03_HYBRID_SUITE.md`
- `final_v0/algorithm_diagrams/m0/04_HEARTBEAT_AND_MOTION_AB.md`
- `final_v0/algorithm_diagrams/m0/05_SCRIPT_ALGORITHM_ATLAS.md`
- `final_v0/algorithm_diagrams/m0/06_FIVE_METHOD_FAMILIES_IMPLEMENTATION_AND_BENCHMARK.md`

## 3. 关键源码证据

| 主题 | 脚本与行段 | 事实 |
|---|---|---|
| Savitzky–Golay误名 | `funcs.py:45-50`, `ppg.py:541-546` | `wavelet_denoise` 不是小波 |
| IMU-NLMS | `funcs.py:959-1066`, `ppg.py:1468-1575` | 6维memoryless NLMS、motion gate、HR后处理 |
| EMD/CEEMD-lite | `funcs.py:1139-1387`, `ppg.py:1703-1900` | 手写sifting、同PPG reference、32-tap leaky NLMS |
| DWT-A2 | `pttppg_pipeline_v7.py:97-106`, `cnnppg_v7.py:146-164`, v7.4 `:306-322` | db4 level2 approximation + interpolation |
| STFT MaskNet | v7.4 `:887-917,:1004-1117` | magnitude mask、noisy phase、广播特征 |
| v8阻断 | `pttppg_denoiser_v8_masknet.py:214-255` | time mask/`F`覆盖/无效频率平滑 |
| Stage2 | `pttppg_stage2_denoiser.py:135-149,:628-650` | time mask沿频率复制、phase/训练问题 |
| SoftHR | `cnnppg_v7.py:416-443` | 每窗rFFT soft-argmax，不是track |
| 姿态EKF | `funcs.py:664-761`, `ppg.py:1161-1250` | Kalman只估姿态/gyro bias，不是HR |
| JOSS字样 | `ppg.py:1981-2017` | HRV Python库注释，不是JOSS运动HR |
| SVM PCA | `svm2_dataset_train.py:20,:1160-1175` | 手工特征降维，不是BSS |
| Motion A/B | `ppg_peak_hr_gating_train.py` 的 detector classes/loaders/benchmark | 10ch PPG+IMU独立路线 |
| PPG peak/IBI/gate | 同脚本主多任务路径 | PPG-only；ECG timing target未delay校正 |
| SQI | `frailty_3class_classifier.py:1671-1766` | 四项加权、top50/top70、质量聚合 |

所有代码的完整读取、SHA-256、函数/类/import 结构位于：

- `final_v0/records/generated/CODE_FILES.jsonl`
- `final_v0/records/generated/ROOT_FILES.jsonl`
- `final_v0/records/generated/CODE_DIAGRAM_COVERAGE.json`

## 4. 输入证据

### 4.1 PTT-PPG

- 路径：`physionet.org/files/pulse-transit-time-ppg/1.1.0/csv/s*_*.csv`
- header：`time,ecg,peaks,pleth_1..6,lc_1,lc_2,temp_1..3,a_x,a_y,a_z,g_x,g_y,g_z`
- 机器 summary：`final_v0/records/generated/inputs/physionet.org.summary.json`
- 完整头部 manifest：`final_v0/records/generated/inputs/physionet.org.jsonl`

### 4.2 SIM/iAMwell/MIMIC/VitalDB

解析路径由 `ppg_peak_hr_gating_train.py` 明确：

- `physionet.org/files/simultaneous-measurements/1.0.0/generated_data/`
- `physionet.org/files/iAMwell Dataset - Intelligent Athlete Monitoring for Cardiovascular Wellness/`
- `physionet.org/files/MIMIC/mimic_perform_*_csv/` 与指定 MAT extra holdout
- 可选 VitalDB API `SNUADC/PLETH`、`SNUADC/ECG_II`

### 4.3 本地双波长/分类数据

- 根：`PPG_Testing_05_01_2026/`
- 示例：`tradeali.csv` → `IrFinger,RedFinger,GreenWrist,RedWrist,SensorTemp,AmbientTemp`
- 机器 summary：`final_v0/records/generated/inputs/PPG_Testing_05_01_2026.summary.json`
- 完整 header manifest：`final_v0/records/generated/inputs/PPG_Testing_05_01_2026.jsonl`

### 4.4 路径与隐私

- 2,387 条代码静态路径引用：`final_v0/records/generated/CODE_PATH_REFERENCES.jsonl`。
- `.env` 只做完整字节校验，值未进入任何报告；证据仅保留变量名和 `values_redacted=true`。
- 历史 notebook 中的外部 D 盘路径只作为运行历史，不写成可移植默认。

## 5. 输出证据

### 5.1 历史 waveform/detector

| 路线 | 实际路径 | 关键文本证据 |
|---|---|---|
| v7 | `results/` | detector/denoiser fold JSON、`compare.json` |
| v7.2 | `results_v72_noleak/` | `summary_compare.csv`、holdout/CV JSON/CSV |
| v7.4 | `results_v7_4/` | detector `cv_summary.json`；walk/run denoiser meta |
| v8 MaskNet | `results_denoiser_v8/` | 空目录，0文件 |
| Stage2 | `results_stage2/` | 空目录，0文件 |
| Stage1 | `results_stage1/` | 历史 OR/AND detector summary |
| legacy v8 | `results_v8_audit/` | 三个窗口配置 summary/audit/图/NPZ |

对应 summary：

- `final_v0/records/generated/outputs/results.summary.json`
- `.../results_v72_noleak.summary.json`
- `.../results_v7_4.summary.json`
- `.../results_denoiser_v8.summary.json`
- `.../results_stage2.summary.json`
- `.../results_stage1.summary.json`
- `.../results_v8_audit.summary.json`

### 5.2 Hybrid

- `results_hybrid_denoiser_raw_imu/`
- `results_hybrid_denoiser_raw_imu_baseline/`
- `results_hybrid_denoiser/`
- `denoiser_preview_output/`

文本证据为 meta/history/splits/delay/export contract；preview 只有 8 PNG。对应 summary 位于 `final_v0/records/generated/outputs/` 同名 `.summary.json`。

### 5.3 Peak/IBI/Motion A-B

- `.CNN_results/<run>/`
- 关键文件：`cv_summary.json`、`holdout_summary.json`、`extra_holdout_summary.json`、`group_scorecards.json`、`detector_benchmark_summary.json`、`deploy_export.json`、PT/ONNX/meta/plots。
- `.CNN_results` 机器 summary：`final_v0/records/generated/outputs/.CNN_results.summary.json`。

### 5.4 SQI/frailty

主效应表：

`results_frailty3/_overfitting_sweep/20260630_0630_overfitting_sweep_generalization_rank2/analysis_report_20260703_1202/tables/main_effects_generalization.csv`

top配置表：同目录 `tables/top5_by_sqi_mode.csv`。整个输出 summary：`final_v0/records/generated/outputs/results_frailty3.summary.json`。

## 6. 机器验证证据

| 文件 | 证明范围 |
|---|---|
| `BASELINE_SUMMARY.json` | workspace/root/code/input/output 数量与字节基线 |
| `SCAN_RUNS.jsonl` | 25 个 baseline/input/output 扫描事务 |
| `SCAN_VERIFICATION.json` | EOF、bytes、SHA、事务和错误复核 |
| `ALGORITHM_DIAGRAM_VERIFICATION.json` | Mermaid 文档/图块完整性 |
| `CODE_DIAGRAM_COVERAGE.json` | 52 份代码路径的图覆盖 |
| `FINAL_V0_VERIFICATION.json` | final_v0 交付结构、工具、报告和临时文件检查 |
| `TOP_LEVEL_DIRECTORIES.json` | 输入/输出目录角色与规模 |

精选小型验证/summary 会复制到 `snapshots/verification/`。大型 canonical manifests 仍在 `final_v0/records/generated/`，并由本文件给出路径；这是有意的单一事实源设计，不是扫描缺失。

## 7. 快照规则

1. `snapshots/` 的每个文件都由 `final_v0/tools/build_m0_history_package.py` 从明示源路径按字节复制。
2. `M0_SOURCE_SNAPSHOT_MANIFEST.json` 保存 source、snapshot、bytes、SHA-256 和 equality。
3. 默认构建器不允许静默覆盖内容不同的既有历史快照；刷新必须显式参数，并应先获用户确认。此次刷新请求被安全审查拒绝，因此既有 v1 文件保持原字节。
4. 本次路线使用追加式 v2：新建 `M0_SOURCE_SNAPSHOT_MANIFEST_V2.json`、`M0_PACKAGE_VERIFICATION_V2.json`、`08_M0_PACKAGE_TREE_V2.md` 和带 v2 名称的变化快照。
5. `06_M0_PACKAGE_TREE.md` 继续作为 v1 历史树；`08_M0_PACKAGE_TREE_V2.md` 给出当前追加包的完整 tree、bytes、SHA-256 和内容说明。
6. 总项目索引仍由 `final_v0/tools/update_final_v0_index_detailed.py` 维护。

## 8. 外部来源边界

本轮未使用网络。TROIKA/JOSS 的本地存在性结论来自全代码检索；“谱重建/运动谱抑制/谱峰跟踪”只作为一般设计映射，不提供未核验论文引用。若用户允许后续文献精读，新的引用、原论文算法和与本项目接口的差异必须另建文献证据文件，不能回写成“本轮已验证”。

## 9. 可重复复核顺序

1. 读取本目录 `README.md` 与 `01`–`04`。
2. 检查 `M0_SOURCE_SNAPSHOT_MANIFEST.json`。
3. 运行 `python3 final_v0/tools/build_m0_history_package.py --verify-only`。
4. 运行既有四份 verifier。
5. 对任一论文数值回到实际 output JSON/CSV，而不是只引用本归档摘要。
6. 检查 Git 状态，确认 workspace 根没有由本轮新增的写入。

## 10. 2026-08-03 用户确认路线的证据索引

| 类型 | 路径 | 证据含义 |
|---|---|---|
| 决策片段 | final_v0/records/decisions/20260803_m0_madenoiser_route.md | 用户确认顺序、评估边界与未决标签语义 |
| 完整路线 | 07_CONFIRMED_MADENOISER_FOLLOWUP_ROADMAP.md | SQI、Motion-29、四路线、PTT 与 frailty 选择的完整合同 |
| 算法流程图 | final_v0/algorithm_diagrams/m0/07_MADENOISER_ROUTE_TO_FRAILTY_FEATURE_SELECTION.md | 串行前置关系、原始分支、统一后端和 nested selection |
| 工作记录 | final_v0/records/log_entries/20260803_madenoiser_confirmed_route.md | 本次路线写入范围与明确未执行事项 |
| 未来结果根 | final_v0/benchmarks/m0_signal_routes/<run>/ | 尚不存在；仅是已确认的输出合同 |
| v2 包证据 | M0_SOURCE_SNAPSHOT_MANIFEST_V2.json、M0_PACKAGE_VERIFICATION_V2.json、08_M0_PACKAGE_TREE_V2.md | 追加式保存本次路线，不覆盖 v1 历史证据 |

### 10.1 证据等级

- 决策片段、路线文档和算法图是 confirmed_plan，证明用户选定了后续方向。
- 它们不是 implementation_evidence，不能证明代码、训练、CV 或 benchmark 已运行。
- 原有 Motion A/B、SQI sweep、denoiser 和 PTT 输出继续按前述历史证据等级引用，不因新路线确认而升级。

### 10.2 后续结果登记规则

每次实际实现必须新增独立 run manifest、split manifest、per-record/per-subject metrics 与失败表，再把其 SHA-256 和状态写入本索引。不得用计划文档路径替代结果文件，也不得覆盖旧 run 来制造“最新结果”。
