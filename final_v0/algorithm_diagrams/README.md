# Algorithm Diagram Registry / 算法图索引

> 此文件由 `tools/sync_algorithm_index.py` 自动生成；请编辑具体图文件，不要手工编辑本索引。
> Generated mechanically; edit the source diagram files rather than this index.

- Diagram documents / 图文档数量：22
- Format / 格式：Markdown + Mermaid
- Convention / 约定：实线为运行数据流；虚线为监督、评价、风险或审计引用。

## Diagram files / 图文件

### `algorithm_diagrams/00_PROJECT_HISTORICAL_SIGNAL_FLOW.md`

- 标题 / Title：项目历史信号处理总图 / Historical Signal-Processing Map
- 内容 / Content：本图把 M0 审计到的历史输入、五类 motion/heartbeat 路线、实际证据和最终研究决策放在同一条可追溯链上。实线表示运行数据流；虚线表示监督、评价或审计引用，不表示部署时输入。
- 大小 / Bytes：2684
- SHA-256：`1ed28714c1f7f46a779f259a31f940dc26e18deaf4c08e3b20e5aea1caf5a9e4`

### `algorithm_diagrams/01_PROJECT_END_TO_END_PIPELINE.md`

- 标题 / Title：当前项目端到端算法总图 / Current End-to-End Project Pipeline
- 内容 / Content：本图表示当前仓库中实际存在的研究路线与产物，而不是未来 M1–M10 的已实现状态。虚线表示监督、评价、分析或历史依赖；实线表示主要数据/产物流。
- 大小 / Bytes：2382
- SHA-256：`d33d7e46b14aed00a76bd75b25cc8e40c872d2b4d4471d71fe0dee4e08acb279`

### `algorithm_diagrams/baseline/00_ARCHIVED_CODE_LINEAGE.md`

- 标题 / Title：归档代码版本关系总图 / Archived-code Lineage Map
- 内容 / Content：箭头表示代码演化或直接替代关系，不表示后代自动修复前代的所有方法学问题。归档输出应归因到其实际生产版本；当前根文件只在有明确schema/路径证据时继承结果。
- 大小 / Bytes：1585
- SHA-256：`6f2e339706863827ea78ca1bec20fe4e8032c4e90ee9c9df1a1be284890e98f8`

### `algorithm_diagrams/baseline/01_NON_M0_ROOT_SCRIPT_ATLAS.md`

- 标题 / Title：非 M0 根脚本与 Notebook 图册 / Non-M0 Root Script and Notebook Atlas
- 内容 / Content：本图册覆盖根目录中不属于 M0 的8个Python入口和5个Notebook；每节描述当前保存代码的直接职责、输入和输出，不代表未来TODO已完成。
- 大小 / Bytes：5755
- SHA-256：`70499233e6f32ada38ca59f8dcd0fef255cf1fa301f6d030d28f5c4ae7aa8a68`

### `algorithm_diagrams/baseline/02_ARCHIVED_SCRIPT_ATLAS.md`

- 标题 / Title：23个归档代码/Notebook结构图册 / Archived Script Atlas
- 内容 / Content：23个图块与 `CODE_FILES.jsonl` 的23个非根路径一一对应。它们只用于历史溯源和输出归因；当前算法候选仍以根代码为准。
- 大小 / Bytes：5283
- SHA-256：`26c6a8d5454e5b8d3856c5368c36b6b8eb070eea3ba9ddae1e5011b8f2bbe22f`

### `algorithm_diagrams/m0/01_FOUNDATION_FUNCS_PPG.md`

- 标题 / Title：M0 基础函数与 Dash 算法图 / Foundation Functions and Dash Flow
- 内容 / Content：基础算法透明且可作为 M3 候选，但当前双实现、参数错位、边界coverage和runtime默认路径使其处于 `implemented_unverified`；不得直接当作统一公共实现。
- 大小 / Bytes：1807
- SHA-256：`c8586eb04bfe58fc3ff56f09e1d0221da600aa853312144ad3cc03638018f431`

### `algorithm_diagrams/m0/02_V7_TO_STAGE2_EVOLUTION.md`

- 标题 / Title：v7 至 Stage-2 演化图 / v7-to-Stage-2 Evolution
- 内容 / Content：Mermaid algorithm structure and audit annotations.
- 大小 / Bytes：3068
- SHA-256：`53d16384e83b8ce98ffafd2dc3107be803f5dee76a19a5a2f0175d5e981efb88`

### `algorithm_diagrams/m0/03_HYBRID_SUITE.md`

- 标题 / Title：Hybrid 去噪、导出与运行图 / Hybrid Denoiser Suite
- 内容 / Content：Mermaid algorithm structure and audit annotations.
- 大小 / Bytes：2256
- SHA-256：`de5bb69fc9fe0d2513986380f2c9d87b9ef88aa53b631af56a6311a1b671944f`

### `algorithm_diagrams/m0/04_HEARTBEAT_AND_MOTION_AB.md`

- 标题 / Title：Heartbeat 与 PPG+IMU Motion A/B 图 / Heartbeat and Motion A/B
- 内容 / Content：P01 与 P02 虽在同一训练脚本中，但输入、target和证据不同：P01 是 PPG-only 的历史 heartbeat尝试；P02 才是 PPG+IMU motion-state benchmark。后续不得把 P02 的外部性能解释为 P01 gate 或 peak 的性能。
- 大小 / Bytes：2515
- SHA-256：`1d9a75b196e673209d6867468a90e96bb6c99c4893deb62e6690822582cae835`

### `algorithm_diagrams/m0/05_SCRIPT_ALGORITHM_ATLAS.md`

- 标题 / Title：M0 逐脚本算法结构图册 / Per-script Algorithm Atlas
- 内容 / Content：本图册覆盖 M0 范围内承担算法、训练、评价、导出或运行职责的每个脚本。每节只画该文件的直接职责；跨脚本关系见上级总图。
- 大小 / Bytes：6934
- SHA-256：`d6681b04fc2c0235be1669ef026001774ddee3f681056e47b77e3629963685d4`

### `algorithm_diagrams/m0/06_FIVE_METHOD_FAMILIES_IMPLEMENTATION_AND_BENCHMARK.md`

- 标题 / Title：M0 五类方法、统一实现与 Benchmark 算法图
- 内容 / Content：本图把本轮五类扩展审计连接到三个工程问题：motion detector、denoising 和动态 HR。实线表示未来可执行数据流；虚线表示监督、评价、安全门或失败历史。所有“未实现”节点只是已定义的实现路线，不是现有结果。
- 大小 / Bytes：5558
- SHA-256：`dec93e8d4901361ec78c512658ca08ce9bc17657851a7a0d9d3036a40aac1a12`

### `algorithm_diagrams/m0/07_MADENOISER_ROUTE_TO_FRAILTY_FEATURE_SELECTION.md`

- 标题 / Title：MAdenoiser 已确认路线到 Frailty 特征选择算法图
- 内容 / Content：状态：路线已确认；实现、训练与 benchmark 尚未开始。
- 大小 / Bytes：3042
- SHA-256：`1cb90ecec9db32867f424c3db82aac91bac53281e075067a26807119690c5905`

### `algorithm_diagrams/m0/08_ACTIVITY_MOTION_TRANSFER_AND_RECOVERY_FEATURES.md`

- 标题 / Title：Activity/Motion 迁移重训、SQI 与恢复特征流程图
- 内容 / Content：> 状态：监督语义已确认；实现、训练与 benchmark 尚未开始。
- 大小 / Bytes：2776
- SHA-256：`3dafee689821990a7318cb2b4c5510a99623e8c4d2855107324bef59da2b7e3a`

### `algorithm_diagrams/m1/00_END_TO_END_MOBILE_PIPELINE.md`

- 标题 / Title：M1 端到端移动处理中心架构图
- 内容 / Content：> 状态：接口与平台合同已定义；模块实现、硬件实测和最终配置尚未完成。
- 大小 / Bytes：2062
- SHA-256：`35eb6e96a8f38985c39a5cb7df12b86cdc64089fec9cfd3a0ffb0fbbbc9f3d53`

### `algorithm_diagrams/m1/01_END_TO_END_MOBILE_PIPELINE_V2.md`

- 标题 / Title：M1 V2 端到端移动处理中心架构图
- 内容 / Content：> 当前权威图；V1 图保留为初始设计历史。
- 大小 / Bytes：1908
- SHA-256：`83019dfe4625ef344940a93d3369df35570e474f7687c5ccc4469fb6f420aea5`

### `algorithm_diagrams/m1/02_SEQUENTIAL_SQI_MOTION_ROUTING_V3.md`

- 标题 / Title：M1 V3 顺序 SQI–Motion–Denoiser 路由图
- 内容 / Content：> 权威范围：取代 V1/V2 中 quality action owner 的路由图；V2 的输入、流式、bundle 和 provider fallback 图继续有效。
- 大小 / Bytes：3484
- SHA-256：`519779862b88b259b039afec7e5dee86449ad28f9425409508d0c1bed7d8a7a6`

### `algorithm_diagrams/m2/00_DATA_MANIFEST_DUAL_FOLD_AND_PROTOCOL.md`

- 标题 / Title：M2 数据 Manifest、双 Fold 注册表与评估协议图
- 内容 / Content：> 当前权威 M2 数据/分折图；实线为数据或物化 membership，虚线为审计、限制或结果引用。
- 大小 / Bytes：1692
- SHA-256：`b74f94efa6ef47e927dffd851b9656057b9e61a7a675660291d078aa49b35151`

### `algorithm_diagrams/m2/01_EXTERNAL_SYNCHRONIZED_REFERENCE_MANIFEST.md`

- 标题 / Title：M2 外部同步 ECG/PPG/IMU 证据与资格图
- 内容 / Content：> 本图区分强人工 ECG reference、pseudo reference、motion 资格与 BSS 元数据门；虚线表示限制，不是模型运行输入。
- 大小 / Bytes：1373
- SHA-256：`3fd614053d8d7a404a6b88cf5a924ec85b9533db9114ff82afe17de294bad7f1`

### `algorithm_diagrams/m3/00_UNIFIED_PREPROCESSING_AND_SIGNAL_API.md`

- 标题 / Title：M3 统一预处理与信号 API / Unified Preprocessing and Signal API
- 内容 / Content：本图固定数据先经过可追溯质量门，再进入任务 profile；EKF 是 IMU 主路线，LPF 只作为输入完全一致的对照。任何 invalid/insufficient 状态都不得伪造特征。
- 大小 / Bytes：1635
- SHA-256：`b6e8911d3b6b4c10b78fa16d3908954f40820295b00199ae3186565044a2be75`

### `algorithm_diagrams/m3/01_IMU_EKF_PRIMARY_AND_LPF_COMPARATOR.md`

- 标题 / Title：M3 IMU：无预校准 ESKF 主路线与 LPF 对照
- 内容 / Content：两条路线共享原始六轴、显式单位、质量 mask、20/40 Hz 前端和 jerk；禁止 EKF 失败后自动输出 LPF，确保比较只改变重力估计方法。
- 大小 / Bytes：1412
- SHA-256：`b5e92b02b41dc325fe463e08efe6425ccbb47bde9b353698cde208bbcc79c936`

### `algorithm_diagrams/m3/02_PEAK_PPI_HR_PRV_COMMON_BACKEND.md`

- 标题 / Title：M3 Peak、PPI、HR 与 PPG-derived PRV 公共后端
- 内容 / Content：corrected_v1 不再让异常 PPI 删除峰，也不生成 RED/IR 共识峰；同一公共后端服务 high-quality raw 与 denoised feature 路线。
- 大小 / Bytes：1007
- SHA-256：`c65ca1668c401305edb61613958be106e33394f377310c44d217af92e007b25e`

### `algorithm_diagrams/m3/03_REFERENCE_TEST_AND_PARITY_MATRIX.md`

- 标题 / Title：M3 固定 Reference Test 与 Parity 矩阵
- 内容 / Content：测试从 deterministic fixtures 覆盖质量门、滤波、单位、ESKF/LPF、physiology 和 fold-only scaling；合成真值只作工程验收，不冒充临床验证。
- 大小 / Bytes：1068
- SHA-256：`34c79d831c9d3fe02c4cdf9901b9f5e769b68f6f30932ca6feec62c53054ba18`
