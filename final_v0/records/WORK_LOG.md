# 工作日志 / Work Log

> 本文件由 `tools/sync_tracking_docs.py` 从 `records/log_entries/*.md` 自动重建。
> 逻辑工作使用不可变条目保存；追踪文档自身的同步不另记日志。

## 2026-08-02 — 全部52份代码/Notebook图覆盖通过

- Mermaid静态验证：`pass`；10份图文档、67个Mermaid图块，结构/fence无失败。
- 代码图覆盖：`pass`；52/52真实manifest路径均有逐脚本图入口。
- 分组：M0根16、非M0根13、归档23；缺失0、重复路径0、分组偏差0。
- 机器证据：`ALGORITHM_DIAGRAM_VERIFICATION.json`与`CODE_DIAGRAM_COVERAGE.json`。

---

## 2026-08-02 — 23个非根归档代码I/O与图册

- 扫描：23份代码/Notebook逐字节复扫、SHA核对；所有Python静态编译通过。
- 写入：`ARCHIVED_CODE_IO_INVENTORY.md`、归档lineage总图、23入口逐文件结构图册。
- 结果：明确`results_detector_v8`、`results_stage1`、`results_v7_3`的历史生产关系；识别detector fix lineage、两个完全相同fingertip副本和历史Notebook失败状态。
- 判定：全部为历史/归档/实验或直系前身，不覆盖当前根实现，不作为独立新科学方法重复计数。

---

## 2026-08-02 — 52份代码总索引与图覆盖校验器

- 写入：`CODE_IO_MASTER_INDEX.md`与`verify_code_diagram_coverage.py`。
- 分组：16个M0根入口、13个非M0根入口、23个非根归档入口；总计52，与代码manifest一致。
- 校验逻辑：每个真实路径必须出现在对应逐脚本图册；重复路径、数量或分组偏差都会失败。
- 同批更正：补充旧SVM Notebook、Esther列名、FilteredWalkTest采样率和16July SpO2异常细节。

---

## 2026-08-02 — final_v0 交付校验器

- 写入：`tools/verify_final_v0_delivery.py`。
- 检查：所有final_v0 Python的AST和双语说明、路径不越界、必需文档、详细树覆盖、扫描/算法图验证状态。
- 输出：严格JSON `records/generated/FINAL_V0_VERIFICATION.json`；任何失败均返回非零状态。
- 边界：工具不读取或修改final_v0之外的项目文件；最终归档图写完后再执行正式验收。

---

## 2026-08-02 — 内容感知文件树索引器

- 原因：旧索引虽有完整树、字节数和SHA，但对部分报告/manifest的说明过于通用。
- 写入：`tools/update_final_v0_index_detailed.py`；Markdown提取标题/实质段落，JSON/JSONL提取范围/记录数，Python提取模块职责/入口。
- 边界：只读 `final_v0`，只重写 `FINAL_V0_TREE.md`；索引更新不递归生成日志。
- 目标：保证 `final_v0` 每个永久文件都在一张树中拥有文件名、内容说明、字节数和SHA-256。

---

## 2026-08-02 — M0 算法图与逐脚本图册

- 写入：项目历史信号总图、基础函数图、v7→Stage-2演化图、Hybrid套件图、Heartbeat/Motion A-B图、16入口逐脚本图册。
- 图示约定：实线为运行数据流；虚线为监督、评价、风险或审计引用；阻断和空结果直接落在相应算法节点旁。
- 自动维护：新增 `sync_algorithm_index.py` 与 `verify_algorithm_diagrams.py`，用于重建图索引并检查 Mermaid fence、图类型和脚本覆盖。
- 边界：图是历史代码结构与证据的审计表示，不代表历史算法被批准恢复。

---

## 2026-08-02 — M0 算法图验证通过

- 验证器：`tools/verify_algorithm_diagrams.py`。
- 结果：`pass`；6份算法图文档、29个 Mermaid 图块、16个预期 M0 脚本入口全部覆盖。
- 机器证据：`records/generated/ALGORITHM_DIAGRAM_VERIFICATION.json`；缺失脚本0，失败项0。
- 索引：`algorithm_diagrams/README.md` 已按标题、内容、字节数和 SHA-256 自动重建。

---

## 2026-08-02 — M0归档输出生产关系附录

- 写入：`M0_ARCHIVED_LINEAGE_EVIDENCE.md`。
- 核对：Stage1 holdout OR/AND、legacy detector v8、v7.3、v8 audit fix链和v7.2目录归因。
- 结果：Stage1 OR BA .6920、AND BA .8623仅为activity detector前身；fix2/3/6/8归为同一D01工程lineage，不重复计方法。
- 影响：补强历史provenance；不改变M0主registry、证据等级或full-waveform结论。

---

## 2026-08-02 — M0 crosswalk 文件数按 manifest 更正

- 发现：人工整理表中的若干目录文件数按预期结构估算，与输出扫描manifest不一致。
- 更正：以 `records/generated/outputs/*.summary.json` 的 `file_count` 为唯一依据，更新 `results=5`、`v72=16`、`v7_4=55`、`v7_3=33`、`v8_audit=30`、两个hybrid variant各8、legacy hybrid=6。
- 工具：`correct_m0_crosswalk_manifest_counts.py` 对6个完整字符串执行唯一命中替换。
- 影响：只纠正实际文件数量；算法、指标、证据等级和M0结论不变。

---

## 2026-08-02 — M0 crosswalk、风险与人工决策门

- 写入：`M0_CODE_OUTPUT_CROSSWALK.md`、`M0_RISK_REGISTER.md`、`PROJECT_WIDE_SCAN_FINDINGS.md`、`HUMAN_DECISION_GATES.md`。
- 流程：用已验证manifest与逐脚本审计结果建立代码—输入—输出关系；将确定性错误、泄漏、代理目标、部署契约和论文表述风险分级。
- 结果：M0 历史输出已能按实际目录/run追溯；识别8项critical风险、18项high风险与10项medium/engineering风险；登记1个当前验收门和7个后续人工决策门。
- 边界：未执行或修复项目代码；未读取二进制载荷；未替用户作研究路线决定。

---

## 2026-08-02 — M0 Dash 工具文件名更正

- 发现：三处审计引用写成 `dash_denoiser_utils.py`，实际根文件为 `ppg_denoiser_dash_utils.py`。
- 更正：使用只允许每目标命中一次的 `correct_m0_dash_filename_reference.py`，精确修改 crosswalk、逐脚本图册与图覆盖校验器。
- 复核：更正后算法图验证仍为 `pass`；6份图文档、29个 Mermaid 图块、16个 M0 入口全部覆盖。
- 影响：只改审计文件名，不改变算法、指标、风险等级或项目源文件。

---

## 2026-08-02 — M0 最终验证

- 扫描证据：`pass`，失败0；baseline + 7输入 + 17输出共25笔事务有效。
- 算法图：`pass`，10份图文档、67个Mermaid图块；52/52代码/Notebook路径均有逐脚本图，缺失0。
- final_v0交付：`pass`；验证时119个永久文件、13个Python工具；AST、双语说明、路径边界、必需文档、详细树、扫描与图状态均有效。
- M0状态：`complete_awaiting_user_acceptance`；未执行M1，未写`_agent`，未修改final_v0之外文件。

---

## 2026-08-02：M0 全量扫描与证据校验 / M0 full scan and evidence verification

- 状态 / Status：`reporting_in_progress`
- 代码 / Code：52 个代码文件逐字节完整读取，均保留 SHA-256 或明确错误记录；错误数 0。
- 输入 / Inputs：7 个目录、6,405 个文件、34,581,621,834 bytes，逐文件读取 65,536 bytes 以内头部并识别结构；错误数 0。
- 输出 / Outputs：17 个目录、28,670 个文件、8,165,668,577 bytes。
- 输出文本 / Output text：9,314 个文本文件完整读取至 EOF，均满足 `bytes_read == file_size` 并记录 SHA-256。
- 非文本输出 / Binary outputs：19,356 个文件仅登记名称、格式和大小。
- 一致性 / Consistency：25 个预期扫描事务全部存在；`SCAN_VERIFICATION.json` 状态 `pass`，失败数 0。
- 当前工作 / Current work：编制 M0 registry、代码—输入—输出 crosswalk、论文证据报告及算法图。

---

## 2026-08-02：M0 Method Registry 完成

- 状态 / Status：`complete`
- 新增 / Added：`records/M0_METHOD_REGISTRY.md`。
- 范围 / Scope：17 组 foundation、v7、detector、hybrid、heartbeat 方法。
- 字段 / Fields：输入、输出、监督、预处理、参数、数据/split、协议、指标、实际结果、失败、部署和统一状态。
- 结论 / Conclusion：没有方法达到 `strictly_validated`；full-waveform reconstruction 保持失败/废弃主路线结论。

---

## 2026-08-02：M0 总报告与论文证据边界 / M0 executive report and paper-evidence boundary

- 状态 / Status：`complete`
- 新增 / Added：`records/M0_EXECUTIVE_REPORT.md`、`records/M0_PAPER_EVIDENCE.md`。
- 内容 / Content：全量扫描流程、算法路线、验收映射、关键定量结果、证据分级、允许与禁止的论文表述。
- 依据 / Evidence：代码 SHA-256 审计、输入/输出 manifests、实际 scorecards、历史 `_agent` 记录。
- 未改变 / Unchanged：原始代码、数据、历史结果与 `_agent`。

---

## 2026-08-02 — 全项目总图与非M0根脚本图册

- 写入：`01_PROJECT_END_TO_END_PIPELINE.md`与`baseline/01_NON_M0_ROOT_SCRIPT_ATLAS.md`。
- 覆盖：当前数据→信号处理→motion/heartbeat→Frailty3/ASA/SVM→评价/论文的总流；8个非M0 Python入口和5个Notebook逐一图示。
- 约束：图表示当前代码和产物，不把M1–M10待办画成已完成；泄漏、不可编译和保存运行错误均在对应节点标注。
- 后续：归档/历史目录中的23个代码文件将在独立lineage图册登记，避免与当前根入口混用。

---

## 2026-08-02 — 根目录45文件逐文件I/O清单

- 覆盖：16个M0代码入口、8个非M0主脚本、5个Notebook、16个配置/文本/二进制/来源附件。
- 依据：逐字节根文件manifest、全代码manifest、静态路径引用、逐脚本算法审计和实际输出文本。
- 安全：不记录`.env`值/hash；不复制Zone.Identifier URL；NPZ仅登记格式成员元数据。
- 结果：每个根文件均记录职责、输入、输出/对应和状态；未来TODO事项没有被提前执行。

---

## 2026-08-02 — 扫描证据重验与交付预验收

- 扫描重验：`verify_scan_evidence.py` 返回 `status=pass`、失败0；源代码SHA、输入头部、输出文本EOF和25笔事务证据未发现漂移。
- 首次交付预验收：仅发现 `verify_scan_evidence.py` 缺少独立中英文行内注释；模块双语docstring本身有效。
- 修复：使用唯一锚点脚本只在该文件加入双语路径边界注释，随后AST/扫描复验通过。
- 第二次预验收：`pass`；107个永久文件、11个Python工具；路径、必需文档、详细树、扫描和算法图状态均有效。
- 状态：归档代码清单/图册完成后仍需执行最终验收，因此本条明确为preverification。

---

# 扫描证据校验器建立 / Scan-evidence verifier added

- 日期 / Date：2026-08-02
- 状态 / Status：`implemented_unverified`
- 文件 / File：`final_v0/tools/verify_scan_evidence.py`
- 目的 / Purpose：核对 baseline、7 个输入目录、17 个输出目录及扫描账本的计数、总字节、EOF、SHA-256 和错误状态。
- 写入边界 / Write boundary：只写 `final_v0/records/generated/SCAN_VERIFICATION.json`。
- 下一步 / Next：运行校验器；通过后把统计结果写入 M0 扫描与结果报告。

---

# 分段扫描工具建立 / Sectioned scanner added

- 日期 / Date：2026-08-02
- 状态 / Status：`implemented_unverified`
- 文件 / File：`final_v0/tools/workspace_audit.py`
- 目的 / Purpose：分别执行根目录与代码完整读取、输入头部结构扫描、输出文本完整读取。
- 安全边界 / Safety：源扫描硬编码排除 `.git` 和 `final_v0`；证据只写入 `final_v0/records/generated/`；`.env` 只保留变量名和不可逆摘要，不保存值。
- 下一步 / Next：完成语法和边界测试，运行 baseline 后逐目录运行 input/output 扫描。
- 备注 / Note：本条因 Windows 沙箱暂时无法就地更新 `records/WORK_LOG.md` 而建立；不得丢失，后续汇总时保留来源链接。

---

## 2026-08-02：会话启动与只读基线 / Session initialization and read-only baseline

- 状态 / Status：`complete`
- 来源 / Source：用户指令、`AGENTS.md`、`_agent/WRITE_RULES.md`、`_agent/TODO.md`、只读命令结果。
- 写入边界 / Write boundary：仅允许写入 `final_v0/`；其余 workspace 内容只读。
- 权威任务清单 / Authoritative task list：`_agent/TODO.md`，按 M0–M10 顺序执行；每项完成报告后等待用户确认。
- Git 基线 / Git baseline：分支 `dev0`；用户已有修改 `AGENTS.md`、`_agent/PROJECT_STRUCTURE.md`、`_agent/README.md`、`_agent/TODO.md`、`_agent/WRITE_RULES.md`，本会话不触碰。
- 验证 / Verification：规则、TODO、根目录文件哈希及全 workspace 元数据基线均已读取。

---

## 2026-08-03 — Activity/Motion 监督确认与早期三分类历史追溯

- 操作 / Action：把用户确认的 B/R 静态、S/W 动态监督语义写入 M0，并追溯早期多类模型、结果与混淆矩阵。
- 数据核验 / Data audit：逐字节读取29人261份CSV；确认两个数据目录、统一8列结构、每角色29份、角色持续时间与全部活动后恢复顺序。
- 历史结论 / History：找到三分类 SVM 数据和649个 SVM 权重；找到“Rest好、Walk与Sit/Stand混淆”记录；未找到三分类 CNN 或 3×3 confusion matrix。
- 当前模型 / Current model：核验 PTT/SIM A/B CNN 为直接二分类；balanced_v2 external SIM 中 Light CNN BA `.7802`、F1 `.7634`，与内部满分共同显示域偏移。
- 新增 / Added：专题文档09、算法图08、决策 `M0-MOT-001`、两份机器审计JSON与既有二分类 confusion/result 证据副本。
- 状态 / Status：监督阻塞已解除；Motion-29 适配、nested 5-fold、阈值、SQI融合、恢复特征和frailty比较仍未实现。
- 边界 / Boundary：未训练、未反序列化pickle/PT/ONNX、未联网、未修改final_v0外文件、未写入 `_agent`。
- 同步 / Synchronization：随后生成追加式 v3 manifest/verification/tree，并刷新算法索引、工作记录和总文件树；追踪更新不递归记日志。

---

## 2026-08-03 — 候选路线未来方向字段规范化 / Candidate future-direction field normalization

- 操作 / Action：根据最终要求矩阵自审，将候选脚本文档末章标题由 [路线选择建议] 规范为 [路线选择与未来方向建议]。
- 范围 / Scope：仅修改 `M0_history_MA_denoising_detector_HR_feature/02_CANDIDATE_SCRIPTS_MOTION_DENOISING_DYNAMIC_HR.md` 的标题；既有算法、路径、结果、状态与建议正文不变。
- 原因 / Reason：使文档字段与用户要求的 [未来方向建议] 逐字对应，便于后续自动核验和人工定位。
- 同步 / Synchronization：随后机械更新聚合工作日志、专题树、快照校验和 `FINAL_V0_TREE.md`；这些追踪更新不递归新增日志。

---

## 2026-08-03 — 三问题候选路线目录 / Candidate-route catalog for three problems

- 操作 / Action：新增用户指定的 motion detector、denoising、动态 HR 候选脚本文档。
- 写入 / Written：`M0_history_MA_denoising_detector_HR_feature/02_CANDIDATE_SCRIPTS_MOTION_DENOISING_DYNAMIC_HR.md`。
- 覆盖 / Coverage：5 组 motion 候选、7 组 denoising 候选/失败历史、5 组动态 HR 候选以及共享 SQI 层。
- 每项字段 / Fields：脚本与应用位置、具体算法、输入数据名称/路径、输出路径/结构、已有结果、状态判定、风险和下一步。
- 主要判断 / Decision：P02 Light CNN 为 motion 首选；hybrid 先补生理 holdout；spectral candidate tracking 为动态 HR 新主线；Aboy++、DWT-A2、NLMS 与 legacy detector 均按对照而非成功方案保存。
- 数据边界 / Data boundary：没有新增或改写任何历史结果；只引用已存在的 JSON/CSV/代码事实。
- 验证 / Verification：最终批次将检查文档链接、快照、SHA-256、Mermaid 和全交付一致性。

---

## 2026-08-03 — M0 证据与来源索引 / M0 evidence and provenance index

- 操作 / Action：新增 M0 源码、输入、输出、机器验证和快照来源链索引。
- 写入 / Written：`M0_history_MA_denoising_detector_HR_feature/05_EVIDENCE_INDEX_AND_PROVENANCE.md`。
- 内容 / Content：列出关键函数行段、真实数据根/header、历史结果目录/关键文本文件、canonical generated manifests 和快照规则。
- 完整性策略 / Integrity strategy：小型人类报告与验证摘要进入快照；大型逐文件 manifest 保留单一 canonical 原位，通过路径与 SHA manifest 引用，避免两套证据漂移。
- 隐私 / Privacy：`.env` 值继续不写入；历史外部盘路径仅作不可移植运行证据。
- 文献边界 / Literature boundary：本轮未联网；TROIKA/JOSS 不提供伪引用，精确论文复现需用户另行授权。
- 验证 / Verification：待快照构建工具写入后执行 source/snapshot 字节一致性和 package tree 检查。

---

## 2026-08-03 — 五类方法与统一测试算法图 / Five-family and unified-gate diagrams

- 操作 / Action：在专业算法图目录新增五类方法、三个问题和统一 benchmark 的 Mermaid 图。
- 写入 / Written：`algorithm_diagrams/m0/06_FIVE_METHOD_FAMILIES_IMPLEMENTATION_AND_BENCHMARK.md`。
- 图块 / Blocks：当前证据→路线选择、五族公共合同、Adaptive安全门、谱域动态HR、双波长BSS+SQI、G0–G6测试/淘汰门，共6图。
- 语义 / Semantics：实线为可执行数据流；虚线为评价、安全或失败路径；未实现节点均有显式状态，不冒充已有结果。
- 同步 / Synchronization：随后机械更新 `algorithm_diagrams/README.md` 和 `FINAL_V0_TREE.md`，同步本身不递归记日志。

---

## 2026-08-03 — 五类方法逐源码审计 / Five-family line-level audit

- 操作 / Action：新增五类方法的代码、理论、应用、测试、缺口与实现可行性总审计。
- 写入 / Written：`M0_history_MA_denoising_detector_HR_feature/03_FIVE_METHOD_FAMILIES_CODE_TEST_IMPLEMENTATION_AUDIT.md`。
- 只读复核 / Read-only review：三项并行审计覆盖 52 份代码/notebook、4,190,267 bytes，SHA-256 mismatch 为 0；主进程复核关键函数行段和实际结果表。
- 代码发现 / Code findings：现有真实实现限于 IMU-NLMS、CEEMD-lite、DWT-A2、STFT工具和部分SQI；Wiener/RLS、标准小波阈值、CWT/WPT、EEMD/VMD/SSA、完整谱追踪和BSS均不存在。
- 算法发现 / Algorithm findings：ANC 独立性前提被运动诱发真实 HR 破坏；现有双路STFT没有IMU频率局部证据；BSS数据可用但实现为0；SQI只覆盖四个加权分量。
- 测试发现 / Test findings：历史 adaptive/decomposition 缺定量 scorecard；谱追踪/BSS 无测试；SQI只有轻微正向 generalization 消融而无独立质量验证。
- 输出 / Outcome：为五族分别规定统一接口、至少 5–10 类可执行测试、安全门和状态判定。
- 边界 / Boundary：没有把“可实现”写成“已实现”；没有联网读取 TROIKA/JOSS，也没有进入新算法编码。

---

## 2026-08-03 — M0 历史归档核心结果 / M0 history package core results

- 操作 / Action：在 `final_v0/M0_history_MA_denoising_detector_HR_feature/` 建立归档入口和完整 M0 结果/决定文档。
- 写入 / Written：`README.md`、`01_M0_COMPLETE_RESULTS_AND_DECISIONS.md`。
- 依据 / Evidence：重新扫描当前 Git/final_v0 状态；复核 52 份代码/notebook、现有 M0 registry/crosswalk/paper evidence、输入 schema、17 个输出目录及四份机器验证结果。
- 算法结论 / Algorithm outcome：完整保留 17 路历史方法状态，并新增自适应滤波、非平稳分解、谱域追踪、双波长 BSS、SQI 五类本地实现边界。
- 结果结论 / Result outcome：严格验证路线仍为 0；P02 motion A/B 为当前 detector 首选；完整波形恢复不恢复为主线；谱域候选峰+轨迹+SQI 为优先新路线。
- 边界 / Boundary：根目录源代码、输入、历史输出、`AGENTS.md` 和 `_agent/` 均未写入；未联网、未训练。
- 验证 / Verification：此条为归档第一批写入；后续批次完成后统一运行 package、diagram、scan 与 final_v0 完整校验。

---

## 2026-08-03 — M0 专题归档快照 / M0 history package snapshots

- 操作 / Action：将 M0 完整历史结论、三类候选路线、五类方法审计、统一测试合同、算法图与关键机器证据组织为独立专题归档。
- 写入 / Written：`M0_history_MA_denoising_detector_HR_feature/` 下的 `snapshots/`、`M0_SOURCE_SNAPSHOT_MANIFEST.json`、`M0_PACKAGE_VERIFICATION.json` 与 `06_M0_PACKAGE_TREE.md`。
- 构建工具 / Builder：新增并执行 `tools/build_m0_history_package.py`；该工具仅从工作区读取源证据，并且只写入本专题目录。
- 结果 / Result：43 份快照，共 1,004,668 字节；6 份必需正文与 7 份算法图齐全，共识别 35 个 Mermaid 图块。
- 校验 / Verification：`status=pass`；无缺失文档、无快照失败、无源文件漂移。
- 同步 / Synchronization：随后机械更新入口说明、工作日志、算法索引和 `FINAL_V0_TREE.md`；这些追踪更新不递归新增日志。

---

## 2026-08-03 — 统一实现与 Benchmark 合同 / Unified implementation and benchmark contract

- 操作 / Action：把五类方法的“可实现”要求固化为公共数据合同、接口、测试先决条件、指标、输出 schema 和验收门。
- 写入 / Written：`M0_history_MA_denoising_detector_HR_feature/04_UNIFIED_IMPLEMENTATION_AND_BENCHMARK_CONTRACT.md`。
- 流程 / Process：先定义 record/subject 边界与标签语义，再定义 nested subject split、公共基线、unit/synthetic/leakage/holdout/external/runtime 六级测试，最后定义路线 G0–G6 gate。
- 算法 / Algorithm：Adaptive、decomposition、spectral HR、BSS、SQI 均具有可编码接口；spectral+Viterbi 和新版SQI被排在优先实现顺序前端。
- 结果 / Result：任何路线都必须与 raw、bandpass、high-quality-only 比较，允许 missing/reject；subject 是统计单位；有合法 clean truth 才报告 waveform recovery。
- 人工决策 / Human gates：外部文献、第三方依赖、PTT双通道语义、motion目标定义和HR-error/coverage效用须在实际编码前询问。
- 状态 / Status：合同完成；尚未创建提议的实现目录、测试或新 benchmark 结果。

---

## 2026-08-03 — MAdenoiser 后续路线确认 / Confirmed MAdenoiser follow-up route

- 操作 / Action：把用户确认的 SQI-v2、Motion-29、四条 MA/HR/PPI 路线、PTT 监督 benchmark 和 Frailty feature/CV 选择规则固化为可执行合同。
- 新增 / Added：专题 `07_CONFIRMED_MADENOISER_FOLLOWUP_ROADMAP.md`、M0 算法图 07 和不可变决策片段 `M0-MAD-001`。
- 核心边界 / Boundary：29-subject cohort 与现有 PTT/SIM detector 结果分开；PTT `peaks` 记为 ECG R-peak reference；同 5-fold 最高 BA 必须按 nested selection 或 development-only 口径报告。
- 状态 / Status：路线已确认，所有实现与新 benchmark 仍未开始；Motion-29 在监督标签语义明确前阻塞。
- 同步 / Synchronization：随后更新 M0 核心文档、归档清单、算法索引、工作日志和文件树；追踪更新不递归记日志。
- 快照安全 / Snapshot safety：显式覆盖 v1 快照的请求被安全审查拒绝；没有既有快照被改写。
- 追加归档 / Additive archive：本次改用 v2 清单、验证、包树和变化快照，完整保留 v1 历史证据。

---

## 2026-08-14 — M1 端到端架构、统一合同与移动处理中心分档

- 操作 / Action：定义 SignalBatch→PipelineResult 模块顺序、机器 schema、可替换 registries、训练/推理隔离和三档中心处理平台。
- 用户确认 / User-confirmed：血压仪大小中心屏显处理设备；可穿戴 PPG+IMU；允许 NumPy/SciPy/ONNX Runtime/scikit-learn；需要高性能和性价比方案。
- 关键设计 / Key design：SQI 保留共同诊断出口，波形动作在 sqi_gate/coarse_denoise 中恰好选一；具体处理器留待 M9 实测锁定。
- 新增 / Added：M1 文档包、3 schemas、4 registries、3 example configs、双语验证器、4幅 Mermaid 流程图和决策 M1-ARCH-001。
- 边界 / Boundary：未联网、未安装依赖、未训练/推理、未修改 final_v0 外文件、未写入 `_agent`。
- 验证 / Validation：随后运行合同验证、算法图/覆盖验证、总树和 delivery verification；这些维护更新不递归记日志。

---

## 2026-08-14 — M1 V2 最终只读审计补强

- 操作 / Action：以追加式 V2 固化有界流式、窗口坐标/coverage、单一 action owner、完整 artifact hash 与 accelerator→CPU fallback。
- 原因 / Reason：现有文件补丁入口受沙箱读取故障；为避免绕过 `apply_patch` 或静默覆盖，保留 V1 历史并新建权威 V2。
- 新增 / Added：V2 当前状态、代码风险审计、3 schemas、4 registries、3 platform examples、V2 validator、3幅 Mermaid 图、M1-ARCH-002。
- 边界 / Boundary：未修改根代码、未联网、未安装 ONNX Runtime、未运行模型、未写 `_agent`。
- 验证 / Validation：运行 V2 schema/cross-registry validator、V1 package validator、JSON Schema 校验、全局图/扫描/delivery 验证。

---

## 2026-08-14 — M1 V2 验证表述修正与补充语义门

- 修正 / Correction：较早日志中的“JSON Schema 校验”仅指 schema 结构和 registry/config 交叉校验；本机无第三方 Draft 2020-12 引擎，因此该完整项未运行。
- 补充 / Added：新增零第三方依赖语义验证器，覆盖 ok/no-result 状态机、概率和、唯一 action owner、CPU fallback、locked artifacts、threshold 与 bundle path containment。
- 边界 / Boundary：这仍是合同 fixture 验证，不是模型 smoke、真实 artifact hash 检查或硬件 benchmark。
- 写入 / Writes：仅 `final_v0/`；`_agent` 未写。

---

## 2026-08-15 — M1 V3 顺序质量路由修订

- 操作 / Action：按用户修正，以追加式 V3 取代 V1/V2 的 SQI/coarse-denoise action-owner 路由；V2 输入、流式、bundle、平台和 provider fallback 继续有效。
- 算法 / Algorithm：必做 SQI + 可选 Motion → join；high/non-motion 绕过 denoiser；low 或 motion 按 run/session 级手动配置互斥执行 drop 或 denoise→FeatureBlock；invalid/unrecoverable 强制 drop，module failure fail-closed。
- 新增 / Added：V3 当前状态与详细合同、config/output schemas、active routing registry、3 platform examples、双语合同/语义验证器、专业 Mermaid 图和 M1-ARCH-003。
- 验证 / Validation：V3 CURRENT contract/cross-registry 3/3 example configs 通过；24/24 routing fixtures 通过；完整结果见 M1 包内机器报告。
- 校正 / Correction：首版 V3 validator 把 legacy migration 元数据中的旧字段名误报为活动字段；保留首版证据，新增 active-registry CURRENT 入口，不静默覆盖。
- 边界 / Boundary：未实现/训练模型，未运行 ONNX 或真实设备 benchmark，未联网、未安装依赖、未改 final_v0 外文件、未写 `_agent`。

---

## 2026-08-15 — M2 数据 manifest、双 fold 注册表与协议合同

- 操作 / Action：只读审计 Frailty3 和五类外部数据源；在 `final_v0/` 新增 M2 数据/阶段/协议包、生成器、验证器入口、机器 schemas、双注册表图和溯源合同。
- 算法 / Algorithm：完整字节/数值扫描 → file/subject manifests；保留 sklearn 1.4.2 历史 SGKF defect membership；同步置换 group 与 class-count rows 生成 corrected SGKF future membership；以固定 5×5、fixed epoch/no early stopping 输出 OOF。
- 用户决定 / User decision：路线 C 双注册表；未来唯一主协议 seeds=`42,10042,20042,30042,40042`；B/R/S/W 家族语义与 S/W before Relax 被确认，编号细节保持未验证。
- 边界 / Boundary：未训练或重跑模型；未修改原始数据、根代码、历史结果、`AGENTS.md` 或 `_agent/`；未安装依赖、未联网。
- 验证 / Validation：由 M2 生成器和验证器完成后写入机器报告；追踪文档自身更新不另记日志。

---

# M3 contract edge tests phase 7 / M3 合同边界测试第 7 阶段

- 时间 / Date：2026-08-15
- 状态 / Status：implemented_pending_execution
- 流程 / Process：新增独立边界测试，覆盖 strict JSON、M1 状态映射、profile/fs 错配、timestamp、双波长比例、训练折幅值门、低覆盖 PRV、无效通道和峰置信度。
- 算法 / Algorithm：所有无效或非有限输出以 explicit status/reason/null 表示；SQI 选择必须先满足 peak status 与 0–1 finite SQI。
- 结果 / Result：测试代码已保存，执行结果将在 reference report 中统一固化。
- 边界 / Boundary：只写 final_v0；没有写入 _agent 或根目录源文件。

---

# M3 core evidence builder phase 8 / M3 核心证据构建第 8 阶段

- 时间 / Date：2026-08-15
- 状态 / Status：implemented_pending_execution
- 流程 / Process：新增根源码 hash/crosswalk、PPG 频响、M2 完整性绑定和 EKF/LPF 合成真值比较的确定性构建器。
- 算法 / Algorithm：两条 IMU 路线使用同一 fixture、同一单位与同一 causal 前端；仅 profile ID 不同；无 silent fallback。
- 结果 / Result：构建器已保存，输出将在执行后写入 M3 evidence 与 M3_BUILD_REPORT。
- 边界 / Boundary：根源码和 M2 manifest 只读；输出仅在 final_v0。

---

# 2026-08-15 M3 公共预处理核心第一阶段

- 范围：仅在 `final_v0/M3_unified_preprocessing_and_signal_algorithms/` 新建公共合同、
  异常门控、PPG 滤波、fold-only scaling、profile registry 和 schema。
- 决策：执行用户确认的 corrected_v1、400 Hz、任务分滤波 profile、训练折统计量隔离，
  并为无预校准 EKF 主路线预留公共合同。
- 算法图：新增 M3 统一质量门、PPG、EKF/LPF 与 M1 路由衔接图。
- 边界：根目录和原始数据保持只读；此阶段尚未声明 M3 完成。
- 验证：后续阶段将加入 IMU、peak/PPI/HRV、fixtures 和机器校验报告。

---

# M3 D8 symmetric scorecard phase 21 / M3 D8 对称评价第 21 阶段

- 时间 / Date：2026-08-15
- 状态 / Status：implemented_and_reference_tested
- 流程 / Process：复扫 PTT evaluator 后，将只在顶层输出 corrected PPI/HR 的非对称结果改为 raw 与 delay-corrected 两个同 schema scorecard。
- 算法 / Algorithm：两分支均报告 one-to-one precision/recall/F1、timing samples/ms、PPI MAE、HR error、coverage 与 failure；delay artifact 新增 dataset、training split、preprocessing profile 与 algorithm provenance。
- 结果 / Result：合成训练/独立评价 fixture 中 raw 因 200 ms 生理延迟在 50 ms 门下无匹配，corrected F1=1、timing/PPI/HR error=0；全量 reference tests 42/42 通过。
- 边界 / Boundary：这是公式与无泄漏合同验证；真实 PTT OOF benchmark 留给 M4/M5，不能把 ECG detector preflight 当作 PPG peak 成绩。

---

# M3 D8 training-split identity phase 24 / M3 D8 训练分割身份第 24 阶段

- 时间 / Date：2026-08-15
- 状态 / Status：implemented_and_reference_tested
- 流程 / Process：在写入前重新扫描 M3 的 reference evaluator、D8 测试和 machine schemas；随后把 `training_split_id` 从可省略的空字符串升级为拟合 transit-delay artifact 时的必填身份，并重新运行完整参考测试。
- 算法 / Algorithm：ECG→PPG transit delay 只能在 training subjects 上估计；artifact 同时绑定 dataset、fold registry、training split、preprocessing profile 与 algorithm 身份。空白 `training_split_id` 立即 fail closed，防止无法追溯的延迟参数进入 OOF 或 external evaluation。
- 结果 / Result：M3 参考测试 42/42 通过，0 failures、0 errors、0 skipped；D8 的 raw 与 delay-corrected scorecard 保持对称，训练/评价 subject overlap 负例继续被拒绝。
- 边界 / Boundary：本阶段只强化身份与泄漏防线，不生成新的 PTT 性能结论；正式数值仍必须在冻结的 M2 fold registry 或显式 external holdout 上重跑。

---

# M3 decision contract phase 16 / M3 决策合同第 16 阶段

- 时间 / Date：2026-08-15
- 状态 / Status：user_decisions_recorded
- 流程 / Process：复扫全局人工决策门与既有不可变决策格式后，汇总本回话用户逐项冻结的 D1–D8。
- 算法 / Algorithm：未来主线固定 400 Hz profile-bound PPG、training-fold-only scaling、公共 corrected physiology、无预校准 quaternion error-state EKF 主路线和 0.3 Hz LPF 独立对照。
- 结果 / Result：新增 M3-PREPROCESS-001；关闭 M0 接受、M2 29-subject 双注册表及 M3 决策门；保留 M4/M5/M8 未来未决事项。
- 边界 / Boundary：记录决策不等于宣告 M4–M8 benchmark 已完成；M3 最终验收仍取决于正式 validator 与全局回归。

---

# M3 deprecated-profile fail-closed phase 29 / M3 弃用 Profile 关闭失败第 29 阶段

- 时间 / Date：2026-08-15
- 状态 / Status：implemented_pending_reference_test
- 流程 / Process：重新扫描 PPG registry 与运行时入口后，发现旧 `mobile_ppg_400_causal_v1` 已在机器合同中降为 deprecated alias，但 `preprocess_ppg` 仍只检查 modality 和采样率，可能被新实验直接运行。
- 算法 / Algorithm：PPG facade 现在同时要求 `status=future_active`、`modality=ppg`、用途属于 static/motion/peak/denoiser input，且 `resampling=no_resample`；任一不符立即拒绝。
- 结果 / Result：新增 deprecated mobile alias 负例；下一阶段运行完整参考测试确认无回归。
- 边界 / Boundary：旧 alias 仍保留在 registry 供历史配置解析与显式迁移，但不允许进入 corrected benchmark；移动端必须明确选择 static、motion 或 peak profile。

---

# M3 deprecated-profile tests phase 30 / M3 弃用 Profile 测试第 30 阶段

- 时间 / Date：2026-08-15
- 状态 / Status：implemented_and_reference_tested
- 流程 / Process：运行完整 M3 测试验证 future-active PPG profile gate 与 deprecated mobile alias 负例。
- 算法 / Algorithm：运行时不接受 `deprecated_alias` 或 `historical_reproduction_only` 作为 corrected preprocessing profile；profile ID、status、modality、purpose、resampling 和 fs 必须共同匹配。
- 结果 / Result：M3 参考测试由 44 增至 45，45/45 通过，0 failures、0 errors、0 skipped；旧 alias 被确定性拒绝，静态、运动、峰检测与 denoiser future profiles 无回归。
- 边界 / Boundary：兼容 alias 仍可被迁移工具读取，但不能作为未来 benchmark 的执行配置；旧结果复现与 corrected 主协议继续严格分离。

---

# M3 evidence authority phase 17 / M3 证据权威边界第 17 阶段

- 时间 / Date：2026-08-15
- 状态 / Status：implemented_evidence_rebuilt
- 流程 / Process：为避免 evidence 与 registries 出现双权威，重定义历史 crosswalk evidence 为 byte-hashed audit snapshot，并改进核心证据重建的合并逻辑。
- 算法 / Algorithm：核心 builder 重算四类核心证据及历史快照，同时逐文件复核并保留独立生成的 261-record EKF/LPF proxy 与 legacy peak parity；build report 登记 producer SHA 和每项 bytes/SHA256。
- 结果 / Result：M3_BUILD_REPORT 为 pass，共登记 6 项证据；全数据 proxy 和 parity 未在重建中丢失。
- 边界 / Boundary：未来 registries/historical_preprocessing_crosswalk_v1.json 是机器 authority；evidence 同名文件只保留扫描时点和源码哈希。

---

# M3 fixture manifest contract phase 25 / M3 Fixture 清单合同第 25 阶段

- 时间 / Date：2026-08-15
- 状态 / Status：generator_strengthened_pending_regeneration
- 流程 / Process：重扫 fixture 生成器、现有清单与 `m3.reference_fixture_manifest.v1` 后，补齐可由生成器确定性产生但旧清单未记录的完整性与科学语义字段。
- 算法 / Algorithm：每个 fixture 记录精确字节数、dtype、shape、逐字节 SHA-256 和双语语义；12 列 IMU fixture 显式冻结为 acceleration、gyroscope、gravity truth、dynamic-acceleration truth 四组三轴顺序；manifest 同时记录 schema 与 generator SHA-256。
- 结果 / Result：生成器源码已更新；本阶段尚未重写 fixture 清单，下一独立阶段将重建并验证字节哈希，避免把计划误记成已完成结果。
- 边界 / Boundary：fixture 只用于工程回归与算法真值测试，不构成 Frailty3 临床或真实姿态性能证据。

---

# M3 fixture manifest regeneration phase 26 / M3 Fixture 清单重建第 26 阶段

- 时间 / Date：2026-08-15
- 状态 / Status：regenerated_and_integrity_tested
- 流程 / Process：使用固定 seed `20260815` 重建三份 NPY fixture 和 strict JSON manifest，并逐项复算 generator SHA、文件 SHA 与文件字节数；随后重新运行完整 M3 参考测试。
- 算法 / Algorithm：PPG fixture 冻结 30 秒原始波形和 37 个真值峰；IMU fixture 冻结 4,800×12 数组，列序为三轴加速度、三轴角速度、三轴重力真值、三轴动态加速度真值。所有含义均写入 manifest，不再依赖测试代码中的隐式切片。
- 结果 / Result：三个 fixture 的原 SHA 保持不变；generator/文件哈希与字节数逐项一致；M3 参考测试 42/42 通过。当前系统 `jsonschema` 版本不提供 Draft 2020-12 validator，因此本阶段不伪报第三方 schema 验证通过，最终由 M3 自有合同 validator 重新执行结构门。
- 边界 / Boundary：合成真值只证明确定性实现和误差计算口径；Frailty3 仍无姿态重力真值或人工 PPG 峰真值。

---

# M3 fold/reference tests phase 12 / M3 训练折与参考评价测试第 12 阶段

- 时间 / Date：2026-08-15
- 状态 / Status：implemented_pending_execution
- 流程 / Process：为 M2 exact roster、OOF 污染、PTT train-only delay 与 evaluation overlap 新增正负例。
- 算法 / Algorithm：fold artifact 使用 payload hash 而非文件 hash；PTT 校正只允许应用到与 delay-training roster 不相交的 subject。
- 结果 / Result：测试已保存，执行结果将写入统一 M3 reference report。
- 边界 / Boundary：未运行正式 PTT benchmark；该测试仅验证合同和公式。

---

# M3 fold artifact envelope phase 22 / M3 训练折 artifact 完整封装第 22 阶段

- 时间 / Date：2026-08-15
- 状态 / Status：implemented_and_reference_tested
- 流程 / Process：在 schema 设计复核中发现 runtime artifact 已声明 m3.fold_fitted_artifact.v1，却缺少初稿 schema 要求的完整身份与 transformer envelope；选择升级 runtime 而不是放宽同版本 schema。
- 算法 / Algorithm：artifact 绑定 M2 fold file SHA 与 payload SHA、dataset/protocol/repeat/fold/seeds、exact train/OOF partition、M3 registry/profile payload hashes、feature schema、median imputer、RobustScaler/no-clip 参数及 canonical parameters SHA。
- 结果 / Result：成功 artifact 与负例测试均通过；全量 reference tests 42/42。
- 边界 / Boundary：调用方必须显式传入非空唯一 preprocessing_profile_ids；历史或 deprecated profile 不可生成 future artifact。

---

# M3 fold registry field correction / M3 fold registry 字段修正

- 时间 / Date：2026-08-15
- 状态 / Status：corrected_pending_retest
- 原因 / Cause：M2 的 subject_input_order 值是 stable_utf8_bytewise 排序规则名称，不是 subject roster。
- 修正 / Correction：fold union 不变量改为 train/OOF 零交集且 union 数量精确等于 n_subjects=29；实际成员仍来自物化 fold。
- 影响 / Impact：只修正 validator 字段解释，不改变任何 M2 fold 成员。

---

# M3 future fold scaling phase 20 / M3 未来训练折缩放第 20 阶段

- 时间 / Date：2026-08-15
- 状态 / Status：implemented_and_reference_tested
- 流程 / Process：复扫 raw8 view、FoldScaler 与 M2 exact-roster facade，确认低层 StandardScaler/clip 仍可绕过 D4 冻结路线后收紧 future-active 边界。
- 算法 / Algorithm：M2-bound fit_fold_scaler 只接受 RobustScaler、no clip；artifact 固定 m3_raw8_dynamic_sequence.v1。model view 还要求 scaler 来自 training role，并显式输出 RED/IR + dynamic-acc XYZ + gyro XYZ 语义。
- 结果 / Result：新增 standard 与 clip 负例；全量 reference tests 42/42 通过。
- 边界 / Boundary：StandardScaler 仍可在低层用于历史对照，但不得进入 corrected future leaderboard 或伪装为 D4 hybrid view。

---

# M3 fold schema alignment phase 23 / M3 训练折 Schema 对齐第 23 阶段

- 时间 / Date：2026-08-15
- 状态 / Status：runtime_aligned_pending_schema_regeneration
- 流程 / Process：对 runtime artifact 与 additionalProperties=false schema 做逐字段 diff，移除未声明的 flattened aliases，并补齐 provenance 与 zero-scale mask。
- 算法 / Algorithm：status=locked、fit_scope=training_subjects_only；transformers 为有序数组，包含 method/stage/feature names/float64/center/scale/impute/zero-scale；parameters_sha256 只哈希规范 transformer payload。
- 结果 / Result：runtime 正例与 leakage/scaling 负例均通过，reference tests 42/42；schema agent 正在同步 fold file SHA 与 payload SHA 两个不同字段。
- 边界 / Boundary：M2 fold registry 文件 SHA 为 c80e780d…388c，canonical payload SHA 为 0bca827f…f46，禁止混用。

---

# M3 Frailty3 IMU proxy builder phase 9 / M3 Frailty3 IMU 代理构建第 9 阶段

- 时间 / Date：2026-08-15
- 状态 / Status：implemented_pending_execution
- 流程 / Process：为 M2 冻结的 261 个文件逐一读取起始六秒，运行同上游 EKF 与 LPF，并按 B/R/S/W role family 汇总。
- 算法 / Algorithm：比较 coverage、dynamic-acceleration RMS 与 gravity-norm error proxy；Frailty3 无姿态真值，因此不计算或宣称 gravity RMSE。
- 结果 / Result：构建器已保存；机器结果待执行后写入 evidence。
- 边界 / Boundary：原始 CSV 只读且不复制；结果仅写 final_v0。

---

# M3 historical preprocessing discovery phase 19 / M3 历史预处理全量发现第 19 阶段

- 时间 / Date：2026-08-15
- 状态 / Status：implemented_evidence_rebuilt
- 流程 / Process：按 M3 TODO 重扫 final_v0 外全部 Python，以滤波、缩放、重采样、重力和 Aboy/peak 关键词的固定正则发现相关实现；逐文件读取并计算 bytes/SHA256。
- 算法 / Algorithm：UTF-8 bytewise 排序；root/active-candidate 与 Arc/archiv/Archive 分别标 historical_reproduction_only 和 historical_archive_reproduction_only；两类都不得成为 future-active。
- 结果 / Result：发现 35/35 个相关脚本、0 missing；其中 archive 17、root/candidate 18。evidence crosswalk 与 M3_BUILD_REPORT 已重建并登记完整 hash。
- 边界 / Boundary：该 evidence 是扫描快照；最终机器 authority 由 registries/historical_preprocessing_crosswalk_v1.json 定义并应覆盖同一 discovery roster。

---

# 2026-08-15 M3 无预校准 ESKF 与 LPF 对照实现

- 新增 quaternion multiplicative error-state Kalman filter 主路线。
- 新增共享单位、质量门、20/40 Hz 前端和 jerk 的 0.3 Hz LPF 重力对照。
- ESKF 显式输出 initialization_pending、tracking、prediction_only 和 no_estimate；
  无静态预校准、yaw 不可观与 bias 部分可观限制随输出保留。
- 禁止 ESKF 失败时静默回退 LPF。
- 修正 raw8 scaling：零 IQR 明确 no-estimate，取消未由训练折拟合的固定 clip。
- 状态：IMU 公共实现已落盘，尚待固定 fixtures 和机器验收。

---

# M3 legacy peak parity phase 15 / M3 历史峰算法一致性第 15 阶段

- 时间 / Date：2026-08-15
- 状态 / Status：implemented_evidence_persisted
- 流程 / Process：逐字节读取根目录 funcs.py、ppg.py 与 frailty_3class_classifier.py，通过 AST 白名单隔离执行所需函数，避免导入 UI/训练依赖及模块副作用。
- 算法 / Algorithm：对同一 400 Hz 固定 PPG fixture 比较 funcs/ppg 重复 Aboy++、classifier 的 aboypp_detect_peaks 及 detect_ppg_peaks alias；结果按 int64 peak 序列哈希冻结。
- 结果 / Result：funcs.py 与 ppg.py 36 峰逐值完全一致；classifier alias 35 峰完全一致；两类历史实现不等价，classifier 相比 funcs/ppg 少 index 8318。新增 2 项测试后全量 40/40 通过。
- 判定 / Decision：差异不是 corrected_v1 失败，而是论文/复现必须保留的历史实现分叉；future-active 入口仍唯一指向 m3_signal_core。
- 证据 / Evidence：M3 evidence/legacy_peak_parity.json，并已登记 M3_BUILD_REPORT.json。

---

# M3 M2 fold-artifact binding phase 10 / M3–M2 训练折 artifact 绑定第 10 阶段

- 时间 / Date：2026-08-15
- 状态 / Status：implemented_pending_tests
- 流程 / Process：新增 corrected M2 materialized fold resolver 与 scaler facade，拒绝自报 training role 但 roster 不符的拟合。
- 算法 / Algorithm：fit 前要求 observed subjects 精确等于 train roster、与 OOF 零交集；artifact 固化 dataset/fold/protocol/hash/seed/feature order/统计量。
- 结果 / Result：公共实现已保存，正负 membership 测试待加入并执行。
- 边界 / Boundary：M2 registry 只读，artifact 未来输出仅允许在 final_v0。

---

# 2026-08-15 M3 Peak、PPI、HR 与 PRV 公共实现

- 新增 corrected_v1 双极性 peak detector，固定 10 秒窗口、5 秒 hop 和 0.15 秒事件合并。
- PPI 固定为 0.30–2.00 秒；无效 PPI 不删除源峰，raw/valid/corrected NNI 分列。
- HR 门固定为至少 8 秒、5 个峰和 4 个有效 PPI。
- PPG-derived variability 使用 PRV 名称；60 秒 time-domain、120/300 秒 frequency tiers。
- RED/IR 分别检测，以 SQI 选主通道，平局选 RED，禁止 consensus 移动峰。
- 新增 physiology/reason-code registries 和公共后端算法图。
- 状态：算法已落盘，尚待 fixtures、真实片段审计和 validator。

---

# M3 physiology provenance phase 13 / M3 生理结果溯源第 13 阶段

- 时间 / Date：2026-08-15
- 状态 / Status：implemented_and_reference_tested
- 流程 / Process：重新扫描 Peak/PRV dataclass、所有构造点与边界测试后，将算法 ID、profile ID 和 NNI 语义从原因码中分离。
- 算法 / Algorithm：PeakResult 明确保存 m3_peak_corrected_v1、输入 profile 和 hard-valid PPI/no-imputation 语义；HrvResult 保存 PRV 算法及上游 peak/profile provenance。
- 结果 / Result：新增严格 JSON provenance 回归；当前全量 reference tests 为 38/38 通过（本阶段不写正式报告，最终收束时统一更新）。
- 边界 / Boundary：profile/algorithm 字段只表示来源，不表示质量原因；旧实现映射保留在历史 crosswalk。

---

# M3 PPG source/repaired views phase 18 / M3 PPG 原始与修复视图第 18 阶段

- 时间 / Date：2026-08-15
- 状态 / Status：implemented_and_reference_tested
- 流程 / Process：复扫 PPG facade 后发现 raw metrics 实际取自插值后的 quality signal；现将 source raw 与 repaired raw 分开保存并分别计算描述量。
- 算法 / Algorithm：source metrics 仅在原始有限样本上计算并记录 nonfinite fraction；repaired metrics 来自显式修复视图；filtered AC/pulse amplitude 继续单独记录，三层语义不再混淆。
- 结果 / Result：单点 NaN fixture 保留 source NaN、repaired view 全有限、repair 状态和比例正确；全量 reference tests 41/41 通过。
- 边界 / Boundary：未知 ADC rail 仍只能报告 amplitude/clipping proxy，不宣称硬件饱和真值。

---

# M3 profile/physiology corrections phase 5 / M3 profile 与生理算法修正第 5 阶段

- 时间 / Date：2026-08-15
- 状态 / Status：implemented_pending_reference_tests
- 写入范围 / Write scope：M3 quality、PPG、physiology、scaling、registries、tests 与本记录。
- 流程 / Process：在只读复审发现 profile 与运行参数可分离后，将 PPG 入口改为注册表驱动；补入 timestamp 网格校验、双波长 raw 比例、训练折振幅风险模型、PRV 80% coverage 门和有效通道优先选择。
- 算法 / Algorithm：滤波参数只能由 profile 决定；峰置信度用单调 1-exp(-x) 映射到 [0,1)；时域 PRV 只在 ≥60 s 且 valid-PPI coverage ≥0.80 时输出；无效通道不能靠高 SQI 抢占主通道。
- 结果 / Result：已写入实现和边界测试调用调整；参考测试将在本逻辑批次同步后重新运行。
- 边界 / Boundary：未修改根目录、原始数据、AGENTS.md 或 _agent/。

---

# M3 profile-locked peak and resampling phase 27 / M3 Profile 锁定峰检测与重采样第 27 阶段

- 时间 / Date：2026-08-15
- 状态 / Status：implemented_pending_reference_test
- 流程 / Process：重扫新 preprocessing profiles、公共 API、峰检测与 external resampling 调用点后，把原先只携带 profile 名的两条路径升级为运行时强制绑定。
- 算法 / Algorithm：峰检测只接受 `purpose=peak_detection_input`、future-active、400 Hz、0.4–8 Hz、三阶 SOS、notch-off 的离线或移动 profile；外部重采样只接受 256/500 Hz，并以一个有理数 sample-coordinate 映射同步处理波形、时间、valid mask 与峰标注。
- 结果 / Result：新增 `ExternalResampleResult` 和唯一 future-active facade `resample_external_ppg_to_400`；低层 `resample_poly_explicit` 不再从包级公共 API 暴露。已补正向同步映射、125 Hz 拒绝、错误 peak-purpose 与错误 fs 负例，下一阶段运行完整测试后冻结结果。
- 边界 / Boundary：MIMIC 125 Hz 当前不属于该 PTT/Sim external profile，必须在未来单独登记，不能通过调用参数绕过；valid mask 的 nearest-source 映射和峰事件的 source→target rounding 均显式写入 provenance。

---

# M3 profile-locked peak and resampling tests phase 28 / M3 Profile 锁定峰检测与重采样测试第 28 阶段

- 时间 / Date：2026-08-15
- 状态 / Status：implemented_and_reference_tested
- 流程 / Process：对第 27 阶段新增的 peak-purpose gate、external resampling facade、公共导出和离散索引映射运行完整参考测试。
- 算法 / Algorithm：500→400 Hz 使用 4/5 polyphase 波形重采样；时间按 target grid 构造，valid mask 使用 target→nearest-source，峰标注使用 source→target rounding。峰检测同时验证 profile status、modality、purpose、400 Hz 与冻结滤波合同。
- 结果 / Result：M3 参考测试由 42 增至 44，44/44 通过，0 failures、0 errors、0 skipped。正例确认波形/时间/mask/峰同步，负例确认 125 Hz、motion-purpose profile 和错误峰检测采样率均 fail closed。
- 边界 / Boundary：external facade 当前只登记 PTT 500 Hz 与 Sim 256 Hz；MIMIC 125 Hz 必须未来新建独立 profile，禁止复用本合同。mask 映射用于对齐有效性，不会把 invalid source 样本静默改为 valid。

---

# M3 PTT ECG reference evaluator phase 11 / M3 PTT ECG 参考评价器第 11 阶段

- 时间 / Date：2026-08-15
- 状态 / Status：implemented_pending_tests
- 流程 / Process：新增 training-only PPG transit-delay artifact 与 disjoint evaluation scorecard。
- 算法 / Algorithm：ECG R peak 后 0.05–0.60 s 内一对一匹配 PPG，训练折取 median delay；评价同时报告 raw/corrected F1、timing、PPI/HR error、coverage/failure。
- 结果 / Result：D8 evaluator 实现已保存；固定合成正例和泄漏负例待执行。
- 边界 / Boundary：detector 对 ECG annotations 的成绩不冒充 PPG peak 成绩；正式 PTT 全数据结果留后续 benchmark。

---

# M3 reference report snapshot phase 14 / M3 测试报告快照第 14 阶段

- 时间 / Date：2026-08-15
- 状态 / Status：implemented_and_reference_tested
- 流程 / Process：复扫冻结 preprocessing/physiology registries 与报告生成器，纠正 peak 默认 profile，并扩展测试报告输入快照。
- 算法 / Algorithm：正式报告现在哈希全部 M3 source、test、registry、schema 和 fixture 文件，并对排序后的路径—哈希映射生成单一 snapshot SHA256；同时记录 Python、NumPy、SciPy 与 scikit-learn 版本。
- 结果 / Result：默认 peak profile 与 registry 的 frailty3_peak_ppg_400_offline_v1 对齐；当前 38/38 reference tests 通过。
- 边界 / Boundary：正式 JSON 报告将在 schema/registry/doc 收束完成后一次生成，避免先写出的报告立即陈旧。

---

# 2026-08-15 M3 Reference Test 首轮修正

- 首轮结果：22 项中 20 通过、1 failure、1 error。
- failure 原因：原 fixture 长度使 100 点 gap 占总样本 2.5%，正确触发了 >1% fatal；
  已改为 10,000 点，使 100 点同时验证恰好 1% 和恰好 0.25 秒。
- 数值修正：非有限比例改为直接计数相除，避免 `1-mean` 在 1% 边界的消减误差。
- error 原因：运行环境 NumPy 1.26.4 没有 `np.trapezoid`；改用兼容的 `np.trapz`。
- 未放宽任何冻结阈值；将重跑全部测试并覆盖机器报告。

---

# 2026-08-15 M3 固定 Fixtures 与 Reference Tests

- 新增固定 seed 20260815 的 PPG/IMU 合成真值生成器，使用稳定 NPY 和 SHA manifest。
- 新增异常 gap/flatline、PPG 频响、重采样、fold-only scaler 泄漏哨兵测试。
- 新增 SI 单位等价、causal 分块、无预校准 ESKF、LPF 隔离和 vector jerk 测试。
- 新增双极性、峰事件 recall、PPI 边界、PRV 公式/分层及 RED/IR 语义测试。
- 新增 unittest JSON runner 和 reference-test 算法矩阵图。
- 状态：测试代码已保存；机器结果由本阶段紧接运行并写入。

---

# M3 Sim 256 Hz resampling fixture phase 31 / M3 Sim 256 Hz 重采样 Fixture 第 31 阶段

- 时间 / Date：2026-08-15
- 状态 / Status：test_added_pending_full_run
- 流程 / Process：重扫 external dataset profile 与已有 500→400 正例后，补充 Simultaneous Measurements 的 256→400 独立正例，避免只验证 PTT 500 Hz 分支。
- 算法 / Algorithm：冻结 25/16 polyphase 比例，2,560 个输入样本生成 4,000 个目标样本；来源峰索引 256/512 映射为目标 400/800，完整 valid mask 保持 valid 状态。
- 结果 / Result：独立 Sim route fixture 已加入正式测试源；下一阶段运行全套测试后冻结结果。
- 边界 / Boundary：该测试只验证同步重采样与索引映射，不代表 Sim 数据上的 heartbeat 或 motion 性能。

---

# M3 Sim 256 Hz resampling tests phase 32 / M3 Sim 256 Hz 重采样测试第 32 阶段

- 时间 / Date：2026-08-15
- 状态 / Status：implemented_and_reference_tested
- 流程 / Process：对 PTT 500→400、Sim 256→400、未登记 125 Hz 和 profile-purpose gates 运行完整 M3 参考测试。
- 算法 / Algorithm：Sim 分支固定 `up=25, down=16`；长度、target 时间网格、valid mask 和 peak annotation 使用与 PTT 分支相同的公共 facade 和 provenance schema。
- 结果 / Result：M3 参考测试由 45 增至 46，46/46 通过，0 failures、0 errors、0 skipped；两个已登记 external source rate 均有独立正例。
- 边界 / Boundary：测试证明采样和索引合同一致，不将 Sim 代理指标解释为 ECG-ground-truth heartbeat 性能；后者属于 M4/M6 正式 benchmark。

---

# M3 stateful IMU runtime corrections phase 6 / M3 有状态 IMU runtime 修正第 6 阶段

- 时间 / Date：2026-08-15
- 状态 / Status：implemented_pending_reference_tests
- 流程 / Process：新增注册表驱动的 causal IMU processor，持久保存 20/40 Hz SOS、ESKF、0.3 Hz LPF、跨块 jerk 和 timestamp 边界状态。
- 算法 / Algorithm：EKF 终止 no-estimate 锁存到显式 session reset；bias random-walk 离散噪声含 attitude 与 cross-covariance；公共 sample mask 同时要求 gravity、dynamic acceleration 和 jerk 有限。
- 结果 / Result：新增 chunk parity、profile mismatch、M1 m/s2 单位兼容、合成真值和 no-estimate latch 测试；执行结果待本批次复验。
- 边界 / Boundary：旧 root EKF 仍为 historical reproduction only；未修改 final_v0 外文件。
