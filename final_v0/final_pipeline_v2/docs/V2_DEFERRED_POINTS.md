# V2 搁置与未完成事项

## 1. 文档目的与状态术语

本文只列出当前 `final_pipeline_v2` 中尚未完成、尚未执行或仍需人工决定的事项。
它不是功能宣传页，也不会把“代码已存在”写成“科学证据已完成”。状态以
2026-08-20 的代码、配置、测试和人工确认内容为准。

权威输入按以下优先级解释：

1. 用户在 V2 对话中的明确人工确认；
2. `AA_TODO/workflow/CODEX_CANONICAL_PIPELINE_WORKFLOW_V1.md`；
3. `AA_TODO/3/CODEX_IMPLEMENTATION_SPEC_PPG_FRAILTY_DEV0_MERGED.md`；
4. V2 当前已物化的 config、decision profile 和实现。

本文使用四种状态：

- **软件已实现，科学未运行**：模块和测试存在，但没有真实 5×5 或外部实验结果；
- **已实现但默认关闭**：具名 ablation 或 diagnostic，不属于 reference；
- **搁置待证据**：模块可以运行，但尚不能据此作出科学优越性或部署结论；
- **搁置待人工决定**：代码不能替代人确定阈值、硬件或研究范围。

V2 pipeline 不使用 readiness、Phase-0、C0、source-lock 或 evidence gate 决定普通训练能否
执行。参数范围、模块兼容性、未知字段、缺依赖和 outer-fold 隔离仍会 fail-fast；这些是
防止配置被接受却不执行的输入合同，不是算法授权门禁。

## 2. 已经解决、不能再列为 pending 的事项

| 事项 | 当前冻结结论 |
|---|---|
| Frailty raw 输入 | `RED, IR, A_dyn_X, A_dyn_Y, A_dyn_Z, GX, GY, GZ`，严格 8 通道 |
| Motion 输入 | 8 通道为 thesis reference；增加 `A_mag, Omega_mag, J_mag` 的 11 通道仅是 augmentation ablation |
| Outer split seeds | `[42, 10042, 20042, 30042, 40042]`，只控制 participant fold generation |
| Named five-member preset | `[50042, 60042, 70042, 80042, 90042]` 只冻结具名历史 comparison；普通 V2 ensemble 接受任意非空、唯一的显式 uint32 seed roster |
| Named member-0 comparator | 具名 comparison 使用 seed `50042`；普通 single model 使用所选配置的 seed policy |
| Ensemble 概率 | 每个 outer fold 内对全部已配置成员 probability 做算术平均；不挑最佳 member，不平均 member metrics |
| Final ensemble refit | 在全部 29 人上从头训练，并继承人工选中配置的 single seed 或 ensemble roster |
| PTT acceleration 单位 | V2-036：external sit acceleration 按源 `m/s²` identity 使用；禁止再乘 `9.80665` |
| Aura 版本 | `hrv-analysis==1.0.2`，只在隔离可选环境做固定 PPI 函数比较 |
| 旧发布门禁 | tracked-clean/source gate、attestation/prepublish、ONNX winner、exact-lock/acceptance dead tools 已按人工授权移除；V3 保留历史版本 |

## 3. 仍然搁置的科学事项

### 3.1 正式 5×5 实验尚未执行

- **状态**：软件已实现，科学未运行。
- **已有内容**：participant-grouped 5-fold × 5-repeat、outer-fold isolation、OOF、
  role-aware aggregation、报告和 resume/nonoverwrite 输出。
- **尚缺内容**：四个 canonical representation 的真实完整运行、计划中的 ablation、
  ensemble comparison 和统计归档。
- **影响**：任何当前测试通过都只能证明软件合同，不代表 BA、macro-F1 或模型优劣。
- **关闭条件**：人工启动正式 study，保存完整 25 cells、OOF、learning curves、confusion、
  配置快照和报告；不允许把 dry-run 或单 fold smoke 当正式结果。

### 3.2 SQI 阈值、权重与 route 的科学证据

- **状态**：软件已实现、可配置执行；科学比较尚未运行。
- **已有内容**：`off`、`diagnostics_only`、七状态 A1/A2 route state machine、原始 SQI
  components、route/role 报告。
- **reference**：SQI `off`；`diagnostics_only` 不能改变 retained data、aggregation 或 prediction。
- **执行合同**：`quality.mode=route` 直接选择 outer-train-only calibrator 与七状态 route；不存在
  `supervised_route_ready`、artifact hash 或 YAML authorization boolean。
- **尚缺内容**：在相同 frozen folds 上预注册并运行阈值、组合权重、coverage 与 downstream
  OOF 的匹配比较，证明它们是否有益。
- **禁止事项**：依据 outer holdout 调阈值；将 diagnostics 当监督 route 结果；把“代码可运行”
  写成“科学性能已验证”。
- **关闭条件**：保存训练范围、阈值 provenance、完整 repeated grouped OOF 与 coverage 报告。

### 3.2.1 A1/A2 segment 接线与 morphology eligibility

- **状态**：七状态机与普通 runtime 已接线；heterogeneous segment planner 尚未实现。
- **当前差距**：`experiment.py::_route_records` 仍把整条 recording 当成一个 segment，尚未按
  heterogeneous segment 保存 start/end/run identity；`rate_only_direct` 又会被折叠成
  `SignalRoute.DIRECT`。当前 runtime 另行持久化 `shape_features_eligible`，因此
  `rate_only_direct` 不会产生 morphology、amplitude 或 dual-wavelength optical predictors。
- **当前影响**：route 可以按 recording 粒度执行，但不能把结果描述为 heterogeneous
  segment-level routing。
- **关闭条件**：如研究问题需要 segment 粒度，再实现 segment planner、逐 segment
  start/end/run identity 与合法 run 内 pulse/PPI adjacency；保留现有 shape-eligibility 负例。

### 3.2.2 A3/A4 endpoint runner 与 sampling-grid 决定

- **状态**：组件部分存在，规定 runner 和产物未完成。
- **A3 人工决定**：external source 是 500 Hz，但当前 adapter 同步到 400 Hz；运行 A3 前必须人工
  冻结“原生 500 Hz endpoint”或“同步 400 Hz adapter”，不得混用或把二者结果直接比较。
- **A3 尚缺内容**：独立 ECG-reference event matcher runner，以及按 activity 输出 event matches、
  rate predictions、metrics、failure/coverage 和 run manifest；现有 motion activity classifier 不能
  代替该 endpoint benchmark。
- **A4 尚缺内容**：internal dynamic segment routes、Q_rate pre/post、reducer failures、coverage、
  rate agreement 和诊断图的独立 runner/report。
- **关闭条件**：先完成人工 grid 决定与 A1/A2 segment 接线，再实现 identity-control A3/A4；
  reducer winner 和 motion override 只能使用这些证据，不能在证据之前冻结。

### 3.3 Artifact reducer 最终选择

- **状态**：实现或注册，但未通过正式比较选择。
- **reference control**：`identity/no denoise`。
- **计划候选**：NLMS IMU-ANC、SSA、spectral mask、PCA-BSS、FastICA-BSS、NMF-BSS；
  historical EMD/CEEMD/DWT 只保留具名历史 ablation 身份。
- **尚缺内容**：在同一 outer split、同一模型、同一输入和同一聚合下的单因素比较；
  pulse/PPI recovery、coverage、失败率和最终 frailty OOF 指标的联合证据。
- **影响**：当前不能声称任何 reducer 是 final 或 clinical improvement。
- **关闭条件**：完成 artifact-reducer ablation 后人工审阅；未胜出时继续保留 identity reference。

### 3.4 Motion override / SQI+motion 路线

- **状态**：外部 motion-evidence 路径可检查；不作为 core classifier 的授权开关。
- **reference input**：8 通道 axes；11 通道 derived augmentation 只作为匹配 ablation。
- **尚缺内容**：internal 29-person participant-grouped motion evidence、external PTT motion
  benchmark、阈值/校准和 override 对 retained coverage 与 frailty OOF 的影响。
- **影响**：motion score 可以作为 diagnostic 或显式配置的恢复依据；不能在 reference 中静默
  覆盖 SQI、波形或预测，也不能在没有结果时声称有益。
- **关闭条件**：先分别完成 internal 与 PTT 证据，再运行 `SQI-only`、`motion-only`、
  `SQI+motion` 匹配比较；activation 必须是新的人工决定。

### 3.5 PTT external benchmark 尚未运行

- **状态**：单位、channel mapping、source identity 和 grouped 5×5 protocol 已定义；科学未运行。
- **已解决**：ACC 使用 `m/s²` identity；gyro 依据 header 从 `deg/s` 转为 `rad/s`。
- **尚缺内容**：真实 external PTT 运行、每 activity/subject 的完整 OOF、失败与 coverage 报告。
- **证据边界**：该数据仍是 repeated grouped validation，不是独立测试队列。

### 3.6 Device-specific physical QC

- **状态**：搁置待设备信息。
- **已有内容**：schema、finite、时长、constant/flatline 等不依赖设备常数的 recording-level QC。
- **尚缺内容**：ADC/device rail、合法物理量程、clipping/saturation 阈值及其来源文件。
- **影响**：目前不能声称执行了 device saturation/range QC。
- **人工输入**：设备型号、ADC 规格、sensor range、unit、厂商或采集系统证据。

### 3.7 Target hardware 与 deployment measurements

- **状态**：搁置待硬件决定。
- **已有内容**：可记录 parameter count、bundle bytes 和可测的描述性 CPU latency。
- **尚缺内容**：目标 CPU/MCU/GPU、线程数、batch、内存、功耗、runtime、量化策略和允许阈值。
- **影响**：预测性能 leaderboard 与 deployment-readiness 必须分开；未测成本显示 N/A，
  不能伪造为 0，也不能据此自动淘汰科学候选。
- **关闭条件**：先人工冻结目标设备和测量协议，再运行独立 operational benchmark。

### 3.8 Independent test cohort

- **状态**：不存在。
- **当前证据**：29 人 internal 结果和 PTT protocol 都只能标为 OOF/repeated grouped validation。
- **影响**：论文和报告不得使用 independent test、external generalization 或 clinical validation 表述。
- **关闭条件**：取得未参与模型、阈值、ablation 和 candidate selection 的新 participant cohort，
  冻结一次性评估协议后执行。

### 3.9 人工 candidate selection 与 final refit 尚未执行

- **状态**：软件路径存在，科学未执行。
- **已有保护**：manual selection record、OOF/config/dataset/model/file hashes、all-29 roster、
  nonoverwrite atomic write、save/reload golden parity。
- **尚缺内容**：实际 5×5 comparison archive、人工 purpose-specific selection、全 29 人 refit 和 bundle。
- **规则**：不能自动选 winner；不能用 all-29 refit 产生新的内部性能指标；final refit 继承
  被选配置的 single seed 或 ensemble roster。`42` 与五成员 roster 只是 reference/具名比较默认。

### 3.10 高计算量模型和时间尺度 ablation 尚未运行

- **ShapeFormer**：PISD/OSD 与 downstream port 已实现并有 tiny fidelity tests，但完整 fold-local
  discovery 尚未在 25 cells 上运行；不得用 tiny fixture 声称模型性能。
- **V2-019 time-scale**：100/160/200 Hz fixed-kernel-samples、10 s context、dilation=2
  配置已定义或可物化，但尚未正式运行。
- **Epoch 7/15**：具名单因素 ablation，尚无正式结果。
- **影响**：这些项目属于计划实验，不是 reference 已证实的改进。

### 3.11 PRV library comparisons 不进入 classifier

- **状态**：Aura/Rhenan 仅限固定、相同 PPI 输入的函数级 diagnostic comparison。
- **尚未授权**：把 library backend、library cleaning 或其输出替换 canonical local PRV predictor。
- **如未来扩展**：必须新增独立 ablation，并重新检查 feature schema、missingness、outer-fold fit
  和 classifier OOF；当前不是一个可静默打开的开关。

### 3.12 V2-027 todo-only scope

- **状态**：名称已保留，但没有足够、互不矛盾的人工定义可转成正式实验。
- **处理**：不得依据名称自行实现。若需要启用，必须由人工给出科学问题、数据范围、
  reference/control、唯一变化量、成功标准和报告位置。

### 3.13 CPU CI 尚未建立

- **状态**：本地 curated safe suite 已通过，但 repository/V2 仍没有自动 CPU CI workflow。
- **影响**：当前只能写“本地实现合同验证通过”，不能写持续集成已建立或每次提交自动回归。
- **关闭条件**：增加轻量 lint/import/unit/synthetic integration workflow；GPU、正式 5×5、PTT 和
  高计算量 ShapeFormer 继续保持人工或 scheduled，不塞进普通 CPU CI。

## 4. 明确不是“未完成缺陷”的边界

- 11 通道不能进入 frailty raw branch；这是科学边界，不是缺功能。
- Equal-files Line A 与 role-aware Line B 都是普通可选聚合模块；Line B 只是 reference 默认。
- Training balance 与 reporting aggregation 独立配置；报告可以从同一 held-out file OOF 并行
  重放 window-balanced、Line A 和 Line B 三种 participant 视图。
- Reference SQI 为 off、artifact reducer 为 identity，是在监督证据不足时的正式 control。
- 不自动选择 winner 是研究流程要求。
- 没有 independent test 时诚实标为 OOF validation 是正确行为。
- 删除 ONNX/exact-lock/attestation/acceptance gate 是人工确认的 V2 简化，不再列入 backlog；
  若以后需要部署发布体系，应建立新的独立 deployment project，而不是偷偷恢复旧 gate。

## 5. 建议关闭顺序

1. 完成 four-representation single-model references，验证 5×5 OOF 与报告完整性。
2. 运行低风险单因素 ablation：epochs、filter、aggregation、gravity、time-scale。
3. 运行 representation/model comparisons 和已冻结的 five-member ensemble comparison。
4. 若进入 A1/A2/A3/A4，先闭合 segment/morphology eligibility，再由人工冻结 A3 的 500/400 Hz
   endpoint grid，并完成 identity-control A3/A4 runner。
5. 运行 artifact-reducer 与 motion 8/11-channel matched ablation；只有 A3/A4 证据完整后才人工
   选择 reducer 或评估 motion override。
6. 可运行 SQI route/motion 方案，但只有保存 outer-train-only calibration 与完整 OOF 证据后，
   才能作出性能结论。
7. 人工选择 purpose-specific candidate，执行 all-29 final refit。
8. CPU CI、目标硬件、device QC 和 independent cohort 分别作为后续工程/证据阶段处理。

具体 case 顺序、开关与参数见 `V2_ABLATION_TEST_PLAN.md`；逐模块算法差距见
`V2_ALGORITHM_GAP_ANALYSIS.md`。
