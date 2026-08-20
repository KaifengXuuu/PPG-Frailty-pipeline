# Decision Log

## 2026-08-19：V2 报告重聚合、源码闭包与归档采用 fail-closed 边界

- 状态：confirmed
- 来源：用户明确“确认录入”；实现、恢复 bundle、源码可达性和测试复核。
- 背景：同一 held-out OOF 需要同时展示 window-balanced、file-balanced 和 role-balanced
  的 participant 结果；旧 OOF 的 source version 可能是 placeholder；逐搏 diagnostics
  在多层 artifact 中造成大量重复；表面冗余文件不等于可安全删除。
- 决策：
  1. `W`（window equal-weight）、Line A（equal-file）和 Line B（equal-role）仅作为
     同一 held-out OOF 的报告重聚合视角；不把 post-hoc aggregation 描述成另一条训练线。
  2. 只有 case 保存了相应 source evidence 才生成视角。缺少 window OOF 的
     feature/matrix/fusion case 标为 N/A，不从 participant OOF 反推 window 结果。
  3. final-refit 对 OOF source version fail closed：必须是小写 SHA-256 且等于当前完整
     `ppg_frailty` 源码树 hash；旧 placeholder/mismatch OOF 必须重新运行而不能豁免。
  4. diagnostics 默认只持久化聚合摘要、参数、计数、原因和错误；逐搏 pairing rows 与
     `beat_audit` 不落盘。摘要构建失败不得改变 retention、route、predictor 或 prediction。
  5. 源码归档必须有 registry、tests、tools、顶层入口和动态 import 的可达性证据；本轮
     仅归档已证明不可达的 `train/sampling.py` facade，不继续按表面重复删除公共模块。
- 决策原因：同时满足报告完整性、无外层标签泄漏、final-refit 可复现性和 artifact 体积
  控制，并避免为缩短测试时间破坏活动 registry/facade。
- 影响范围：V2 experiment/final-refit、study reporting/recovery、diagnostics persistence、
  historical V1 transition archive。
- 后续追踪：代码稳定后重跑 fresh formal 5×5；另行对齐陈旧协议文档；如需测试提速，
  优先评估 lazy import 与 fixture 复用。

## 2026-08-19：Stage 2 保留 Logistic/CompactCNN，Stage 3 ShapeFormer 分级止损

- 状态：confirmed
- 来源：用户明确“确认录入”；Stage 1/Stage 2 study report、file/participant OOF、
  fold/per-class metrics 和历史 canonical InceptionFull repeat-0 证据只读复核。
- 背景：Stage 2 三个补跑 case 共 15/15 outer cells 全部通过，均覆盖 29 名
  participant 和 145 条 file OOF，canonical Line B 重放与保存的 participant OOF
  一致。Stage 1/2 的 repeat-0 Line B BA/Macro-F1 分别为 Logistic
  `0.4583/0.4504`、Raw CompactCNN `0.4444/0.4328`、Extra Trees
  `0.3565/0.3761`、RBF-SVM `0.3194/0.3369`、Raw InceptionSmall
  `0.3009/0.2877`；历史可比 Raw InceptionFull 400 Hz repeat-0 为
  `0.3565/0.3687`。当前证据均只有一个 repeat，没有 CI/LCB，不能声明最终胜者。
- 决策：
  1. 保留 Logistic 作为当前跨 representation static incumbent，保留
     CompactCNN 作为 Raw matched incumbent。CompactCNN 虽略低于 Logistic，
     但 worst-class F1 `0.3636` 明显高于 Logistic 的 `0.1905`，Raw 路线不淘汰。
  2. Extra Trees、RBF-SVM 和 InceptionSmall 不继续普通模型扩展；InceptionFull
     也不作为当前 Raw 主基线。Extra Trees 的 post-hoc Line A
     `0.4630/0.4733` 只表示 aggregation sensitivity，不是 Line A 重训练结果，
     不进入晋级或 selection。
  3. Stage 3 仅运行 canonical channel-specific OSD/PISD ShapeFormer，并保持与
     其他模型隔离，按 one-cell implementation test → repeat 0 全五折 diagnostic →
     条件式 full 5×5 的顺序人工放量。
  4. One-cell 只验收无 OOM/NaN/Inf/fallback、9 个 shapelets（每类 3 个）、
     outer-train-only discovery、有限归一概率以及完整 OOF/config/model/provenance；
     单 fold 指标不得用于排名。
  5. One-repeat 必须 5/5 folds 通过、覆盖 29 participants/145 files、Line B source
     replay 通过且无类别整体塌缩。省算力筛选带暂定为 Line B BA 约不低于
     `0.404`、Macro-F1 约不低于 `0.393`，即不同时落后 CompactCNN 超过约
     `0.04`；该带只用于探索止损，不是统计显著性阈值。Line A 仅作敏感性展示。
  6. 只有 repeat-0 通过上述门槛，才运行 ShapeFormer full 5×5。若要正式声称
     ShapeFormer 优于普通 Raw 模型，必须给 CompactCNN 补同协议 full 5×5；
     否则只能报告 ShapeFormer 自身重复稳定性。
  7. 当前 Stage 4 Inception ensemble 不自动启动，因为 InceptionSmall/Full 均未
     成为 Raw winner。ShapeFormer 即使晋级，也不得套用 Inception ensemble 配置；
     需要另行登记 matched ShapeFormer ensemble 才能比较。
- 备选方案：直接运行 ShapeFormer full 5×5，或凭 Extra Trees 的 post-hoc Line A
  数值改变 incumbent，均因算力风险或科学语义不成立而未采用。
- 决策原因：Stage 2 已足以停止明显落后的普通模型，但单 repeat 不足以支持最终排名；
  分级稳定性门槛可以用最低预算验证 ShapeFormer fidelity，同时保护 participant split、
  Line B 主协议和后续正式比较的可解释性。
- 影响范围：
  `final_v0/final_pipeline_v2/configs/studies/static_line_b_staged_v2/03_shapeformer_stability_v2.yaml`、
  Stage 3/Stage 4 人工晋级流程、study 双线报告及后续 full 5×5 matched benchmark。
- 后续追踪：执行 ShapeFormer one-cell；通过后运行 repeat 0 全五折并人工裁决。
  报告的 route-role coverage 目前实际 role OOF 为 `B=29, R=29`，但表格按
  `B/R1/R2/R3/R4` 展开而漏显聚合 `R`，该显示问题不影响 BA/F1，但需单独修复。

## 2026-08-18：Static Line B 模型筛选采用六阶段人工晋级流程

- 状态：confirmed
- 来源：用户明确“确认录入”；六个 staged YAML、catalog/schema、runner 和测试复核。
- 背景：原 `static_line_b_all_models_v2.yaml` 一次展开 13 个普通 candidate 的
  39 个 sparse profiles，完整 5×5 需要 975 outer cells，适合保留作完整 mega-study，
  但不适合作为每次模型探索的首个入口。
- 用户需求：保留原 YAML 不动，另建六个配置，先筛 representation，再在晋级线路内
  比模型；ShapeFormer 独立逐级放量；ensemble 只在选中的 Inception route 上运行；
  SQI/motion 和训练超参数均按单因素、分线路比较。
- 决策：
  1. Stage 1 使用 Raw CompactCNN、Feature-vector Logistic、Feature-matrix
     ROCKET/Ridge 和 Fusion Compact 四个低成本代表；默认 repeat 0 的五个 frozen
     folds，共 20 outer cells。结果只作线路筛选；若线路接近，先升级完整 5×5。
  2. Stage 2 由人工删除未晋级 representation 的完整 case block，再对晋级线路内
     的普通模型做完整比较；不实现自动 winner 或跨 YAML 结果依赖。
  3. Stage 3 将 canonical ShapeFormer 与普通模型隔离，按 one-cell → one-repeat →
     full 5×5 人工升级，前一级稳定后才启动后一级。
  4. Stage 4 仅允许已登记的 Raw InceptionFull 或 Matrix Inception matched pair：
     member 0/seed 50042 对固定五成员概率算术平均；若赢家没有 registered matched
     ensemble，则跳过而不是套用其他模型。
  5. Stage 5 当前只执行 static `off` 对非因果 `diagnostics_only`；真实 SQI/motion、
     8/11-channel motion input、EKF/Profile-A、reducer 和 motion override 按顺序规划，
     但在正式 runner 完成前保持 deferred。
  6. Stage 6 使用可重复编辑的单轴模板。每轮人工把上一轮 winner 写入新的 locked
     base，再依次运行 LR、batch、epochs/WD 或 classical model-specific factor；
     禁止构造 LR × batch × epoch 笛卡尔积。
- 备选方案：原 39-case mega-study 继续保留并可单独运行；自动晋级/自动选择器未采用。
- 决策原因：把算力优先投入有竞争力的 representation/model，同时避免单 repeat
  诊断被误写成最终结论、避免跨阶段隐式选择和多因素混杂。
- 影响范围：
  `final_v0/final_pipeline_v2/configs/studies/static_line_b_staged_v2/`、
  `src/ppg_frailty/study/schema.py`、`expand.py` 及相应 study tests/report metadata。
- 后续追踪：每阶段人工记录晋级原因；Stage 5 的 deferred 模块完成正式 runner 和
  no-training/focused validation 后，才能把对应 YAML 从规划说明升级为可执行 comparison。

## 2026-08-18：长实验终端进度采用双层 repeat/fold 显示

- 状态：confirmed
- 来源：用户要求；progress/runner 实现与 4 个专用进度测试、完整 study test 复核。
- 背景：旧进度只显示 case，启动/预处理期间不计时；`jobs=2` 时 child 的 cell 事件
  只写入各 attempt JSONL，主终端可能长时间停在 0/39。
- 决策：总条从加载 plan 前开始计时，以完成 repeat 数作为稳定工作单位，显示
  elapsed 和近似 ETA；子条显示当前 case 的 repeat/fold 并自动覆盖、完成后收起；
  ProcessPool 的 child JSONL 由父进程定期 relay。
- 备选方案：按 fold/member 显示总进度会过细且频繁跳动；只保留 case 级进度无法
  解释单个 5×5 case 的长时间运行，因此均未采用。
- 决策原因：repeat 是论文指标汇总的自然单位，fold 适合作为瞬时细节；两层显示兼顾
  ETA 稳定性和人工观察，同时不把 progress callback 深入科学算法模块。
- 影响范围：`frailty_3class_sweep_v2.py`、`src/ppg_frailty/study/progress.py`、
  `runner.py` 和 `tests/study/test_progress_v2.py`。已启动的旧进程不会热加载新显示。
- 后续追踪：后续若增加模块级进度，只作为同一 transient sub-line 的 detail，
  不改变 repeat 级总进度语义。

## 2026-08-17：`final_pipeline_v3` 冻结为旧门禁版历史快照

- 状态：confirmed
- 背景：用户已将重构前的 V2 完整备份为
  `final_v0/final_pipeline_v3`，随后要求把当前开发方向收敛为清晰、可由人工
  一条命令运行的算法 pipeline。
- 决策：`final_pipeline_v3` 作为原 V2 复杂门禁版本的只读历史拷贝；
  后续简化、算法核对、sweep/ablation、报告与 Dash 工作只写入
  `final_pipeline_v2`。
- 决策原因：保留旧门禁实现供追溯，同时避免其复杂度继续进入当前人工实验流程。
- 影响范围：不得在 V3 中继续开发、生成实验输出或同步 V2 新改动；比较 V2/V3 时
  必须明确 V3 是历史基线而非当前执行版本。

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

## 2026-08-17：V2 收敛为人工一键运行的论文算法 Pipeline，V3 保存门禁版历史

- 状态：confirmed
- 来源：用户明确确认录入；目录只读核验确认 `final_pipeline_v2` 与
  `final_pipeline_v3` 均存在且当前文件数均为 433；排除 `__pycache__`、
  `*.pyc` 和 `.pytest_cache` 后，两树内容聚合 SHA-256 均为
  `81747c7c22d71244d24671820302f13233ecb7e473bce82e40c6a86a34e96d3a`。
- 背景：当前 V2 在终审过程中扩展了复杂的源码、依赖、运行中漂移、
  attestation 和发布门禁，但用户当前首要目标是得到类似历史
  `frailty_3class_overfitting_sweep.py` 的清晰、可维护、无需 agent 监督的
  完整 Frailty3 实验 Pipeline。
- 决策：
  1. `final_pipeline_v3` 作为当前复杂门禁版 V2 的历史拷贝保留，不作为活动开发主线。
  2. `final_pipeline_v2` 重新作为活动主线，优先交付符合 thesis、两份 canonical
     规范和后续人工确认的纯算法模块。
  3. V2 的主要人工入口是一条配置驱动的命令行 Pipeline，可直接运行 grid search、
     单因素 ablation、完整 OOF 评估、图表和总结报告；运行过程不依赖 agent 监督。
  4. 复杂源码漂移、恶意模块遮蔽、artifact attestation、Git clean/branch 等形式化门禁
     暂不作为 V2 的阻塞目标，并从活动 V2 中收敛或移除；基础配置、seed、输入和环境记录
     可保留为普通可重复性信息。
  5. 不得因工程简化而删除 thesis 必需的科学边界：participant-grouped split、
     outer-heldout 隔离、fold-local fitting、统一 folds/seeds、正确聚合、完整 OOF 和
     模块算法合同仍为强制要求。
- 影响范围：`final_v0/final_pipeline_v2/` 的入口脚本、配置、训练编排、报告、可视化
  和 Dash 人工检查应用；`final_v0/final_pipeline_v3/` 仅作历史追溯。
- 后续追踪：先核验 V3 备份完整性和暂停时 V2 半成品，再对照最新版 canonical workflow
  清理门禁代码并实现一键 sweep/ablation/report Pipeline。

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
