# CHANGELOG

状态：confirmed
来源：用户确认后的 `_agent` 整理、git history、代码/数据/结果目录复核
最后手动更新时间：2026-08-18

## 2026-08-18

- 状态：confirmed
- 来源：用户明确“确认录入”；代码检查、CLI dry-run、unittest 和 diff 自审。
- 变更：保留
  `final_v0/final_pipeline_v2/configs/studies/static_line_b_all_models_v2.yaml`
  原 39-case mega-study 不变；新增
  `configs/studies/static_line_b_staged_v2/`，其中 6 个 YAML 组成省算力的
  Static Line B 分阶段模型筛选流程，并附目录级 `README.md`。
- 变更：study catalog 编排新增 `selected_ordinary` 和
  `matched_ensemble_pair` scope；canonical catalog case 可使用空 override；
  matched pair 强制恰好一个 member-0 comparator 和一个 five-member ensemble、
  同 representation 且不得混入额外 override/formal profile。
- 变更：单因素 study 可同步模型运行参数与 `architecture_parameters` provenance，
  使 Logistic、SVM、ExtraTrees 和 ROCKET 的模型专属轴仍保持一个科学因素。
- 变更：终端进度升级为双层刷新显示。总进度以 `case × repeat` 为单位并显示
  elapsed/ETA；子进度显示当前 case/repeat/fold，完成后自动覆盖或收起；
  `jobs=2` ProcessPool 通过 child `executor_events.jsonl` 向父进程回传进度。
- 更新目的：避免原 39-case × 3 profiles × 5 × 5 的 mega-study 成为日常首筛入口，
  同时保持人工可读、可逐阶段运行、无自动 winner propagation 的实验流程。
- 影响：Stage 1 默认仅运行 4 个低成本 representation baseline、1 repeat × 5 folds，
  共 20 outer cells；线路接近时才人工升级到完整 5×5。Stage 2 以后由人工根据前一
  报告裁剪或替换候选。
- 未完成：Stage 5 当前真正可运行的只有 `quality.mode=off` 对
  `diagnostics_only`；supervised SQI routing、motion override、denoiser efficacy、
  formal motion 5×5、A3/A4 runners 仍为 deferred，不得描述为已实现。
- 验证状态：6/6 staged YAML CLI dry-run 通过；ShapeFormer one-repeat/full-5×5
  override dry-run 通过；Stage 1 默认/完整升级 dry-run 通过；study tests 37/37
  通过；原 mega-study 仍展开 39 cases；`git diff --check` 通过；未运行训练、CV
  或正式 sweep。

## 2026-07-26

- 变更：以 2026-06-10 archive handoff 为基线，补录此后 frailty3 代码、
  数据审计、sweep、分析报告和方法决策。
- 变更：重组 `_agent/TODO.md`，新增 P0/P1：
  - 通用 Frailty3 benchmark。
  - 全历史 sweep/grid 整合与严格 Top 5。
  - hierarchical InceptionTime。
  - Base/Motion/Relax 生理特征路线。
  - 统一消融、preprocessing/scaler audit、异步特征融合 audit。
- 变更：将旧 stage2/采样率检查等任务迁移为 completed/superseded，
  保留 final export、dynamic heartbeat、ShapeFormer 等依赖任务。
- 变更：同步更新 `README.md`、`MODULES.md`、`NOTES.md`、
  `docs/decision-log.md`、`PROJECT_STRUCTURE.md` 和 `ROADMAP.md`。
- 影响：计划中的新脚本只记录为 `planned/not implemented`；
  `_agent/arc/PROJECT_HANDOFF.md` 和 `_agent/WRITE_RULES.md` 未修改。
- 验证状态：用户已明确“确认录入”；Markdown diff 与归档边界待本次写入后复核。

## 2026-07-15

- 变更：完成输入数据和 subject/class 结构审计。
- 数据：raw 29 subjects/261 files；静态 `B,R1,R2,R3,R4` 为
  Pre-Frail 9 subjects/45 files、Robust 12/60、Young 8/40。
- 权限：`PPG_Testing_05_01_2026/` 和 `physionet.org/` 为只读；
  `datasets/` 确认为生成/读取 cache。

## 2026-07-13

- 代码提交：`1d860988`。
- 变更：增加 generalization grid、subject-aware samplers、
  Small InceptionTime，并修正 `analyze_sweep.py` 历史 artifact/reference
  completeness 处理。
- 注意：`analyze_sweep.py` 的 config columns 仍未显式覆盖所有新参数，
  默认模型过滤仍需手动包含 `small_inceptiontime`。

## 2026-07-06

- 变更：生成三组 canonical sweep reports：
  - `20260706_0947_overfitting_inceptiontime_small_inceptiontime`
  - `20260706_0947_overfitting_inceptiontime_small_inceptiontime_02`
  - `20260706_0956_overfitting_inceptiontime`
- 原因：分别复核 2026-06-08 holdout sweep、2026-06-30 generalization sweep
  和 2026-06-25 stage1 sweep，并修正 reference completeness。

## 2026-06-30 至 2026-07-01

- 变更：完成
  `20260630_0630_overfitting_sweep_generalization_rank2`。
- 范围：224 个新 configs + 8 个 references，合计 232 configs、
  1160 runs，全部完整。
- 新因素：full/small InceptionTime、epoch、regularization、SQI、
  subject samplers、per-subject window quota 和 train overlap。
- 结果：最佳新 config BA 约 0.581；最佳 overall 为旧 `s1_122`
  reference，BA 约 0.623；仍未达到 0.73，且 train-validation gap 明显。
- 注意：早期 `0617/0618/0619` 目录是 manifest/少量运行，不是完整正式结果。

## 2026-06-25 至 2026-06-29

- 代码提交：`57c42752`。
- 变更：
  - frailty3 活动流程统一 400 Hz、无 resampling。
  - 增加 IMU gravity removal、local Aboy++ file features、morphology。
  - 增加 SQI gating/quality-weighted aggregation。
  - 增加 weighted CE、balanced softmax、focal loss 和 class-weight modes。
  - reference 与其他 configs 使用公平 epoch/no-early-stopping 协议。
- 修复：首次运行出现 `sqi_static` 未定义 `NameError`；当前代码已定义并可完成 sweep。
- 结果：完成
  `20260625_2320_overfitting_sweep_stage1_rank2`，129 configs、
  645 runs，全部完整；top `s1_122` BA 约 0.610。
- 注意：`2231/2241/2300` 是 preliminary/aborted outputs，不作为正式总榜。

## 2026-06-23 至 2026-06-24

- 变更：将原 `_agent/PROJECT_HANDOFF.md` 按职责拆分到活动文档，
  并归档为 `_agent/arc/PROJECT_HANDOFF.md`。
- 目的：避免单一 handoff 过长、职责混杂；后续新增记录写入对应活动文档。
- 影响：archive handoff 只作历史追溯，不继续追加。

## 2026-06-16

- 变更：为 `20260608_1206_overfitting_sweep_stage1_rank2`
  生成正式 `analyze_sweep.py` 报告。
- Canonical output：
  `results_frailty3/_sweep_analyse/20260616_1143_overfitting_inceptiontime`。
- 范围：930 runs、186 complete configs；`1139` 是内容相同的前一份输出。

## 2026-06-10 基线

- 归档 handoff 已记录到
  `20260608_1206_overfitting_sweep_stage1_rank2`：5-fold
  `StratifiedGroupKFold`、fixed epoch、no early stopping，
  930 runs、186 configs。
- 该节点之后的更新以本 changelog 和活动 `_agent` 文档为准；
  archive 中与新记录冲突的实验状态不得覆盖新事实。
