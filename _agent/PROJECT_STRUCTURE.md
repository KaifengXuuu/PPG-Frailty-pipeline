# PROJECT_STRUCTURE

状态：confirmed
来源：`_agent/arc/PROJECT_HANDOFF.md`、代码/数据/结果目录检查、用户说明
最后手动更新时间：2026-07-26

| 路径 | 类型 | 内容描述 | 状态 | 来源 | 最后手动更新时间 |
|---|---|---|---|---|---|
| `AGENTS.md` | file | 项目级 agent 长期规则和铁则。 | confirmed | 用户确认 | 2026-07-26 |
| `_agent/` | dir | 跨 chat 项目记录、模块状态、待办、决策和交接文档。 | confirmed | 用户确认 | 2026-07-26 |
| `_agent/README.md` | file | `_agent` 入口、阅读顺序和当前接手重点。 | confirmed | 本次录入 | 2026-07-26 |
| `_agent/WRITE_RULES.md` | file | `_agent` 写入规范、职责划分和归档规则；本次未修改。 | existing | 文件检查 | 2026-07-26 |
| `_agent/MODULES.md` | file | 核心模块、脚本、函数、算法、实验结果和状态。 | confirmed | 本次录入 | 2026-07-26 |
| `_agent/TODO.md` | file | P0/P1 可执行任务、验收条件、依赖和历史状态迁移。 | confirmed | 本次录入 | 2026-07-26 |
| `_agent/ROADMAP.md` | file | benchmark、两条新模型路线、消融和最终导出的阶段关系。 | confirmed | 本次录入 | 2026-07-26 |
| `_agent/NOTES.md` | file | 观察、风险、用户偏好和待验证假设。 | confirmed | 本次录入 | 2026-07-26 |
| `_agent/docs/decision-log.md` | file | 已定案的重要技术和流程决策。 | confirmed | 本次录入 | 2026-07-26 |
| `_agent/CHANGELOG.md` | file | 已发生的重要文档、代码和实验里程碑。 | confirmed | 本次录入 | 2026-07-26 |
| `_agent/arc/PROJECT_HANDOFF.md` | archived file | 截止约 2026-06-10 的原始 handoff；只作历史追溯，本次未修改。 | archived | 历史归档 | 2026-06-10 |
| `PPG_Testing_05_01_2026/` | readonly raw dir | Frailty 标签、StudyData 和 Young 数据源；不得作为输出目录。 | confirmed/readonly | 权限与数据检查 | 2026-07-15 |
| `physionet.org/` | readonly raw dir | 外部生理数据来源；不得被训练脚本原地修改。 | confirmed/readonly | 权限检查 | 2026-07-15 |
| `datasets/` | cache dir | frailty3 manifest/features 等生成与读取 cache，不是 raw input source。 | confirmed | 代码与目录检查 | 2026-07-26 |
| `datasets/frailty3_features_v2_aboy_morph_gravity_B_R1_R2_R3_R4_fs400_w10_h5.csv` | generated cache | 400 Hz、Aboy++/morphology/gravity-removal 静态 file features；145 files、29 subjects。 | confirmed | 数据检查 | 2026-07-26 |
| `funcs.py` | file | PPG preprocessing、Aboy++ peak/HR/HRV、IMU motion helper 等函数集合。 | confirmed | 代码检查 | 2026-07-26 |
| `ppg.py` | file | 旧交互式/可视化入口；frailty3 未直接调用其完整预处理/Aboy++ pipeline。 | legacy/reference | 用户评价 + 代码检查 | 2026-07-26 |
| `ppg_analyse4_calib.ipynb` | file | 当前主分析 notebook；计划接入 motion detector、dynamic heartbeat 和阶段特征。 | active/pending integration | handoff | 2026-07-26 |
| `ppg_peak_hr_gating_train.py` | file | 动态 peak/IBI/HRV extraction 与 motion detector benchmark 主脚本。 | implemented, validation pending | handoff + 代码检查 | 2026-07-26 |
| `frailty_3class_classifier.py` | file | frailty3 数据读取、400 Hz window、模型训练/CV、manual fusion、SQI 和报告核心。 | active | 代码检查 | 2026-07-26 |
| `frailty_3class_overfitting_sweep.py` | file | 5-fold fixed-epoch/no-early-stopping regularization/generalization sweep。 | active | 代码检查 | 2026-07-26 |
| `analyze_sweep.py` | file | 单个/多个输入目录的 config-level leaderboard、class summary、CM 和图表。 | active; config schema improvement pending | 代码检查 | 2026-07-26 |
| `frailty_3class_holdout_eval.py` | file | train/inner-validation/test holdout 复核脚本；非当前主 CV 协议。 | available/reference | 代码检查 | 2026-07-26 |
| `frailty_3class_cnn_fusion.py` | file | 早期 CNN/manual-feature fusion 旁支，核心思路已由主脚本吸收。 | legacy/reference | handoff | 2026-07-26 |
| `shapeformer_port.py` | file | ShapeFormer/ShapeFormer-PISD 移植模块。 | available; low-priority ablation | handoff + 代码检查 | 2026-07-26 |
| `frailty_3class_benchmark.py` | planned file | 统一 manifest/folds/protocol/model wrappers/metrics 的 Frailty3 benchmark。 | not implemented | 2026-07-26 approved plan | 2026-07-26 |
| `analyze_all_frailty_experiments.py` | planned file | 跨 sweep metadata normalization、paired analysis、消融和严格 Top 5。 | not implemented | 2026-07-26 approved plan | 2026-07-26 |
| `frailty_3class_hierarchical.py` | planned file | Young-vs-Old 后 Pre-Frail-vs-Robust 的两层 InceptionTime。 | not implemented | 2026-07-26 approved plan | 2026-07-26 |
| `asa_classifier.py` | file | VitalDB ASA 旁支分类实验，不属于 frailty 主线。 | side experiment | handoff + 代码检查 | 2026-07-26 |
| `pttppg_denoiser_hybrid_*.py` | file group | 旧 dynamic denoising 训练、预览、A/B、ONNX 和 runtime。 | deprecated/reference | handoff + 用户评价 | 2026-07-26 |
| `pttppg_stage2_denoiser.py` | file | 旧 denoiser 相关实验。 | deprecated/reference | handoff | 2026-07-26 |
| `pttppg_pipeline_v7_4_noleak_viz_ae.py` | file | 旧 denoiser/AE 思路来源。 | deprecated/reference | handoff | 2026-07-26 |
| `pttppg_detector_v8_scores.py` | file | 旧 detector scoring，部分版本可能位于 `Arc/`。 | deprecated/inferred | git 状态 + handoff | 2026-07-26 |
| `pttppg_detector_v8_scores_audit_fix9.py` | file | 旧 detector audit/fix baseline。 | reference | handoff + 代码检查 | 2026-07-26 |
| `results_frailty3/` | output dir | frailty3 sweep、holdout、overfitting 和报告输出根目录。 | active | 目录检查 | 2026-07-26 |
| `results_frailty3/_sweep_analyse/` | output dir | `analyze_sweep.py` 的时间戳分析报告。 | active | 目录检查 | 2026-07-26 |
| `results_frailty3/_holdout_eval/` | output dir | strict holdout 复核输出。 | reference | 目录检查 | 2026-07-26 |
| `results_frailty3/_overfitting_sweep/20260608_1206_overfitting_sweep_stage1_rank2/` | result dir | 930 runs、186 configs 的 fixed-epoch stage1 baseline。 | complete | 结果检查 | 2026-07-26 |
| `results_frailty3/_overfitting_sweep/20260625_2320_overfitting_sweep_stage1_rank2/` | result dir | SQI/manual/loss/class-weight sweep；645 runs、129 configs。 | complete | 结果检查 | 2026-07-26 |
| `results_frailty3/_overfitting_sweep/20260630_0630_overfitting_sweep_generalization_rank2/` | result dir | model/sampler/quota/overlap generalization sweep；1160 runs、232 configs。 | complete | 结果检查 | 2026-07-26 |
| `.CNN_results/` | output dir | `ppg_peak_hr_gating_train.py` 动态 peak/IBI/detector 输出。 | active/pending full run | handoff | 2026-07-26 |
| `results_hybrid_denoiser_raw_imu/` | output dir | old denoiser A 输出。 | deprecated | handoff | 2026-07-26 |
| `results_hybrid_denoiser_raw_imu_baseline/` | output dir | old denoiser B 输出。 | deprecated | handoff | 2026-07-26 |
| `denoiser_preview_output/` | output dir | old denoiser preview 图。 | deprecated | handoff | 2026-07-26 |
| `test_asa_classifier/` | output dir | ASA 旁支实验输出。 | side experiment | handoff | 2026-07-26 |
