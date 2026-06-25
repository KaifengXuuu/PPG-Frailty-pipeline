# PROJECT_STRUCTURE

状态：draft  
来源：`PROJECT_HANDOFF.md`、代码结构检查、用户说明  
最后手动更新时间：2026-06-23

| 路径 | 类型 | 内容描述 | 状态 | 来源 | 最后手动更新时间 |
|---|---|---|---|---|---|
| `AGENTS.md` | file | 项目级 agent 长期规则和铁则。 | confirmed | 用户确认 | 2026-06-23 |
| `_agent/` | dir | 跨 chat 项目记录、模块状态、待办、决策和交接文档。 | confirmed | 用户确认 | 2026-06-23 |
| `_agent/README.md` | file | `_agent` 入口说明和阅读顺序。 | draft | 本次整理草稿 | 2026-06-23 |
| `_agent/WRITE_RULES.md` | file | `_agent` 写入规范、职责划分和归档规则。 | draft | 本次整理草稿 | 2026-06-23 |
| `_agent/MODULES.md` | file | 核心模块、脚本、函数、算法和状态。 | draft | 本次整理草稿 | 2026-06-23 |
| `_agent/TODO.md` | file | 未完成任务与对应脚本。 | draft | 本次整理草稿 | 2026-06-23 |
| `_agent/ROADMAP.md` | file | 项目路线演化和中长期目标。 | draft | 本次整理草稿 | 2026-06-23 |
| `_agent/NOTES.md` | file | 观察、风险、用户偏好和待验证想法。 | draft | 本次整理草稿 | 2026-06-23 |
| `_agent/docs/decision-log.md` | file | 重要决策和原因。 | draft | 本次整理草稿 | 2026-06-23 |
| `_agent/CHANGELOG.md` | file | 已发生的重要文档/流程变更。 | draft | 本次整理草稿 | 2026-06-23 |
| `_agent/arc/` | dir | 归档旧 handoff 或旧交接文件。 | draft | 用户要求 | 2026-06-23 |
| `_agent/arc/PROJECT_HANDOFF.md` | file | 原 `_agent/PROJECT_HANDOFF.md` 归档副本/移动后文件。 | draft | 用户要求 | 2026-06-23 |
| `funcs.py` | file | PPG preprocessing、Aboy++ peak/HR/HRV、IMU motion helper 等函数集合。 | confirmed | 代码检查 | 2026-06-23 |
| `ppg.py` | file | 旧交互式/可视化入口，已过时，仅适合抽取旧函数。 | confirmed | 用户评价 + handoff | 2026-06-23 |
| `ppg_analyse4_calib.ipynb` | file | 当前主分析 notebook，应整合 motion detector、dynamic heartbeat extractor 和静态 PPG 分析。 | confirmed | handoff | 2026-06-23 |
| `ppg_peak_hr_gating_train.py` | file | 动态 peak/IBI/HRV 提取和 motion detector benchmark 主脚本。 | confirmed | handoff + 代码检查 | 2026-06-23 |
| `frailty_3class_classifier.py` | file | frailty3 主训练、CV、sweep、模型分支和 feature fusion 脚本。 | confirmed | handoff + 代码检查 | 2026-06-23 |
| `frailty_3class_overfitting_sweep.py` | file | InceptionTime overfitting/regularization stage1/stage2 sweep 脚本。 | confirmed | handoff + 代码检查 | 2026-06-23 |
| `analyze_sweep.py` | file | frailty3 sweep 结果后处理、config-level leaderboard 和图表输出。 | confirmed | handoff + 代码检查 | 2026-06-23 |
| `frailty_3class_holdout_eval.py` | file | top config strict train/inner-val/test holdout 复核脚本。 | confirmed | handoff + 代码检查 | 2026-06-23 |
| `frailty_3class_cnn_fusion.py` | file | 早期 CNN/manual feature fusion 旁支脚本，核心思路已被主脚本吸收。 | confirmed | handoff | 2026-06-23 |
| `shapeformer_port.py` | file | ShapeFormer/ShapeFormer-PISD 移植模块。 | confirmed | handoff + 代码检查 | 2026-06-23 |
| `asa_classifier.py` | file | VitalDB ASA 旁支分类实验脚本，不属于 frailty 主线。 | confirmed | handoff + 代码检查 | 2026-06-23 |
| `pttppg_denoiser_hybrid_*.py` | file group | 旧 dynamic denoising 训练、预览、A/B 对比、ONNX 导出和 runtime 脚本。 | deprecated | handoff + 用户评价 | 2026-06-23 |
| `pttppg_stage2_denoiser.py` | file | 旧 denoiser 相关实验脚本。 | deprecated | handoff | 2026-06-23 |
| `pttppg_pipeline_v7_4_noleak_viz_ae.py` | file | 旧 denoiser/AE 思路来源脚本，作为历史参考。 | deprecated | handoff | 2026-06-23 |
| `pttppg_detector_v8_scores.py` | file | 旧 detector scoring 脚本，部分文件可能已移至 `Arc/`。 | deprecated/inferred | git 状态 + handoff | 2026-06-23 |
| `pttppg_detector_v8_scores_audit_fix9.py` | file | 旧 detector audit/fix 脚本，可作为 baseline/audit 参考。 | confirmed | handoff + 代码检查 | 2026-06-23 |
| `results_frailty3/` | dir | frailty3 sweep、holdout、overfitting 输出目录。 | confirmed | handoff | 2026-06-23 |
| `results_frailty3/_sweep_analyse/` | dir | sweep 后处理输出目录。 | confirmed | handoff | 2026-06-23 |
| `results_frailty3/_holdout_eval/` | dir | strict holdout 复核输出目录。 | confirmed | handoff | 2026-06-23 |
| `results_frailty3/_overfitting_sweep/` | dir | overfitting/regularization sweep 输出目录。 | confirmed | handoff | 2026-06-23 |
| `.CNN_results/` | dir | `ppg_peak_hr_gating_train.py` 动态 peak/IBI/detector 输出目录。 | confirmed | handoff | 2026-06-23 |
| `results_hybrid_denoiser_raw_imu/` | dir | old denoiser A 输出目录。 | deprecated | handoff | 2026-06-23 |
| `results_hybrid_denoiser_raw_imu_baseline/` | dir | old denoiser B 输出目录。 | deprecated | handoff | 2026-06-23 |
| `denoiser_preview_output/` | dir | old denoiser preview 图输出目录。 | deprecated | handoff | 2026-06-23 |
| `test_asa_classifier/` | dir | ASA 旁支实验输出。 | confirmed | handoff | 2026-06-23 |
