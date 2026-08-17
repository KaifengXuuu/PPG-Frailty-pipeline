## 2026-08-03 — 三问题候选路线目录 / Candidate-route catalog for three problems

- 操作 / Action：新增用户指定的 motion detector、denoising、动态 HR 候选脚本文档。
- 写入 / Written：`M0_history_MA_denoising_detector_HR_feature/02_CANDIDATE_SCRIPTS_MOTION_DENOISING_DYNAMIC_HR.md`。
- 覆盖 / Coverage：5 组 motion 候选、7 组 denoising 候选/失败历史、5 组动态 HR 候选以及共享 SQI 层。
- 每项字段 / Fields：脚本与应用位置、具体算法、输入数据名称/路径、输出路径/结构、已有结果、状态判定、风险和下一步。
- 主要判断 / Decision：P02 Light CNN 为 motion 首选；hybrid 先补生理 holdout；spectral candidate tracking 为动态 HR 新主线；Aboy++、DWT-A2、NLMS 与 legacy detector 均按对照而非成功方案保存。
- 数据边界 / Data boundary：没有新增或改写任何历史结果；只引用已存在的 JSON/CSV/代码事实。
- 验证 / Verification：最终批次将检查文档链接、快照、SHA-256、Mermaid 和全交付一致性。
