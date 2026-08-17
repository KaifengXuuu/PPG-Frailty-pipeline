# `final_v0` 项目收尾工作区 / Project Finalization Workspace

## 目的 / Purpose

本目录是本次项目收尾会话唯一允许写入的位置。根目录代码、原始数据、历史结果、`AGENTS.md` 与 `_agent/` 均保持只读。

This directory is the only writable project location for the current finalization session. Root-level source code, original data, historical outputs, `AGENTS.md`, and `_agent/` remain read-only.

## Start here / 阅读入口

| Package / 包 | Scope / 范围 | Current use / 当前用途 |
|---|---|---|
| [M0 history](M0_history_MA_denoising_detector_HR_feature/README.md) | Motion detector, artifact/rate-recovery, dynamic HR/PPI history and candidate audit | Preserve historical evidence, failed routes, candidate algorithms and benchmark contract. |
| [M1 architecture](M1_end_to_end_architecture_contract/README.md) | End-to-end architecture, mobile/offline boundary, SQI-first routing | Contract evidence; its V3 SQI→direct/drop-or-rate-recovery order is the current routing authority. |
| [M2 data/evaluation](M2_data_manifest_and_evaluation_protocol/README.md) | 29-subject manifest, role semantics, dual fold registry, external synchronized data | Frozen data/split authority reused by V1; no runtime SGKF recreation. |
| [M3 signal algorithms](M3_unified_preprocessing_and_signal_algorithms/README.md) | Preprocessing, QC, IMU gravity, peaks/PPI and engineering verification | Frozen mathematics/evidence migrated into V1 with stricter typed interfaces. |
| [Final Pipeline V1](final_pipeline_v1/README.md) | Isolated implementation of the merged dev0 specification | Runnable engineering package, comparisons, ablations, acceptance gates, reports, and V2 decision registry. |

For current V1 completion and claim boundaries, read [V1 STATUS](final_pipeline_v1/STATUS.md); for copyable commands, read [V1 RUNBOOK](final_pipeline_v1/RUNBOOK.md).

如需判断“已实现”与“已有科学结果”的区别，先读 V1 状态页；如需直接运行指定模块、对照或消融，使用 V1 运行手册。

## 目录职责 / Directory responsibilities

- `records/`：扫描证据、工作记录、方法注册表、输入输出映射及待录入 `_agent` 的草稿。
- `algorithm_diagrams/`：项目总算法流程图和逐脚本算法结构图，统一采用 Markdown + Mermaid。
- `tools/`：只在 `final_v0` 内写入结果的审计与索引工具。
- `M0_history_MA_denoising_detector_HR_feature/`：M0 历史结果、运动检测/去噪/动态 HR 候选路线、五类方法审计、统一 benchmark 合同及其可校验证据快照。
- `M1_end_to_end_architecture_contract/`：M1 端到端数据流、SQI-first 路由、训练/移动端边界与平台配置。
- `M2_data_manifest_and_evaluation_protocol/`：M2 数据 manifest、阶段映射、外部同步数据证据、历史/未来双 fold 注册表和结果溯源合同。
- `M3_unified_preprocessing_and_signal_algorithms/`：M3 统一信号预处理、IMU 重力估计、峰/PPI/PRV 与验证记录。
- `final_pipeline_v1/`：按合并 dev0 规范实现的隔离、可运行、可审计 V1；包含所有并行模块、对照/消融入口、验收证据和 V2 人工确认点。
- `FINAL_V0_TREE.md`：`final_v0` 全部永久文件的树状结构和逐文件说明。

## 强制更新规则 / Mandatory update rule

每次业务代码、报告或结果文件完成保存后，必须同步更新：

1. `records/WORK_LOG.md` 中对应的逻辑操作记录；
2. `FINAL_V0_TREE.md` 中的文件树和文件说明；
3. 受影响的算法流程图或方法记录。

上述记录文档自身的同步更新不再递归触发新一轮记录。

After each logical save of code, reports, or results, the work log, file index, and affected algorithm documentation must be synchronized. Synchronization of those tracking documents does not recursively trigger another update.

## 安全边界 / Safety boundary

- 不复制或记录 `.env` 的值、凭据、密钥或 token。
- 原始输入只记录必要的文件头、结构和脱敏摘要。
- 输出文本可完整读取并计算校验值；二进制输出仅登记名称、类型、大小及必要的格式元数据。
- 所有推断必须明确标记为 `inferred` 或 `待确认`。
