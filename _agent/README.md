# README

状态：draft  
来源：用户要求、`AGENTS.md`、`_agent/WRITE_RULES.md`、`_agent/PROJECT_HANDOFF.md`  
最后手动更新时间：2026-06-23

## `_agent` 目录用途

`_agent` 用于保存所有 chat/AI agent 的项目交接记录、模块状态、待办任务、路线演化、决策日志和协作规范。其他 chat 应能仅通过根目录 `AGENTS.md` 和 `_agent` 目录理解项目目的、数据结构、核心模块、历史版本、当前进度、已实现内容、待实现内容和后续方向。

## 推荐阅读顺序

1. `../AGENTS.md`
2. `_agent/WRITE_RULES.md`
3. `_agent/README.md`
4. `_agent/MODULES.md`
5. `_agent/TODO.md`
6. `_agent/ROADMAP.md`
7. `_agent/NOTES.md`
8. `_agent/docs/decision-log.md`
9. `_agent/PROJECT_STRUCTURE.md`
10. `_agent/CHANGELOG.md`
11. `_agent/arc/PROJECT_HANDOFF.md`

## 文件职责索引

| 文件 | 职责 |
|---|---|
| `../AGENTS.md` | 所有 chat/AI agent 必须遵守的长期规则。 |
| `_agent/WRITE_RULES.md` | `_agent` 文档系统的职责划分、写入边界和统一格式。 |
| `_agent/README.md` | `_agent` 目录入口说明、文件索引和推荐阅读顺序。 |
| `_agent/MODULES.md` | 核心模块、脚本、函数、算法、输入输出、状态和改进方向。 |
| `_agent/TODO.md` | 明确可执行任务、优先级、涉及脚本、阻塞点和下一步。 |
| `_agent/ROADMAP.md` | 项目中长期路线、阶段目标和主线演化。 |
| `_agent/NOTES.md` | 临时观察、风险、用户偏好、推测和待验证问题。 |
| `_agent/docs/decision-log.md` | 已定案的重要技术和流程决策。 |
| `_agent/PROJECT_STRUCTURE.md` | 项目文件结构、文件内容描述和最后手动更新时间。 |
| `_agent/CHANGELOG.md` | 已发生的重要项目记录变更。 |
| `_agent/arc/PROJECT_HANDOFF.md` | 归档的原始 handoff 文件，仅作历史追溯。 |

## 当前接手重点

当前项目主线是 Python-based PPG signal processing pipeline for frailty classification。重点包括：

- 静态 PPG 预处理、Aboy++ peak detection、PPI、HRV。
- IMU-led motion/static detection。
- 动态 PPG direct heartbeat / IBI / HRV extraction。
- frailty3 三分类模型：`Pre-Frail / Robust-Non-Frail / Young`。
- InceptionTime overfitting sweep、strict holdout 和最终模型导出。
- 旧 dynamic denoising 路线已失败，应作为 deprecated/reference 处理。
