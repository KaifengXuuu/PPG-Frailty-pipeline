# README

状态：confirmed
来源：用户要求、`AGENTS.md`、`_agent/WRITE_RULES.md`、`_agent/arc/PROJECT_HANDOFF.md`、2026-06-10 后代码与结果复核
最后手动更新时间：2026-07-26

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

- 数据基线为 400 Hz，不在当前 frailty3 主流程中重采样；原始输入目录已设为只读。
- frailty3 三分类为 `Pre-Frail / Robust-Non-Frail / Young`，当前主评估协议为
  subject-level 5-fold `StratifiedGroupKFold`、固定 epoch、no early stopping。
- 当前优先目标不是继续无边界扩展 grid，而是建立统一 benchmark、整合所有可比
  sweep、完成消融，并在同一协议下选择可复现的 Top 5 配置。
- 两条待验证建模路线为：flat InceptionTime 的两层二分类版本，以及
  Base/Motion/Relax 分阶段生理特征模型。
- 静态 PPG 的 Aboy++、PPI、HRV、morphology 和 IMU gravity removal 已进入
  frailty3 实验代码；动态 heartbeat / IBI / HRV extraction 仍需独立验证。
- 旧 dynamic clean-waveform denoising 路线已失败，只保留为
  deprecated/reference；“半去噪”新路线只以可靠 peak/PPI/HR 为目标。
- 2026-06-25 和 2026-06-30 sweep 仍显示明显 train-validation gap；当前可比结果
  未达到 balanced accuracy 0.73，不能把单次或跨协议最高分作为最终模型能力。
