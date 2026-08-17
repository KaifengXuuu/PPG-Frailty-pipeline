# ADR-001: Canonical experiment entry point / 规范实验入口

- 状态 / Status: accepted_for_v2
- 依据 / Source: merged dev0 contract §§1, 4, 5.12, 8.2, 10
- 日期 / Date: 2026-08-15

## Decision / 决策

唯一规范入口为 `python -m ppg_frailty.cli`。入口读取显式配置、manifest 和冻结
split，不扫描目录猜配置，不从历史 leaderboard 短行重建实验。命令至少提供
已实现并注册的V2子命令。不存在的旧命令名不得作为当前能力声明。

The sole canonical entry point is `python -m ppg_frailty.cli`. It consumes an
explicit configuration, manifest, and frozen split; it never guesses configuration
from directories or reconstructs a run from a shortened leaderboard row.

## Consequences / 影响

- 历史脚本只作历史 provenance，不是活动入口。
- 每次运行必须生成完整 `run_manifest.json` 和 config hash。
- 缺少必填 artifact、未知字段或 provenance 不一致时 fail closed。
