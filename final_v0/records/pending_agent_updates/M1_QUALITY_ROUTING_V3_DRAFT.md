# 私有待审草稿：M1 V3 质量路由 / Private pending draft

## 写入边界

- 本文件只保存未来可能写入 `_agent` 的候选内容。
- 未收到用户明确要求时，不展示本文件内容，不写入 `_agent`。
- 真正草拟时须重新读取 `_agent/WRITE_RULES.md` 并按职责拆分。

## 待同步事实

- M1 当前 quality-routing authority 为 `m1.architecture.v3`；V3 只取代 V1/V2 冲突的质量路由语义。
- SQI 必做，Motion detector 可选；两者 join 后才路由，denoiser 只能后置。
- high/non-motion 绕过去噪进入共享 feature extractor。
- low 或 motion 由 run/session 级人工配置互斥选择 drop 或 denoise-then-features。
- Motion 与 signal quality 为正交轴；B/R vs S/W 是 activity supervision。
- invalid/unrecoverable 强制 drop；module failure fail-closed，无 stale result/raw fallback。
- V3 CURRENT 合同验证与 24 项路由语义 fixture 已通过；模型/ONNX/硬件仍未执行。
- M2–M9 必须同步 coverage/no-result、factorial benchmark、FeatureBlock compatibility 与移动 worst-case 路径。

## 建议目标文档

- `_agent/docs/decision-log.md`：记录 M1-ARCH-003。
- `_agent/PROJECT_HANDOFF.md`：更新当前 M1 authority 和暂停点。
- `_agent/TODO.md` / `_agent/ROADMAP.md`：仅在用户要求草拟并确认录入后，修正后续 M4/M8 路由措辞。
- `_agent/CHANGELOG.md`：记录 V3 合同与验证结论。

