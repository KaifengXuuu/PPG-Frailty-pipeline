# M3 deprecated-profile tests phase 30 / M3 弃用 Profile 测试第 30 阶段

- 时间 / Date：2026-08-15
- 状态 / Status：implemented_and_reference_tested
- 流程 / Process：运行完整 M3 测试验证 future-active PPG profile gate 与 deprecated mobile alias 负例。
- 算法 / Algorithm：运行时不接受 `deprecated_alias` 或 `historical_reproduction_only` 作为 corrected preprocessing profile；profile ID、status、modality、purpose、resampling 和 fs 必须共同匹配。
- 结果 / Result：M3 参考测试由 44 增至 45，45/45 通过，0 failures、0 errors、0 skipped；旧 alias 被确定性拒绝，静态、运动、峰检测与 denoiser future profiles 无回归。
- 边界 / Boundary：兼容 alias 仍可被迁移工具读取，但不能作为未来 benchmark 的执行配置；旧结果复现与 corrected 主协议继续严格分离。
