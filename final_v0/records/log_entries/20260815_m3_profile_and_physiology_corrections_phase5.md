# M3 profile/physiology corrections phase 5 / M3 profile 与生理算法修正第 5 阶段

- 时间 / Date：2026-08-15
- 状态 / Status：implemented_pending_reference_tests
- 写入范围 / Write scope：M3 quality、PPG、physiology、scaling、registries、tests 与本记录。
- 流程 / Process：在只读复审发现 profile 与运行参数可分离后，将 PPG 入口改为注册表驱动；补入 timestamp 网格校验、双波长 raw 比例、训练折振幅风险模型、PRV 80% coverage 门和有效通道优先选择。
- 算法 / Algorithm：滤波参数只能由 profile 决定；峰置信度用单调 1-exp(-x) 映射到 [0,1)；时域 PRV 只在 ≥60 s 且 valid-PPI coverage ≥0.80 时输出；无效通道不能靠高 SQI 抢占主通道。
- 结果 / Result：已写入实现和边界测试调用调整；参考测试将在本逻辑批次同步后重新运行。
- 边界 / Boundary：未修改根目录、原始数据、AGENTS.md 或 _agent/。
