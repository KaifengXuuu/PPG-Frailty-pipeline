# M3 Sim 256 Hz resampling fixture phase 31 / M3 Sim 256 Hz 重采样 Fixture 第 31 阶段

- 时间 / Date：2026-08-15
- 状态 / Status：test_added_pending_full_run
- 流程 / Process：重扫 external dataset profile 与已有 500→400 正例后，补充 Simultaneous Measurements 的 256→400 独立正例，避免只验证 PTT 500 Hz 分支。
- 算法 / Algorithm：冻结 25/16 polyphase 比例，2,560 个输入样本生成 4,000 个目标样本；来源峰索引 256/512 映射为目标 400/800，完整 valid mask 保持 valid 状态。
- 结果 / Result：独立 Sim route fixture 已加入正式测试源；下一阶段运行全套测试后冻结结果。
- 边界 / Boundary：该测试只验证同步重采样与索引映射，不代表 Sim 数据上的 heartbeat 或 motion 性能。
