# M3 Sim 256 Hz resampling tests phase 32 / M3 Sim 256 Hz 重采样测试第 32 阶段

- 时间 / Date：2026-08-15
- 状态 / Status：implemented_and_reference_tested
- 流程 / Process：对 PTT 500→400、Sim 256→400、未登记 125 Hz 和 profile-purpose gates 运行完整 M3 参考测试。
- 算法 / Algorithm：Sim 分支固定 `up=25, down=16`；长度、target 时间网格、valid mask 和 peak annotation 使用与 PTT 分支相同的公共 facade 和 provenance schema。
- 结果 / Result：M3 参考测试由 45 增至 46，46/46 通过，0 failures、0 errors、0 skipped；两个已登记 external source rate 均有独立正例。
- 边界 / Boundary：测试证明采样和索引合同一致，不将 Sim 代理指标解释为 ECG-ground-truth heartbeat 性能；后者属于 M4/M6 正式 benchmark。
