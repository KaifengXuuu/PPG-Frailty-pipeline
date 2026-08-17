# M3 fixture manifest contract phase 25 / M3 Fixture 清单合同第 25 阶段

- 时间 / Date：2026-08-15
- 状态 / Status：generator_strengthened_pending_regeneration
- 流程 / Process：重扫 fixture 生成器、现有清单与 `m3.reference_fixture_manifest.v1` 后，补齐可由生成器确定性产生但旧清单未记录的完整性与科学语义字段。
- 算法 / Algorithm：每个 fixture 记录精确字节数、dtype、shape、逐字节 SHA-256 和双语语义；12 列 IMU fixture 显式冻结为 acceleration、gyroscope、gravity truth、dynamic-acceleration truth 四组三轴顺序；manifest 同时记录 schema 与 generator SHA-256。
- 结果 / Result：生成器源码已更新；本阶段尚未重写 fixture 清单，下一独立阶段将重建并验证字节哈希，避免把计划误记成已完成结果。
- 边界 / Boundary：fixture 只用于工程回归与算法真值测试，不构成 Frailty3 临床或真实姿态性能证据。
