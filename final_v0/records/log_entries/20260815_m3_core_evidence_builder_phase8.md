# M3 core evidence builder phase 8 / M3 核心证据构建第 8 阶段

- 时间 / Date：2026-08-15
- 状态 / Status：implemented_pending_execution
- 流程 / Process：新增根源码 hash/crosswalk、PPG 频响、M2 完整性绑定和 EKF/LPF 合成真值比较的确定性构建器。
- 算法 / Algorithm：两条 IMU 路线使用同一 fixture、同一单位与同一 causal 前端；仅 profile ID 不同；无 silent fallback。
- 结果 / Result：构建器已保存，输出将在执行后写入 M3 evidence 与 M3_BUILD_REPORT。
- 边界 / Boundary：根源码和 M2 manifest 只读；输出仅在 final_v0。
