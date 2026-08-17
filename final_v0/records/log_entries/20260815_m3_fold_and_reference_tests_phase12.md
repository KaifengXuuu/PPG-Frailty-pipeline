# M3 fold/reference tests phase 12 / M3 训练折与参考评价测试第 12 阶段

- 时间 / Date：2026-08-15
- 状态 / Status：implemented_pending_execution
- 流程 / Process：为 M2 exact roster、OOF 污染、PTT train-only delay 与 evaluation overlap 新增正负例。
- 算法 / Algorithm：fold artifact 使用 payload hash 而非文件 hash；PTT 校正只允许应用到与 delay-training roster 不相交的 subject。
- 结果 / Result：测试已保存，执行结果将写入统一 M3 reference report。
- 边界 / Boundary：未运行正式 PTT benchmark；该测试仅验证合同和公式。
