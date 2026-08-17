# M3 reference report snapshot phase 14 / M3 测试报告快照第 14 阶段

- 时间 / Date：2026-08-15
- 状态 / Status：implemented_and_reference_tested
- 流程 / Process：复扫冻结 preprocessing/physiology registries 与报告生成器，纠正 peak 默认 profile，并扩展测试报告输入快照。
- 算法 / Algorithm：正式报告现在哈希全部 M3 source、test、registry、schema 和 fixture 文件，并对排序后的路径—哈希映射生成单一 snapshot SHA256；同时记录 Python、NumPy、SciPy 与 scikit-learn 版本。
- 结果 / Result：默认 peak profile 与 registry 的 frailty3_peak_ppg_400_offline_v1 对齐；当前 38/38 reference tests 通过。
- 边界 / Boundary：正式 JSON 报告将在 schema/registry/doc 收束完成后一次生成，避免先写出的报告立即陈旧。
