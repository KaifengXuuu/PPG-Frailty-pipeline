# M3 physiology provenance phase 13 / M3 生理结果溯源第 13 阶段

- 时间 / Date：2026-08-15
- 状态 / Status：implemented_and_reference_tested
- 流程 / Process：重新扫描 Peak/PRV dataclass、所有构造点与边界测试后，将算法 ID、profile ID 和 NNI 语义从原因码中分离。
- 算法 / Algorithm：PeakResult 明确保存 m3_peak_corrected_v1、输入 profile 和 hard-valid PPI/no-imputation 语义；HrvResult 保存 PRV 算法及上游 peak/profile provenance。
- 结果 / Result：新增严格 JSON provenance 回归；当前全量 reference tests 为 38/38 通过（本阶段不写正式报告，最终收束时统一更新）。
- 边界 / Boundary：profile/algorithm 字段只表示来源，不表示质量原因；旧实现映射保留在历史 crosswalk。
