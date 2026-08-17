## 2026-08-03 — 统一实现与 Benchmark 合同 / Unified implementation and benchmark contract

- 操作 / Action：把五类方法的“可实现”要求固化为公共数据合同、接口、测试先决条件、指标、输出 schema 和验收门。
- 写入 / Written：`M0_history_MA_denoising_detector_HR_feature/04_UNIFIED_IMPLEMENTATION_AND_BENCHMARK_CONTRACT.md`。
- 流程 / Process：先定义 record/subject 边界与标签语义，再定义 nested subject split、公共基线、unit/synthetic/leakage/holdout/external/runtime 六级测试，最后定义路线 G0–G6 gate。
- 算法 / Algorithm：Adaptive、decomposition、spectral HR、BSS、SQI 均具有可编码接口；spectral+Viterbi 和新版SQI被排在优先实现顺序前端。
- 结果 / Result：任何路线都必须与 raw、bandpass、high-quality-only 比较，允许 missing/reject；subject 是统计单位；有合法 clean truth 才报告 waveform recovery。
- 人工决策 / Human gates：外部文献、第三方依赖、PTT双通道语义、motion目标定义和HR-error/coverage效用须在实际编码前询问。
- 状态 / Status：合同完成；尚未创建提议的实现目录、测试或新 benchmark 结果。
