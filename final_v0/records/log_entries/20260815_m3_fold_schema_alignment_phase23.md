# M3 fold schema alignment phase 23 / M3 训练折 Schema 对齐第 23 阶段

- 时间 / Date：2026-08-15
- 状态 / Status：runtime_aligned_pending_schema_regeneration
- 流程 / Process：对 runtime artifact 与 additionalProperties=false schema 做逐字段 diff，移除未声明的 flattened aliases，并补齐 provenance 与 zero-scale mask。
- 算法 / Algorithm：status=locked、fit_scope=training_subjects_only；transformers 为有序数组，包含 method/stage/feature names/float64/center/scale/impute/zero-scale；parameters_sha256 只哈希规范 transformer payload。
- 结果 / Result：runtime 正例与 leakage/scaling 负例均通过，reference tests 42/42；schema agent 正在同步 fold file SHA 与 payload SHA 两个不同字段。
- 边界 / Boundary：M2 fold registry 文件 SHA 为 c80e780d…388c，canonical payload SHA 为 0bca827f…f46，禁止混用。
