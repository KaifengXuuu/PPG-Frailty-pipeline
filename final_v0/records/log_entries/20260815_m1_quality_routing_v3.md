## 2026-08-15 — M1 V3 顺序质量路由修订

- 操作 / Action：按用户修正，以追加式 V3 取代 V1/V2 的 SQI/coarse-denoise action-owner 路由；V2 输入、流式、bundle、平台和 provider fallback 继续有效。
- 算法 / Algorithm：必做 SQI + 可选 Motion → join；high/non-motion 绕过 denoiser；low 或 motion 按 run/session 级手动配置互斥执行 drop 或 denoise→FeatureBlock；invalid/unrecoverable 强制 drop，module failure fail-closed。
- 新增 / Added：V3 当前状态与详细合同、config/output schemas、active routing registry、3 platform examples、双语合同/语义验证器、专业 Mermaid 图和 M1-ARCH-003。
- 验证 / Validation：V3 CURRENT contract/cross-registry 3/3 example configs 通过；24/24 routing fixtures 通过；完整结果见 M1 包内机器报告。
- 校正 / Correction：首版 V3 validator 把 legacy migration 元数据中的旧字段名误报为活动字段；保留首版证据，新增 active-registry CURRENT 入口，不静默覆盖。
- 边界 / Boundary：未实现/训练模型，未运行 ONNX 或真实设备 benchmark，未联网、未安装依赖、未改 final_v0 外文件、未写 `_agent`。

