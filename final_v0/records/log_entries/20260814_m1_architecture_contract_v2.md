## 2026-08-14 — M1 V2 最终只读审计补强

- 操作 / Action：以追加式 V2 固化有界流式、窗口坐标/coverage、单一 action owner、完整 artifact hash 与 accelerator→CPU fallback。
- 原因 / Reason：现有文件补丁入口受沙箱读取故障；为避免绕过 `apply_patch` 或静默覆盖，保留 V1 历史并新建权威 V2。
- 新增 / Added：V2 当前状态、代码风险审计、3 schemas、4 registries、3 platform examples、V2 validator、3幅 Mermaid 图、M1-ARCH-002。
- 边界 / Boundary：未修改根代码、未联网、未安装 ONNX Runtime、未运行模型、未写 `_agent`。
- 验证 / Validation：运行 V2 schema/cross-registry validator、V1 package validator、JSON Schema 校验、全局图/扫描/delivery 验证。

