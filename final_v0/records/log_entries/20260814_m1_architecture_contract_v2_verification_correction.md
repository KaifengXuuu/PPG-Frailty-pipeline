## 2026-08-14 — M1 V2 验证表述修正与补充语义门

- 修正 / Correction：较早日志中的“JSON Schema 校验”仅指 schema 结构和 registry/config 交叉校验；本机无第三方 Draft 2020-12 引擎，因此该完整项未运行。
- 补充 / Added：新增零第三方依赖语义验证器，覆盖 ok/no-result 状态机、概率和、唯一 action owner、CPU fallback、locked artifacts、threshold 与 bundle path containment。
- 边界 / Boundary：这仍是合同 fixture 验证，不是模型 smoke、真实 artifact hash 检查或硬件 benchmark。
- 写入 / Writes：仅 `final_v0/`；`_agent` 未写。

