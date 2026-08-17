# M1 V2 验证限制与补充语义门

## 1. 为什么需要补充语义门

V2 主验证器已完成 JSON 解析、schema 结构检查、registry/config 交叉引用、动作互斥、provider fallback 和有界 buffer 检查。本机 WSL、系统 Python 与工作区自带 Python 均未安装 `jsonschema`，因此没有运行第三方 Draft 2020-12 引擎，也没有下载或安装依赖。

`M1_CONTRACT_VERIFICATION_V2.json` 已如实记录：

```text
jsonschema_validation = not_installed_structural_checks_only
```

这不是模型或架构失败，但意味着不能把主报告描述为“完整第三方 JSON Schema 验证已完成”。

## 2. JSON Schema 之外必须执行的运行时规则

`tools/validate_m1_v2_semantic_invariants.py` 以零第三方依赖检查：

1. `status=ok` 必须有三类概率、label 与 confidence，且概率和为 1。
2. `invalid_input/insufficient_quality/processing_lag/runtime_error` 必须清空概率、label 和 confidence。
3. `action_owner=sqi` 时 `COARSE_REPLACE=0`；`action_owner=coarse_denoise` 时 `SQI_DROP=SQI_WEIGHT=0`。
4. ONNX Runtime provider chain 必须以 `CPUExecutionProvider` 作为最后回退。
5. `deploy_locked` 必须具有非空、逐文件 hash 的 artifact 列表；threshold 不得为 null/NaN/Inf。
6. artifact path 必须是 bundle 内 POSIX 相对路径；拒绝绝对路径、Windows drive、反斜杠和 `..` traversal。

## 3. 字段语义澄清

- V2 示例中的 `algorithmic_latency_sec=0` 表示“完整注册窗口已经到达后，额外要求的 future look-ahead 为 0”；不表示首次输出不需要收集 30 s 窗口，也不是计算耗时实测。
- per-hop latency 与 `timing_ms` 在 M9 从窗口 ready 时开始计时；initial window fill 单独报告。
- `buffer_duration_sec=40` 是 M1 示例容量，不是 M3 最终值。
- `TARGET_VENDOR_EP_PENDING_M9` 是合同占位符，不是已安装或已选择的 execution provider。

## 4. 仍需后续完成

- 若用户后续允许安装/提供 Draft 2020-12 validator，再补做完整 schema engine 验证。
- M4 的实际 bundle loader 必须复用等价语义门，不能只做 JSON 解析。
- M8/M9 对真实 artifact 执行文件存在、bytes、SHA-256、provider、parity、latency、RSS、功耗与回滚测试。

