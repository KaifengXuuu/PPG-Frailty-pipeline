# M1 当前权威合同 V3 / Current Authoritative Contract V3

## 1. 版本地位 / Authority

- 当前权威版本：`m1.architecture.v3`。
- V3 只替换 V1/V2 中冲突的 quality routing / denoiser routing 语义；V2 的 `m1.signal_input.v2`、有界流式、窗口坐标、bundle 完整性、平台 profile、provider/CPU fallback、分类器输出与 no-result 规则继续有效。
- `README.md`、V1 与 V2 文件均保留为历史证据，不删除、不覆盖。冲突时优先级为 V3 > V2 > V1。
- V3 是架构与机器合同，不表示 M3/M4 实现、模型训练、ONNX smoke、M8 benchmark 或 M9 硬件实测已经完成。

## 2. 冻结的顺序路由 / Frozen sequential routing

```text
validated + versioned-preprocessed window
→ first-stage evidence:
     SQI (required)
     Motion detector (optional; NOT_EVALUATED when disabled)
→ join and route
     invalid / unrecoverable → forced drop
     high SQI and non-motion → return unchanged to the common feature extractor
     low SQI or motion → one manually preselected branch:
         drop
         denoise_then_extract_features
→ common FeatureBlock contract
→ classifier / aggregation / calibration
→ explicit result or explicit no-result
```

冻结含义：

1. SQI 是每个通过 validation 的窗口都必须经过的一级判断，不再是可与 denoiser 替换的 policy。
2. Motion detector 是可选一级判断。启用时可与 SQI 在同一阶段并行计算，但两者必须 join 后才能决定路线；关闭时输出 `NOT_EVALUATED`，不能伪装为 static。
3. Denoiser 不得与 SQI 并行启动。它只处理已经被判为 low-quality 或 motion 的窗口。
4. high-quality “直接 return”是从 quality/denoiser 子系统返回未经过 denoiser 的预处理信号给共享 feature extractor；不是提前终止整个 Frailty pipeline。
5. `drop` 与 `denoise_then_extract_features` 是 run/session 启动前手动冻结的互斥配置臂；不允许逐窗临场选择，也不允许依据 test-fold 表现事后切换。
6. motion 与 low-quality 是两个正交轴：B/R 对 S/W 监督 activity，不是真实 SQI 标签。`high SQI + motion` 仍进入 degraded/motion 分支。

## 3. 状态优先级与失败语义 / Precedence and failures

路由优先级固定为：

```text
invalid
> unrecoverable_quality
> motion
> low_quality
> high_quality
```

- `invalid` 与 `unrecoverable_quality` 强制 drop，不受手动策略影响。
- SQI 失败，或已启用的 Motion detector 失败，返回显式 no-result。
- denoiser 失败不得把 low/motion 原始波形静默回退给 feature extractor。
- drop 是预期 abstention；module failure 是运行失败。两者必须分别统计。

## 4. V3 机器合同导航 / Machine-readable files

| 路径 | 作用 |
|---|---|
| `06_SEQUENTIAL_SQI_MOTION_ROUTING_V3.md` | 顺序状态机、真值表、接口与下游 benchmark 约束 |
| `schemas_v3/pipeline_config_v3.schema.json` | run/session-level 手动路由配置 |
| `schemas_v3/inference_output_v3.schema.json` | 独立 SQI/Motion 轴、动作守恒与 coverage 输出 |
| `registries_v3/quality_routing_registry_v3.json` | 路由、动作、denoiser adapter 与迁移注册 |
| `examples_v3/*.json` | 三档平台的 V3 可替换示例 |
| `tools/validate_m1_contracts_v3.py` | schema/registry/config 交叉验证与 V3 完整性树 |
| `tools/validate_m1_v3_routing_invariants.py` | 状态机、互斥、失败与 coverage 语义测试 |

## 5. 旧字段迁移状态 / Legacy migration

| V1/V2 术语 | V3 状态 | 迁移 |
|---|---|---|
| `quality_strategy.action_mode=sqi_gate` | deprecated | 显式迁移为必做 SQI + `degraded_branch.manual_policy=drop`；语义不是原样兼容 |
| `quality_strategy.action_mode=coarse_denoise` | deprecated | 必须显式迁移为 SQI-first + high-quality bypass + `denoise_then_extract_features` |
| `action_owner=sqi/coarse_denoise` | removed from V3 | 改为 join 后 exactly-one terminal action |
| `SQI_WEIGHT` | not in user-selected V3 route | 不允许作为隐式第三种策略 |
| `COARSE_REPLACE` | removed | 改为 denoiser-assisted FeatureBlock，不声明恢复真实波形 |
| `diagnostic_candidates_may_run_in_parallel` | narrowed | 仅 SQI 与可选 Motion 可在一级并行；denoiser 必须后置 |

## 6. 完成边界 / Completion boundary

| 项目 | 状态 |
|---|---|
| 顺序 SQI + optional Motion 路由 | `defined_v3` |
| high-quality bypass 与 low/motion 手动二选一 | `defined_machine_checked_v3` |
| 独立 quality/activity 轴与 action-count conservation | `defined_machine_checked_v3` |
| V2 输入、流式、bundle、平台合同 | `inherited_active` |
| 实际 SQI/Motion/denoiser/FeatureBlock adapter | `not_started_M4` |
| 29-subject grouped CV 与 confusion matrix | `not_started_M5` |
| 路线 benchmark 与 Frailty 重训 | `not_started_M8` |
| ONNX/真实硬件 | `not_run` |

