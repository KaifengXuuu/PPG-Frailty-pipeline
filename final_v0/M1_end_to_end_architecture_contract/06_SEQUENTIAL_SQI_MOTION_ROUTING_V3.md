# M1 V3 顺序 SQI、可选 Motion 与 Denoiser 路由合同

## 1. 目的 / Purpose

本合同落实用户确认的新顺序：先做 SQI 与可选 Motion detector，再把窗口分流。high-quality 窗口绕过 denoiser；low-quality 或 motion 窗口只执行预先选定的 `drop` 或 `denoise_then_extract_features`。它废止 V1/V2 的 `sqi_gate/coarse_denoise` 动作所有者模型。

## 2. 术语边界 / Semantic boundary

- `quality_state`：`high / low / unrecoverable / unknown`，来自 SQI，不把 activity label 当作质量真值。
- `motion_state`：`static / motion / not_evaluated / unknown`，来自可选 Motion detector。
- `intervention_required`：有效窗口满足 `quality=low OR motion=motion`。
- `high_quality_return`：未经过 denoiser 的 canonical/preprocessed window 返回给共享 feature extractor。
- `denoise_then_extract_features`：选定 denoiser/frontend 处理 degraded/motion 窗口，并通过注册 adapter 产出与 high-quality 路线兼容的 `FeatureBlock`。这不是“恢复真实波形”的声明。
- `manual_policy`：每次 experiment run 或 deployment session 启动前写入配置并计算 config hash；同一 run/session 内不可逐窗更改。

## 3. 执行状态机 / State machine

1. Validation 先判断 input/window 是否有效。invalid 直接 `FORCE_DROP_INVALID`。
2. 对有效窗口完成共同 preprocessing。
3. 对同一个 window ID、右开 sample 边界和时间边界计算 SQI；SQI 必做且只算一次。
4. Motion detector 若启用，可与 SQI 同阶段计算；若关闭则记录 `not_evaluated`。
5. 两路结果 join。启用的 detector 尚未返回前不得进入高质量分支。
6. SQI=unrecoverable 时强制 `FORCE_DROP_UNRECOVERABLE`。
7. 仅当 SQI=high 且 Motion=static/not_evaluated 时，执行 `RETURN_HIGH_QUALITY_TO_FEATURES`。
8. SQI=low 或 Motion=motion 时，读取已冻结的 `manual_policy`：
   - `drop`：产生对应 `POLICY_DROP_*`，不运行 denoiser，也不运行窗口 feature extractor；
   - `denoise_then_extract_features`：只运行一个已注册 denoiser/frontend，再产出统一 FeatureBlock。
9. SQI、启用的 Motion、denoiser 或 feature extractor 失败时输出显式 failure/no-result；禁止 stale value 与 raw fallback。

## 4. 真值表 / Truth table

| Validity | SQI | Motion | 手动策略 | 唯一动作 |
|---|---|---|---|---|
| invalid | any | any | any | `FORCE_DROP_INVALID` |
| valid | unrecoverable | any | any | `FORCE_DROP_UNRECOVERABLE` |
| valid | high | detector disabled | any | raw/preprocessed → shared features |
| valid | high | static | any | raw/preprocessed → shared features |
| valid | high | motion | drop | `POLICY_DROP_MOTION` |
| valid | high | motion | denoise | denoise → shared FeatureBlock |
| valid | low | disabled/static | drop | `POLICY_DROP_LOW_QUALITY` |
| valid | low | disabled/static | denoise | denoise → shared FeatureBlock |
| valid | low | motion | drop | `POLICY_DROP_LOW_QUALITY_AND_MOTION` |
| valid | low | motion | denoise | denoise → shared FeatureBlock |
| valid | unknown | any | any | `FAILURE_NO_RESULT` |
| valid | high/low | enabled detector unknown | any | `FAILURE_NO_RESULT` |

## 5. 稳定 FeatureBlock 边界 / Stable feature boundary

high-quality 与 denoised 路线可以采用不同内部算法，但进入同一 classifier arm 前必须满足：

- 相同 `feature_schema_id`、列名/通道序、dtype、单位和 mask 语义；
- 相同 window ID、sample/time bounds 与 `available_at_sec`；
- 明确 `signal_source=preprocessed_raw | denoiser_features | none`；
- high-quality route 的 `denoiser_executed=false`；
- drop route 的 features 必须为 null，不允许零向量伪装；
- denoiser 失败不得切回 raw 生成 FeatureBlock；
- route/SQI/Motion 元数据默认只作审计信息；若作为 Frailty 输入，必须另设预先冻结的消融臂。

V3 三档示例使用 heartbeat physiology FeatureBlock，以便四条 denoiser 路线对 HR/PPI/HRV 采用同一输出合同。序列模型、ROCKET 或其他 classifier 仍可沿用 V2 registry，但必须先登记 raw 与 denoised 两路的兼容 adapter。

## 6. 计数与 coverage 守恒 / Conservation

每次输出至少满足：

```text
scheduled_window_count
= FORCE_DROP_INVALID
+ FORCE_DROP_UNRECOVERABLE
+ RETURN_HIGH_QUALITY_TO_FEATURES
+ all POLICY_DROP_*
+ all DENOISE_*_THEN_EXTRACT successes
+ FAILURE_NO_RESULT
```

同时分别报告：

- window coverage；
- 去除 overlap 后按时间区间并集计算的 time coverage；
- HR/PPI event coverage；
- B/R/S/W stage coverage；
- subject coverage；
- S/W→Relax recovery-transition feature coverage；
- policy drop、denoiser failure、feature failure与 subject no-result rate。

drop 是预期 abstention，不能和 failure 合并；总体分母在路由前冻结，不能删掉困难窗口后重算。

## 7. Benchmark 冻结 / Benchmark freeze

M8 最小 factorial arm：

| Motion detector | low/motion policy |
|---|---|
| disabled | drop |
| disabled | denoise_then_extract_features |
| enabled | drop |
| enabled | denoise_then_extract_features |

denoise arm 内再比较 spectral tracking、dual-wavelength BSS、non-stationary decomposition、adaptive filtering。所有 arm 使用相同 subject-level folds、seeds、candidate-window hash、SQI 定义和训练内阈值；policy 必须在运行前冻结。最终选择先满足最低 subject/stage/dynamic coverage 与最大 no-result 约束，再比较同 folds/seeds 的 Frailty balanced accuracy，最后用 latency/RAM/bundle/power 打破平局。

## 8. 下游 TODO 影响 / Downstream impact

- M2：manifest 区分 B/R/S/W activity truth、PTT-PPG peak truth 与派生 route state。
- M3：四条路线共享基础 preprocessing、窗口坐标、单位、mask 和时间轴。
- M4.1：独立保存 quality/activity 轴、candidate reason、selected policy 与 terminal action。
- M4.2/M4.3：drop 产出合法 null；denoiser/heartbeat adapter 产出同一 HR/PPI/HRV FeatureBlock。
- M4.4：drop 后不得拼接跨 gap 波形，须有 gap/missing mask。
- M5：29-subject B/R vs S/W grouped CV 单独报告 confusion matrix 及 routing-weighted FP/FN 成本。
- M6–M7：每个 arm 的 scaler/imputer 只在其 training fold 拟合；stage/subject aggregation 保留缺失与 coverage。
- M8：同时报告 BA、macro-F1、HR/PPI error、risk–coverage、no-result 与 paired fold/seed 差异。
- M9：分别 benchmark high-quality bypass、drop、denoiser worst-case 与真实比例加权整链。

## 9. 迁移门 / Migration gate

- V1/V2 `sqi_gate` 只能作为“可能迁移为 drop arm”的历史提示，不自动升级。
- V1/V2 `coarse_denoise` 因缺少 SQI-first/high-quality bypass 语义，必须人工显式迁移。
- V3 config、output 与 registry 禁止出现 `SQI_WEIGHT`、`COARSE_REPLACE` 或 `action_owner`。
- 同一次实验不得同时提交 V2 与 V3 路由结果作为同一合同版本。

