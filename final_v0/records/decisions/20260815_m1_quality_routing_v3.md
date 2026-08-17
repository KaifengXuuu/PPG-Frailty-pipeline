# M1-ARCH-003 — SQI-first、可选 Motion 与 low/motion 手动路由

- 日期 / Date：2026-08-15
- 状态 / Status：`user_direction_recorded_contract_defined_waiting_m1_reacceptance`
- 来源 / Source：用户本回话明确修正
- 关系 / Relation：在 quality routing 冲突处取代 M1-ARCH-001 与 M1-ARCH-002；其余 V2 流式、bundle、平台和回退合同继续有效

## 决定 / Decision

1. SQI 是所有通过 validation 窗口的必做一级质量判断，不再与 denoiser 构成平级可选 policy。
2. Motion detector 为可选一级 activity 判断；启用时可与 SQI 同阶段计算，但必须 join 后再路由。关闭时记录 `NOT_EVALUATED`。
3. 只有 `SQI=high` 且为 static/未启用 Motion 时可绕过 denoiser，直接把共同预处理后的信号交给共享 feature extractor。
4. `SQI=low OR motion=motion` 时进入 degraded/motion 分支。
5. degraded/motion 分支只有 `drop` 或 `denoise_then_extract_features` 两种路线；由人工在 experiment run 或 deployment session 开始前冻结，同一 run/session 内不得逐窗改变。
6. invalid/unrecoverable 强制 drop；SQI、启用后的 Motion、denoiser 或 feature extractor 失败均显式返回 no-result，不允许 stale result 或 denoiser failure 后 raw fallback。
7. Motion 与 quality 保持为两个正交轴；29-subject B/R vs S/W 只监督 activity，不作为 SQI 真值。
8. high-quality 与 denoised 路线必须通过 adapter 产生相同 FeatureBlock schema 后才能进入同一个 classifier arm。

## 旧合同状态 / Superseded terms

- `sqi_gate/coarse_denoise` 动作所有者二选一失效；
- `SQI_WEIGHT` 不属于用户冻结的两种 degraded 策略；
- `COARSE_REPLACE` 波形替换语义失效；
- denoiser 只能在一级 join 与 low/motion 判定后启动。

## 验证与后续 / Validation and follow-up

- V3 机器 schema、active routing registry、三档平台示例和状态机 fixture 已定义。
- M4 实现真实 SQI/Motion/denoiser/FeatureBlock adapters。
- M5 完成 29-subject grouped CV 与 confusion matrix。
- M8 使用相同 folds/seeds/candidate set 的 coverage-aware factorial benchmark。
- M9 分别测 high bypass、drop、denoiser worst-case 与真实比例整链。

