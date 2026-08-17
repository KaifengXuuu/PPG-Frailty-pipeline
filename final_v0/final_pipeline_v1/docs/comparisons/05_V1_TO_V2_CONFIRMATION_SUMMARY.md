# V1→V2 confirmation summary / V1→V2 逐项确认摘要

The authoritative detailed list is
[HUMAN_CONFIRMATION_POINTS.md](../../records/v2_decision_points/HUMAN_CONFIRMATION_POINTS.md).
It contains 28 individually status-labelled decisions. A V1 implementation default is not
automatically user-confirmed. Accepted changes will be implemented under the separate
user-requested `final_pipeline_v2` directory with new config/schema identities and affected
paired reruns.

权威清单为上方详细文档。V1 为保证可运行采用的默认值不自动等于用户确认；任何会改变
数据资格、折、阈值、特征、模型身份、聚合或预算的选择，都必须进入 V2 新配置并重跑。

| ID | Status / 状态 | Decision / 决策点 | V1 value or remaining question / V1 值或剩余问题 |
|---|---|---|---|
| V2-001 | Pending / 待确认 | PTT participant split | Provisional grouped five-fold seed 42; non-independent. |
| V2-002 | Pending / 待确认 | PTT RED/IR map | Unresolved; wavelength-dependent external claims disabled. |
| V2-003 | Pending / 待确认 | Class display strings | IDs 0/1/2 unchanged; title-case display plus M2 alias. |
| V2-004 | Pending / 待确认 | Role weights | Equal files within role and equal available roles. |
| V2-005 | Pending / 待确认 | Detailed stage timing | Only B/R/S/W meanings and S/W-before-Relax are confirmed facts; exact timing remains unknown. |
| V2-006 | Partially confirmed / 部分确认 | Device QC thresholds | D1–D4/D6–D8 recommendation set frozen; device rails/ranges/minimum endpoint duration unresolved. |
| V2-007 | **Confirmed / 已确认** | Gravity primary | No-precalibration quaternion ESKF primary; causal 0.3 Hz LPF mandatory comparator. |
| V2-008 | Partially confirmed / 部分确认 | PRV eligibility | Strict V1 duration/count/adjacency contract frozen; shorter exploratory outputs unresolved. |
| V2-009 | Pending / 待确认 | SQI calibration/cuts | Endpoint-separated fixed/train-only logic; final cuts not selected. |
| V2-010 | Pending / 待确认 | Motion detector gate | Optional and disabled without frozen model/threshold. |
| V2-011 | Partially confirmed / 部分确认 | Degraded policy | SQI-first and run-locked drop XOR rate-recovery confirmed; final deployment branch unresolved. |
| V2-012 | Pending / 待确认 | Reference reducer | Identity baseline; six non-identity modules remain comparisons. |
| V2-013 | Pending / 待确认 | BSS eligibility | Known two-channel internal only; external anonymous-pleth policy unresolved. |
| V2-014 | Pending / 待确认 | Feature allowlist | Physiology plus validity only; exploratory additions require preapproval. |
| V2-015 | Partially confirmed / 部分确认 | Dependencies | NumPy/SciPy/scikit-learn/ONNX Runtime allowed; remaining core/optional profiles need formal decision. |
| V2-016 | Pending / 待确认 | ROCKET source | Self-contained NumPy/SciPy; MiniROCKET named ablation. |
| V2-017 | Pending / 待确认 | ShapeFormer discovery | Effect-size experimental patch/attention route, not PISD parity. |
| V2-018 | Pending / 待确认 | Epoch rule | Fixed 50, no outer early stopping. |
| V2-019 | Pending / 待确认 | Raw time scale | 5 s/400 Hz primary; 10 s and 100/160/200 Hz ablations. |
| V2-020 | Pending / 待确认 | Ensemble budget | Wrapper/smoke only until full member OOF is run. |
| V2-021 | Pending / 待确认 | sklearn serialization | Trusted version-pinned joblib plus hashes. |
| V2-022 | Pending / 待确认 | ONNX gate | Runtime dependency allowed; export/mobile parity scope unresolved. |
| V2-023 | Pending / 待确认 | Formal OOF format | Parquet required by V1 formal contract; alternative not selected. |
| V2-024 | Pending / 待确认 | Full benchmark budget | Candidate/epoch/hardware matrix requires authorization. |
| V2-025 | Pending / 待确认 | Independent frailty test | None; current scores must be named OOF validation. |
| V2-026 | Pending / 待确认 | Target platform | Portable CPU contract only; device/latency/RAM/power target unresolved. |
| V2-027 | Pending / 待确认 | TODO-only routes | Hierarchy/recovery/zoo/Top-5 scope deferred. |
| V2-028 | Pending / 待确认 | Direct PPG band | Canonical 0.2–8 Hz versus explicit 0.4–8 Hz alternative/paired ablation. |

## Confirmation and rerun rule / 确认与重跑规则

- Confirm by ID and, for partially confirmed points, answer the explicitly listed remainder.
- Reopening V2-007 or changing a confirmed/frozen subdecision is allowed, but creates a new
  decision event and requires all affected paired cells to rerun.
- V2 never mutates or relabels V1 artifacts. New data eligibility, fold membership, feature
  schema, threshold, aggregation, model identity, or training budget receives a new identity.
- Synthetic contract evidence and real-input integration smoke remain separate from scientific
  Frailty3 or external-PTT benchmark results.
