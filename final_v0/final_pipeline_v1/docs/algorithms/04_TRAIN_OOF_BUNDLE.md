# Unified training, OOF, and bundle / 统一训练、OOF 与模型包

```mermaid
sequenceDiagram
    participant C as CLI run-experiment
    participant F as Frozen split registry / 冻结折
    participant S as Signal + SQI route
    participant X as Representation gate / 表征门
    participant T as Unified Trainer / 统一训练器
    participant E as Evaluator / 评估器
    participant O as Fixed OOF writer
    participant B as Separate bundle subsystem

    C->>F: load exact repeat/fold, seeds, train and OOF IDs
    F-->>C: immutable outer roster + hashes
    C->>S: load synchronized records and direct SQI components
    S->>S: fit empirical SQI calibrator on outer-train IDs only
    S-->>C: retained/dropped routes + fitted provenance
    C->>X: representation_mode

    alt feature_vector (current formal executor)
        X-->>T: file vectors + validity + exact train IDs
        Note over T: Fit imputer, scaler and model on outer-train only
        T-->>C: model + immutable fitted-artifact provenance
        F-->>E: held-out inputs and labels only for evaluation
        C->>E: frozen transforms + model
        E->>O: file probabilities + explicit dropped/no-result rows
        O->>O: file → role → participant; preserve complete roster
        O-->>C: run manifest, metrics, confusion, four OOF parquet, result JSON
    else raw / feature_matrix / fusion
        X-->>C: failed_closed (current formal-runner limitation)
        Note over X,C: Modules and comparison/training contracts exist;<br/>no implicit feature-vector substitution
    end

    opt independent bundle workflow / 独立模型包流程
        T->>B: config, schemas, fitted states, hashes, aggregation, golden input
        B->>B: transactional save → load → golden parity
        Note over B: Tested subsystem; not emitted by current experiment runner
    end
```

## Public execution modes / 公共执行模式

| Mode / 模式 | Data budget / 数据预算 | Scope / 范围 |
|---|---|---|
| `reduced-smoke` | Complete frozen roster; 60 s; one record/participant; one epoch-equivalent | `smoke_not_scientific_benchmark` |
| `full` without indices | Complete recordings, all eligible files, configured epoch rule, all 25 cells | Full 5×5 only after all cells complete successfully |
| `full --repeat R --fold F` | One complete, uncapped cell | `selected_full_length_cells_not_complete_5x5` |
| Any unsupported representation | No substitution or shortened roster | Structured `failed_closed` |

The passing public example is:

```bash
python3 -B -m ppg_frailty.cli run-experiment --config configs/motion_benchmark_v1.yaml --budget reduced-smoke --repeat 0 --fold 0 --output-dir artifacts/experiments/my_reduced_r0_f0
```

`reference_static_v1.yaml` is raw and is not a passing current-runner example.

## Train-only and OOF contract / 仅训练折与 OOF 合同

1. Frozen membership selects exact train and held-out participants; runtime SGKF
   recreation is prohibited.
2. Base direct SQI is computed before routing; empirical SQI bounds are fit only on
   outer-train participant rows and carry fitted-ID provenance.
3. Imputer, scaler, optional feature transform and model fit inside the unified trainer
   on the same exact outer-train IDs. Outer-held-out labels never enter fitting.
4. The reference epoch rule is preregistered fixed epoch. The only legal alternative is
   inner participant-grouped selection followed by a fresh refit on all outer training
   participants.
5. Feature-vector prediction starts at file level. Window and member parquet remain
   schema-bearing scientific-empty tables when not applicable; they are never fabricated
   predictions.
6. File probabilities aggregate to role and participant with equal weights. An
   all-dropped participant receives an explicit empty-probability OOF row and affects
   coverage.
7. Existing output directories are rejected. A run stages all files on the same
   filesystem and publishes atomically.

## Fixed experiment artifacts / 固定实验产物

Every reduced run and every full cell writes exactly:

- `run_manifest.json`
- `metrics_per_fold_seed.json`
- `confusion_matrices.json`
- `oof_window_predictions.parquet`
- `oof_file_predictions.parquet`
- `oof_subject_predictions.parquet`
- `oof_member_predictions.parquet`
- `experiment_result.json`

Failed-closed runs write strict schema-bearing empty OOF tables and no fabricated metric.

## Evidence and claim boundary / 证据与声明边界

- [Authority registry](../../artifacts/experiments/reference_registry.json)
- [Current width-preserved 60 s real r0/f0 smoke](../../artifacts/experiments/reduced_real_r0_f0_reference_width_preserved_v2/experiment_result.json)
- [12 s fail-closed evidence](../../artifacts/experiments/reduced_real_r0_f0_12s_failed_closed/experiment_result.json)

The current passing smoke retained 5/6 OOF participants (coverage 0.8333, BA 0.5);
these are real runner diagnostics only. Full 5×5 execution, multi-representation formal
dispatch, model selection and publication performance remain uncompleted.

当前 60 秒单折结果只证明真实 runner、仅训练折拟合、OOF 完整性和主动弃权行为；
它不是完整 5×5 benchmark。raw、matrix、fusion 仍只有模块/对照/训练合同覆盖，
不得被写成同一 formal runner 已完成。
