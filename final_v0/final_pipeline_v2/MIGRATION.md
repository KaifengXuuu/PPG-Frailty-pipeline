# Legacy-to-V1 migration map / 历史到 V1 迁移映射

本文件只定义迁移边界，不把任何历史脚本升级为活动实现。所有活动代码位于
`final_pipeline_v1/src/ppg_frailty`，历史路径只读。

This map defines migration boundaries only. It does not promote a legacy script to an
active implementation. Active code lives exclusively in the V1 package.

| Legacy source / 历史来源 | V1 destination / V1 目标 | Decision / 判定 |
|---|---|---|
| `frailty_3class_classifier.py::Cnn1DClassifier` | `models.CompactCNN1D` | Copy architecture behavior, rename accurately, add shape/parameter snapshots; never copy its evaluator. |
| `frailty_3class_classifier.py::InceptionTime*` | `models.InceptionTimeFull/Small` | Copy single-network definitions; never call them a five-member ensemble. |
| legacy sklearn estimator definitions | `models.feature_baselines` | Preserve estimator families only; use frozen folds and fold-local transforms. |
| `shapeformer_port.py` effect-size discovery | `models.ShapeFormerExperimental` | Self-contained experimental route; no hard-coded external PISD import. |
| historical CV/evaluation loops | `training.Trainer` + `training.Evaluator` | Replace: outer labels were exposed for epoch selection in several routes. |
| historical feature/window caches | `data.ContentAddressedCache` | Replace: old keys omit source/config/schema/fold/code hashes. |
| M2 manifest and corrected SGKF registry | `data.manifest` + materialized V1 CSVs | Adapt and bind exact hashes; never recompute membership at runtime. |
| M3 quality/PPG/IMU/physiology pure algorithms | `signal` and `features` | Migrate source-hashed behavior with parity tests; do not import sibling package paths. |
| `funcs.py`, `ppg.py` NLMS/decomposition snippets | `artifacts` reducers | Reimplement under typed alignment/failure/provenance contracts; historical files remain reproduction-only. |
| hybrid/v7/v8/stage2 denoisers | comparison evidence only | Do not activate: fold leakage, silent fallback, misnamed algorithms, or missing route semantics. |
| historical direct window-to-subject aggregation | named legacy ablation | Reference uses window→file→role→participant. |
| historical per-window repeated feature fusion | named legacy ablation | Reference pools raw windows to one file embedding, then concatenates file features once. |
| historical result directories | `artifacts/audit/legacy_characterization.json` | Immutable evidence only; never overwrite or place on the corrected leaderboard. |

## Schema transitions / Schema 迁移

- Unversioned recording rows → `ManifestRow` / `frailty3_internal_manifest_v1`.
- Ad-hoc arrays → `SignalViews` with direct and rate-only views.
- Mixed quality scalar → endpoint-separated `QualityResult(q_rate, q_morph)`.
- Compressed PPI arrays → linked `PulseResult` with interval and adjacency indices.
- Ad-hoc feature dictionaries → ordered `FeatureVectorV1` and validity mask.
- Window-feature arrays → `EngineeringFeatureSequenceV1`.
- Repeated rows → `OrderedFeatureMatrixV1`, `K=32`, explicit row mask.
- Untraceable probabilities → window/file/role/participant OOF rows with full provenance.

## No compatibility shortcuts / 禁止兼容捷径

V1 does not import root scripts at runtime, does not load old caches/checkpoints as an
active bundle, and does not silently translate failed reducers to identity. A legacy
artifact can enter V1 only through an explicit converter plus schema, source-hash, and
golden-parity tests.
