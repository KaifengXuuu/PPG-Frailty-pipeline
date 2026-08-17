# Final Pipeline V2 status

## Current state

| Area | Status |
|---|---|
| V2 identity/config/contracts | Implemented and under active validation |
| Data build | Materialized and hash-indexed; no split recomputation |
| Formal catalogue | Explicit multi-representation Line A/B catalogue; profile-only additions do not auto-run |
| SQI | Default off; diagnostics-only non-causal; supervised route deferred |
| Physical recording QC | Manifest-bound finite/flatline/minimum-duration gate; device rails/absolute scale deferred |
| Motion reference | Explicit scientific CLI boundary; internal run requires confirmation; PTT remains closed without frozen unit evidence |
| Comparison statistics | Explicit run-directory archive; BA and macro-F1 bootstrap LCB95, paired permutation and Holm; no auto-selection |
| Final selection/refit | Immutable eligible-top10 manual selection plus a source-bound internal all-29 materializer/refit/bundle boundary are implemented; explicit scientific confirmation is required; no refit/bundle run |
| Scientific execution | 5×5 benchmark, ablations and PTT benchmark not run |
| Deployment | Source-bound sklearn/PyTorch ONNX export, ONNX Runtime readback and winner certificate verification are implemented behind an exact isolated optional lock and explicit confirmation; unsupported model families emit no certificate; no project-winner export/release run; V2-026 hardware/power/latency target deferred |

## Frozen data identities

- Internal manifest: `manifests/internal_records_v2.csv`, 261 files / 29
  participants, with each source hash verified.
- Motion single SGKF5 seed42:
  `splits/sgkf5_seed42_v2.csv`, 29 rows, SHA-256
  `130b2887eb29a5a534397b4ce4dc7032f9de30ae46533fa0b2c41559ff4a1284`.
- Frailty repeated grouped 5×5:
  `splits/sgkf5_repeated_grouped_5x5_v2.csv`, 145 rows / 25 cells,
  SHA-256
  `1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702`.
- Both source tables retain M2 `source_registry_id =
  frailty3_future_corrected_sgkf5_v2`; the motion derived-registry identity is
  separate and cannot replace the source artifact identity.

## Explicitly deferred

- V2-006 device ADC rails, absolute scale and device-specific QC.
- V2-009a/b/c SQI fusion weights, thresholds and supervised route.
- V2-010 motion override activation pending internal supervision and PTT.
- V2-012 final artifact-reducer winner.
- V2-026 deployment hardware, power and end-to-end latency thresholds.
- V2-027 todo-only scope.

## Dependency boundary

The main environment is inspected, not mutated. Core, deep and formal-benchmark
profiles are exact-lock materialized and the scientific gate also compares the
live Python/platform/prefix/package inventory and runs `pip check`; static lock
status alone cannot authorize execution. The V2-035a Aura comparison is now
frozen to an isolated Python 3.11 profile with `hrv-analysis==1.0.2`,
`nolds==0.6.2`, `astropy==5.2.2` and `numpy==1.26.4`. The default resolver
still selects broken `nolds==0.6.3`; Astropy 8 also removes the import surface
used by hrv-analysis 1.0.2. The exact isolated pins pass import, `pip check`
and all five fixed-PPI function-only fixtures without changing conda ml.
The rhenan backend remains a normally disabled historical comparison profile.
Exact lock status must reflect only commands actually verified; pending
profiles cannot be promoted by documentation. ONNX now has a separate validated
Python 3.11 exact lock with real tiny sklearn and PyTorch export/ORT readbacks;
it does not mutate conda ml and is required only for the post-selection winner
gate. The normally disabled rhenan profile remains pending and does not block
ordinary formal configs.

## Historical boundary

All copied V1 configs, reports, old acceptance/current/manual artifacts, reduced
runs, phase logs and V1-only gates/tests are under
`historical/v1_transition`. They are excluded from active V2 loaders and
acceptance. `historical/v1_transition/INVENTORY.json` is the immutable
path/byte/SHA inventory.
