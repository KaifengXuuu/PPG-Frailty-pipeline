# PPG Frailty Final Pipeline V2

`final_pipeline_v2` is the active, reviewable pipeline generation.  The code,
configuration, manifests, frozen folds and reports use V2 identities.  No formal
5×5 candidate benchmark, ablation study, PTT benchmark, final refit, or ONNX
winner export has been run in this repair cycle.

## Frozen default

- Participants/classes: 29 internal participants; display labels
  `Pre-Frail`, `Robust/Non-Frail`, `Young`; machine labels `0/1/2`.
  M2 snake_case names and label-source fields are provenance only.
- Inputs: B and R files. R1–R4 are separate files from one relax role; files
  have no cross-file temporal continuity.
- SQI: `off`. `diagnostics_only` saves raw observations without changing
  retention, aggregation or prediction. `route` is fail-closed pending a
  supervised threshold/weight decision.
- Line A default: training `equal_files`; aggregation
  `equal_files_no_role_layer`.
- Line B comparison: training and aggregation both
  `equal_role_families`. A/B halves cannot be mixed.
- Internal formal folds: imported, never recomputed, repeated grouped 5×5 with
  split seeds `42,10042,20042,30042,40042`. Split seeds control membership
  only. Every single-model outer cell and full-cohort refit uses training seed
  42; five-member ensembles use the exact five-member list.
- Direct filter default: 0.2–8 Hz. The 0.5–5 Hz profile is a named ablation.
- Deep epoch default: fixed 10. Fixed 7 and 15 are named ablations.
- Aggregated internal results are OOF validation, never an independent test.

The five active entry configs cover raw, feature-vector, feature-matrix and
fusion representations plus the matched Line B comparison. The formal
catalogue lists explicit architecture and training fields; registration does
not execute a model. Ensemble entries are comparison-only.

ShapeFormer's literature-reference route is channel-specific OSD/PISD:
`num_pip_ratio=0.20`, candidates bounded by three consecutive PIPs, three
shapelets per class, at most 180 participant/file-balanced discovery windows,
and channel/sample/second endpoints persisted. At a 5 s, 400 Hz discovery
sequence, T=2000 and the derived PIP count is 400. `w=128` is a position-search
neighbourhood, not shapelet length. It has no silent effect-size fallback and
remains unavailable until the faithful ShapeBlock/IG implementation passes its
explicit fidelity gate. Fixed 128/stride-64 effect-size discovery is a separate
ablation.

## Non-scientific validation

```bash
cd final_v0/final_pipeline_v2
export PYTHONPATH="$PWD/src"
export PYTHONDONTWRITEBYTECODE=1

python -B tools/materialize_data_contracts.py
python -B tools/validate_v2.py
python -B -m ppg_frailty.cli validate --all-configs
python -B tools/run_test_suite.py --suite safe
python -B tools/acceptance_gate_v2.py --output artifacts/acceptance_v2/<new-id>.json
```

The data materializer verifies all source hashes, copies frozen memberships,
and writes V2 manifests/reports/indexes. It does not invoke a splitter, train a
model, run an ablation, or execute the PTT benchmark. Acceptance additionally
checks isolated double materialization and four representation constructors.

## Evidence boundary

Everything under `historical/v1_transition` is inactive V1-transition
provenance. Its immutable inventory records archive paths, bytes and SHA-256.
Old names such as `current`, `passed`, or `manual` inside that namespace do
not describe V2 status, and active V2 gates reject equivalent stale paths.

Dependency state, executable gates and remaining work are in
[`STATUS.md`](STATUS.md). Human execution commands and safety boundaries are
in [`RUNBOOK.md`](RUNBOOK.md).
