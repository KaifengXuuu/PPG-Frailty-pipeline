# Final Pipeline V2 runbook

## 1. Bind V2

```bash
cd final_v0/final_pipeline_v2
export PYTHONPATH="$PWD/src"
export PYTHONDONTWRITEBYTECODE=1
```

Do not run active commands from `final_pipeline_v1` or from
`historical/v1_transition`.

## 2. Build and verify data contracts

```bash
python -B tools/materialize_data_contracts.py
python -B tools/validate_v2.py
python -B -m ppg_frailty.cli validate --all-configs
```

The materializer re-hashes source files and imports frozen memberships. It does
not train, evaluate, split, run an ablation, or execute PTT.

## 3. Run the non-scientific gate

```bash
python -B tools/run_test_suite.py --suite safe
python -B tools/acceptance_gate_v2.py \
  --output artifacts/acceptance_v2/<new-unique-id>.json
```

Outputs are non-overwriting. Do not use a `current` filename. A green result
means contract/config/build-data/four-representation smoke passed; it is not a
scientific result.

## 4. Formal experiment boundary

`run-experiment` is the only frailty benchmark entry. A full run requires an
exact dependency gate, clean source snapshot, new output directory, and
explicit scientific confirmation. Reduced mode is diagnostic only. Operational
CPU batch-1 cost measurement is opt-in and never runs during validation.

The five formal entry configs are:

- `reference_static_line_a_v2.yaml` — raw Line A.
- `reference_static_feature_vector_v2.yaml` — feature-vector Line A.
- `reference_static_feature_vector_line_b_v2.yaml` — matched Line B.
- `reference_static_feature_matrix_v2.yaml` — matrix Line A.
- `reference_static_fusion_v2.yaml` — fusion Line A.

Do not run the 5×5 benchmark or any registered ablation unless separately
authorized. Registered entries are retained for comparison; registration is
not execution and never implies a winner.

## 5. Comparison and manual selection

The comparison command accepts only complete, hash-indexed run directories with
the same participant/repeat roster. It rebuilds metrics and both LCB95 columns
from typed participant OOF, then runs paired tests/Holm. Coverage mismatch is an
error; no participant intersection and no automatic selection are allowed.

Comparison archives have `selections=()`. A later purpose-specific manual
selection record must identify the comparison index hash, config ID/hash,
original registry role (including ablation), purpose and human rationale.
Final refit must consume that verified record plus one complete run directory;
caller-authored plan hashes are not trusted.

```bash
python -B -m ppg_frailty.cli record-selection \
  --comparison-archive <archive> --config-id <eligible-top10-config> \
  --purpose <purpose> --rationale <human-rationale> --output <new-record.json>
python -B -m ppg_frailty.cli final-refit \
  --run-directory <complete-run> --selection-record <record.json> \
  --selection-record-sha256 <out-of-band-sha256> \
  --comparison-archive <archive> --config <registered-config> \
  --confirm-scientific-execution
```

The CLI `final-refit` command remains a read-only trust-path preflight. The
programmatic `execute_final_refit_from_verified_artifacts` boundary is now the
only executable full-29 route: it accepts no caller dataset, trainer, plan,
factory, estimator, metadata, transform, adapter or ShapeFormer bank. It
re-verifies the selected 5x5 run and manual-selection record, clean tracked
source and exact dependencies, then internally loads the manifest-bound source
bytes, materializes preprocessing and the selected representation, refits and
publishes one immutable bundle. The source gate is repeated before and after
fit. In an untracked or dirty V2 tree, execution remains fail-closed.

Only that verified executor may publish `trusted_final_refit_v2`. It derives
the frozen run identity and all-29 roster from verified inputs and commits the
bundle plus `final_refit_attestation.json` atomically. The generic public bundle
writer emits only `generic_research` and rejects final-refit identity metadata;
generic bundles cannot enter the ONNX winner gate.

## 6. Motion commands

`motion-train-internal` and `motion-evaluate-ptt` are explicit scientific
commands and require `--confirm-scientific-execution`. Without it they exit
nonzero before execution. The internal command binds the frozen 29-participant
single SGKF5 table. The PTT command also requires a hash-bound unit-resolution
artifact; while that evidence is absent, it remains fail-closed.

## 7. Final refit and ONNX

Final refit is never selected automatically. The full-29 single-model seed is
42; a five-member final ensemble uses exactly
`[42,10042,20042,30042,40042]`. A selected winner must later pass the ONNX
gate; other ablations need not export. No final refit or ONNX export is run by
the non-scientific gate.

The source-bound exporter + ONNX Runtime readback producer is implemented for
reviewed PyTorch and sklearn LogisticRegression/SVC/ExtraTrees routes. Other
families return a structured `unsupported_no_certificate` result. It must be
run from the validated isolated `onnx_winner_gate` Python 3.11 lock, not the
ordinary conda ml prompt, and requires explicit confirmation. It never trains,
selects or releases:

```bash
python -B -m ppg_frailty.cli produce-onnx-winner-certificate \
  --bundle-directory <final-bundle> \
  --bundle-manifest-sha256 <sha256> \
  --final-refit-attestation <final-bundle/final_refit_attestation.json> \
  --final-refit-attestation-sha256 <sha256> \
  --selection-record <record.json> \
  --selection-record-sha256 <sha256> \
  --config <registered-config.yaml> \
  --output-dir <new-output-directory> \
  --confirm-onnx-execution
```

A caller-authored certificate can never open the release gate. The read-only
preflight re-verifies the selected final bundle, actual ONNX bytes, artifact
index, source/lock identities, both archived probability matrices, absolute and
relative error bounds, and class-order/argmax agreement:

```bash
python -B -m ppg_frailty.cli winner-release-preflight \
  --bundle-directory <final-bundle> --onnx-model <model.onnx> \
  --bundle-manifest-sha256 <sha256> \
  --final-refit-attestation <final-bundle/final_refit_attestation.json> \
  --final-refit-attestation-sha256 <sha256> \
  --certificate <onnx-certificate.json> --certificate-sha256 <sha256> \
  --selection-record <record.json> --selection-record-sha256 <sha256> \
  --config <registered-config.yaml>
```

## 8. Archived V1 material

Historical material is read-only provenance under
`historical/v1_transition`. Validate its byte inventory through the V2
acceptance gate. Never copy an archived `passed/current/manual` result back
into active `artifacts` or cite it as V2 evidence.
