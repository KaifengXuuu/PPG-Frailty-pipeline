# Phase 05 — Data and protocol layer / 数据与协议层

- Date / 日期: 2026-08-15
- Status / 状态: completed_and_verified
- Write boundary / 写入边界: final_v0/final_pipeline_v1 only
- Tracking sync / 跟踪同步: intentionally not run; root task performs one merged sync

## 1. Scope / 范围

中文：本阶段把 M2 已审核的数据身份、recording QC、corrected subject folds、
统一切窗和 provenance-bound cache 落地为可运行代码，并物化内部与外部数据合同。
没有重新推断 frailty label，没有调用运行时 SGKF，也没有把任何外部数据称为独立
test。所有 malformed row 均聚合报告后 fail closed，不允许 silent skip。

English: This phase implements the audited M2 data identity, recording QC,
corrected subject folds, unified window planning, and provenance-bound cache.
It materializes both internal and external contracts. Frailty labels are never
re-inferred, no runtime SGKF is invoked, and no external data is called an
independent test. Malformed rows are aggregated and fail closed; silent skip is
forbidden.

## 2. Frozen authorities / 冻结权威输入

| Authority / 权威文件 | Identity / 身份 |
|---|---|
| Implementation specification | cd7c4907a8beccd301048dea07ae6fdc4e9d2dc839759cb8dddd4a461e3c5000 |
| M2 internal file manifest | bd429ae9c56974ba9ffcb924dfbad0ed930f7d2d47418365754a1929ada06e90 |
| M2 dataset version | frailty3_m2_20260815_a054800abda272f6 |
| M2 corrected fold JSON file | c80e780d60162ff1132101ee368ee3a55d95f3c7a6d51a5c9a8feec8605d388c |
| M2 corrected fold payload | 0bca827fa8e4f9679b6ee9435b5497a60d91e74ddb366e058fc69614fb007f46 |
| M2 corrected fold registry ID | frailty3_future_corrected_sgkf5_v2 |
| M2 active protocol registry | beae2a6922ae0ca840cec1a5c501cde6b6fc029afed16fc798aa2ef8e05fa394 |
| Active protocol ID | frailty3_fixed_epoch_oof_v2_corrected_sgkf |
| M2 external record manifest | 43ab3273346469e9f689ce32da9c5ad280d0a53a8bc8864adf5716f40f9f024e |
| Historical external 15/3/4 split | 6350387f086dfb289b541ff61832572d55a0bc33fa7b6fc0a2428aaec61c687f |

## 3. Implemented code / 已实现代码

### 3.1 Internal manifest / 内部清单

- src/ppg_frailty/data/schema.py
  - exact RED, IR, AX, AY, AZ, GX, GY, GZ schema;
  - exact class names and nine registered roles;
  - strict JSON CSV encoding and typed read-back;
  - machine-readable QC and fold row contracts.
- src/ppg_frailty/data/manifest.py
  - build_internal_manifest(source_csv, output_csv);
  - load_manifest(path) and load_internal_manifest(path);
  - audit_manifest(rows);
  - exact M2 path and SHA enforcement;
  - all 261 source recordings re-hashed before a successful build;
  - 261 records, 29 participants, 9/12/8 class roster, and nine-role coverage
    checked as one indivisible contract.

### 3.2 Recording QC / Recording 级质量控制

- src/ppg_frailty/data/qc.py
  - all thresholds are explicit constructor fields;
  - missing channel, all-nonfinite channel, long gap, insufficient duration,
    flatline, clipping, saturation, implausible scale, timestamp failure, and
    synchrony failure have stable reason codes;
  - parse failure remains a visible failed assessment;
  - unknown channel schemas return missing_required_channel rather than aborting
    through an unknown threshold key;
  - QC is recording eligibility, not SQI, activity, or frailty label.

### 3.3 Frozen folds and evaluation protocol / 冻结折叠与评价协议

- src/ppg_frailty/data/folds.py
  - load_frozen_memberships(json);
  - materialize_fold_csvs(...);
  - FrozenFoldRegistry.from_csv(path) and get_split(repeat_index, fold_index);
  - M2 file SHA and M2 builder-compatible pretty-JSON payload SHA both verified;
  - participant train/OOF disjointness, exact OOF partition, file inheritance,
    all-class presence, and per-class fold spread at most one verified;
  - no splitter is exposed or invoked.
- Primary and repeat protocol:
  - seeds 42, 10042, 20042, 30042, 40042;
  - 5 folds x 5 repeats;
  - subject grouped;
  - fixed epoch;
  - outer OOF invisible during fitting;
  - historical sklearn-1.4.2 membership is reproduction-only.

### 3.4 Unified WindowPlan / 统一切窗

- src/ppg_frailty/data/windows.py
  - WindowPlan.plan(n_samples, fs);
  - explicit physical window/hop, start or end alignment, short-record action,
    padded-tail policy, cap, and cap policy;
  - WindowSlice carries source_record_id, fs, exact sample boundaries,
    valid_length, and padding_mask;
  - uniform-progress cap preserves recording progress rather than using only the
    first K windows;
  - extract_window copies data and applies explicit right padding.

### 3.5 Content-addressed cache / 内容寻址缓存

- src/ppg_frailty/data/cache.py
  - ContentAddressedCache;
  - identity includes source, config, schema, producer, and fold hashes;
  - payload_sha256 is the SHA-256 of raw bytes;
  - metadata and payload tampering fail closed;
  - NPZ is loaded with allow_pickle=False.

### 3.6 External heartbeat/motion contract / 外部 heartbeat-motion 合同

- src/ppg_frailty/data/external_manifest.py
  - imports exactly the 80-row M2 external authority;
  - PTT: 66 included records, 22 subjects, sit/walk/run per subject;
  - SIM: 14 authority rows, exactly 13 included and one excluded;
  - PTT pleth_1..pleth_6 wavelength mapping remains
    unresolved_red_ir_mapping_conflict and is never inferred as RED/IR;
  - PTT single-file SHA and SIM file-snapshot JSON checksum encodings are both
    preserved and validated;
  - all external uses are heartbeat/motion benchmark candidates;
  - independence claim is none_not_an_independent_external_test.

### 3.7 Provisional external grouped folds / 暂定外部分组折叠

- Registry ID: v1_provisional_external_grouped_split_seed42.
- Scope: 22 PTT subjects only; each subject carries sit/walk/run.
- Algorithm: SHA-256 rank of seed 42 plus subject_id, then deterministic
  round-robin assignment to five folds.
- OOF subject counts: fold 0=5, fold 1=5, folds 2/3/4=4.
- Every OOF fold covers sit, walk, and run.
- CSV materializes train and OOF rows; runtime recomputation is false.
- Status: provisional_pending_v2_human_confirmation.
- It is not an independent test split and must be included in V2 human decisions.
- The legacy 15/3/4 split is recorded only as historical and is not active.

## 4. Materializer workflow / 物化流程

tools/materialize_data_contracts.py performs this exact order:

1. Verify the implementation-spec SHA and active M2 protocol registry.
2. Replace prior pass reports with materializing_incomplete_fail_closed state.
3. Import the internal M2 manifest and re-hash all 261 raw source recordings.
4. Import the external M2 manifest without changing source channel semantics.
5. Load and validate the frozen corrected fold JSON; never recompute SGKF.
6. Materialize primary seed-42 and all-five-repeat internal fold CSVs.
7. Materialize the provisional PTT grouped five-fold CSV.
8. Verify and register historical-only fold assets.
9. Write strict JSON audit reports with artifact and producer-source SHA-256.

中文：第 2 步确保若后续失败，旧的 pass 报告会先失效；因此部分生成物不能与
上一次成功报告错误配对。报告中的 artifact SHA 还必须在回读测试中逐项匹配。

English: Step 2 revokes stale success before any later failure can occur, so
partial outputs cannot be paired with a prior pass report. Every artifact digest
is also checked during report read-back tests.

## 5. Generated artifacts and byte identities / 生成物与字节身份

| Artifact / 生成物 | Rows or status / 行数或状态 | SHA-256 |
|---|---:|---|
| manifests/internal_records_v1.csv | 261 data rows | 5b5788fff09910e6c224e2548869f4085fd2bbb480adcc92e0f11b09ee0387ee |
| manifests/external_records_v1.csv | 80 data rows | e6be12bf1578553dccbcc8fa76c2c1e7be47e38b54e3581b6b03dbe9fc4cb7ee |
| splits/sgkf5_v1.csv | 29 participant assignments | 130b2887eb29a5a534397b4ce4dc7032f9de30ae46533fa0b2c41559ff4a1284 |
| splits/sgkf5_repeats_v1.csv | 145 participant assignments | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 |
| splits/v1_provisional_external_grouped_split_seed42.csv | 110 fold/subject train-OOF rows | d37926011b61184742d819951329e96f7f87bd34108733fca182a8e08469ec6b |
| reports/data_contract_report.json | pass | 8b58a84d400e4749b474ebbd37e5952a4c647331d5b048ef39a3ad4aafa9df5c |
| reports/external_data_contract_report.json | pass with provisional confirmation pending | 3a424374c727d54bc84061b917cfcfb8e2cc2c07c75cdbbd52803e2d1d45dab2 |

The materializer was executed twice without code or input changes. All seven
artifact SHA-256 values were identical on the second run.

## 6. Verification / 验证

### Syntax / 语法

- python3 -m py_compile on all data modules and the materializer: PASS.

### Standard-library unit tests / 标准库单元测试

Command:

    PYTHONPATH=final_v0/final_pipeline_v1/src +    python3 -m unittest discover -s final_v0/final_pipeline_v1/tests/data -v

Result:

- 17 tests;
- 17 passed;
- 0 failures;
- 0 errors.

Covered behaviors:

- exact internal and external roster;
- no PTT wavelength inference;
- QC pass/failure/parse/unknown-channel behavior;
- frozen registry identity, tamper rejection, and partition resolution;
- WindowPlan alignment, padding mask, short-record policy, and uniform cap;
- raw-byte cache SHA, tamper rejection, and safe NPZ round trip;
- generated CSV typed read-back;
- generated report artifact SHA and byte-size matching.

## 7. Failures found and corrected / 发现并修正的问题

1. Initial external validation assumed every checksum cell was one SHA-256.
   M2 SIM rows actually contain a strict JSON mapping from every snapshot file
   path to its SHA-256. The adapter was corrected to validate and preserve both
   authority encodings without flattening or dropping component hashes.
2. Initial frozen-registry payload verification used the V1 compact JSON hash.
   M2 defines payload identity over sorted, indented strict JSON plus a final
   newline. The loader now reproduces the M2 builder rule exactly while retaining
   the separate byte-exact file SHA check.
3. An unknown channel name could reach a missing threshold key after the schema
   mismatch was already detected. QC now returns the explicit
   missing_required_channel result and keeps the recording visible.
4. The materializer initially could leave a previous pass report during a failed
   rerun. It now writes an incomplete fail-closed state before producing outputs.

## 8. Known boundary and V2 point / 已知边界与 V2 确认点

- The provisional PTT subject five-fold registry requires explicit V2 human
  confirmation before it can become a frozen benchmark protocol.
- Even after confirmation, it remains grouped development/benchmark CV unless a
  separate independence argument is approved; it is not an independent test.
- External dataset component paths are dataset-relative in the M2 record table.
  This V1 run verifies the M2 authority table byte hash and preserves its audited
  component checksum map; it does not invent repository-relative snapshot paths.
- OOF prediction storage and hierarchical window-to-file-to-role-to-participant
  aggregation consume these fold/data contracts in the training layer; they are
  not reimplemented inside the data package.

## 9. Self-review / 自审结论

- No files outside final_v0/final_pipeline_v1 were written.
- AGENTS.md and _agent were not modified.
- No runtime SGKF or legacy split was activated.
- No class label was inferred from a source filename.
- No PTT wavelength was inferred.
- QC has no silent skip path.
- Generated artifacts round-trip and match their report hashes.
- No tracking sync was run in this phase, by explicit parent-task instruction.
