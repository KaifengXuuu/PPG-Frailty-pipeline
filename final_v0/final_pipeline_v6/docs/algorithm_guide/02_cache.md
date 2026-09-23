# 2. Deterministic Preprocessing Cache

## 2.1 Purpose and Reuse Boundary

**Intuition.** If the same long paper strip is smoothed and cut into equal pieces using the same rules, a second pass should produce the same pieces. A cache stores the first set in a drawer labeled with the ingredients and procedure. If the drawer exists, take the pieces out; otherwise, make them again. It neither guesses class labels nor copies another person's answer.

**Inputs and outputs.** Inputs are source-recording identity, unlabeled signals from that recording or the same participant's B recording, current computation parameters, and upstream computation identities. Outputs are the numerical arrays and necessary metadata returned unchanged by the builder. The cache does not modify filtering, windowing, normalization, or other mathematical formulas.

`src/ppg_frailty/data/preprocessing_cache.py:1–7,47–54` defines four cache layers. `experiment.py:308–351,413–423,953–969,1533–1558` contains their actual calibration, signal-view, motion-window, and raw-window calls (here and below, omitted shared prefixes refer to files under `src/ppg_frailty/`).

| Layer | Stored content | Explicitly excluded content |
|---|---|---|
| `imu_calibration` | Same-participant static B biases, initial orientation, and calibration configuration | Frailty labels and parameters learned by classifier training |
| `canonical_signal_views` | Raw/filtered/analysis PPG, validity and repair markers, and processed IMU | Artifact-reduced waveforms and quality-route selection |
| `raw_windows` | Raw windows before routing selection, valid masks, and start positions | Labels, quality weights, final aggregation-retention masks, and within-fold transforms |
| `motion_windows` | Initial 8-second/2-second windows required by the motion model | Fold-fitted IMU scaling, motion probabilities, thresholds, and routing results |

`preprocessing_cache.py:799–819` explicitly records fitted states that are not shared. Although static calibration contains the word `fit`, it reads the same participant's B signal without using their frailty label; this is not equivalent to applying all participants' training statistics to a test participant. Using this calibration still requires a same-participant B recording at deployment.

## 2.2 Three Read/Write Modes

**Intuition.** `off` repeats the work every time; `read_only` may take existing pieces from the drawer but adds nothing; `read_write` both reuses existing pieces and stores new ones.

**Step-by-step code correspondence.** `PreprocessingCacheSession._operate` is at `src/ppg_frailty/data/preprocessing_cache.py:290–334`:

1. Line 295 starts timing; line 296 checks whether the current layer is selected.
2. Lines 297–301: when caching is disabled or the layer is unselected, call `builder()` directly and return its arrays and metadata without reading or writing entries.
3. Lines 302–311: `read_only` first calls `load(identity)`; only `CacheMissError` triggers recomputation, without publishing an entry. Corruption and source changes are not ordinary misses and cannot be treated as successful reads by this branch.
4. Lines 312–316: `read_write` calls `get_or_compute`, reusing a hit or computing and storing a miss.
5. Lines 317–334: record elapsed time, key, array bytes, hit status, and return values. This timing covers the cache operation, not the whole training run.

For a deterministic stage $y=f_\theta(x)$, all three modes should return the same $y$; only repeated evaluation and persistence differ. Here $\theta$ means the selected parameters of that stage, not trained classifier weights. A changed source or $\theta$ must not reuse the old drawer.

## 2.3 What Determines the Cache Key

**Intuition.** A drawer label must say more than “this person's recording”: it must identify the exact recording, the scissors used, whether cutting preceded smoothing, and each piece's length. Changes in order, parameters, or ingredients define a different result.

**Code and mathematics.** `src/ppg_frailty/data/recording_cache.py:136–208` defines identity:

1. `NamedSourceDependency.to_payload` (142–147) stores the dependency name, content hash, and recording attributes.
2. `OrderedModuleSpec.to_payload` (157–169) stores each module's chain position, name, version, implementation hash, switch, and parameters. Order is part of identity.
3. `RecordingCacheIdentity.to_payload` (182–204) sorts source dependencies by name, preserves computational module order, and adds output structure and extra information.
4. `key` (207–208) computes SHA-256 of this canonical representation.
5. `stable_payload_sha256` at `src/ppg_frailty/provenance.py:25–35` uses JSON with sorted keys, compact separators, UTF-8, and NaN disallowed, then hashes its bytes.

In notation:

$$K=H\bigl(\operatorname{JSON}_{\mathrm{canonical}}(I)\bigr).$$

$I$ is the identity object containing source dependencies, ordered modules, parameters, implementation hashes, output structure, and extra information.

$H$ is SHA-256, not a signal transform. `preprocessing_cache.py:63–127,131–158` obtains source/implementation file hashes. Within one process, previously computed hashes are reused based on stable file identity, including device, inode, size, and modification time. Absolute checkout paths are not included in implementation-hash content, so moving identical code should not by itself count as a new algorithm.

`preprocessing_cache.py:247–288` compares the actual source hash with `source_hash` in the manifest. `canonical_views` also includes upstream B-calibration identity (450–465), preventing reuse of old dynamic-recording views after changing B calibration. Relevant layers include NumPy/SciPy versions in the key's `extra` field. Automatic cache invalidation governs result reuse; it is not a runtime environment lock preventing dependency upgrades.

## 2.4 Calibration-Layer Storage and Retrieval

**Intuition.** A stationary ruler has a fixed offset. Remember that offset before measuring motion. Reprocessing the same source does not require re-estimating the same offset, but another person's stationary recording cannot be substituted.

`src/ppg_frailty/data/preprocessing_cache.py:336–438`:

1. Lines 342–383 construct identity from the B source recording, IMU configuration, implementation, and units.
2. Lines 385–406 call the actual calibration function and store the three-dimensional acceleration bias `acceleration_bias_mps2`, three-dimensional angular-velocity bias `gyroscope_bias_rads`, initial roll/pitch, calibration start/end samples, quality notes, and parameters.
3. Line 408 reads/writes through the shared mode branch.
4. Lines 410–427 restore tuple parameters and construct `MotionImuCalibration` without re-estimating biases.
5. Lines 433–437 retain the association between the calibration object and its upstream key for subsequent signal-view processing.

Calibration mathematics is described in the preprocessing chapter; this layer only performs lossless persistence, without additional averaging or precision reduction. `experiment.py:327–336` first selects an eligible same-participant B source by descending duration and ascending record ID; arbitrary static files are not interchangeable inputs.

## 2.5 Complete Signal-View Layer

**Intuition.** A curve has its original paper, a version with slow drift and fine jitter removed, a version used to inspect rhythm, and a transparent sheet marking gaps. The cache stores these separately so downstream modules do not confuse them.

`src/ppg_frailty/data/preprocessing_cache.py:440–590`:

1. Lines 450–537 include the target recording, calibration, filter/gap parameters, physical checks, IMU method, 400 Hz rate, and implementation in identity.
2. Lines 539–543 receive the view from `builder()` and require pristine `direct` state without `x_ar`. This defines the cache boundary, not a restriction to direct algorithms.
3. Lines 544–552 store `x_native`, `x_filter`, `x_analysis_rate`, `source_valid_mask`, `repair_mask`, and each `imu_processed` array.
4. Lines 553–562 store routing, physical checks, and metadata.
5. Lines 564–578 reload and reconstruct `CanonicalSignalViews`; `x_ar=None` because downstream artifact-reduction results are not cached here.

Some arrays may be numerically identical, such as un-reduced analysis and filtered views. The current implementation nevertheless writes separately named array files; identical content is not automatically deduplicated.

## 2.6 Raw-Window Layer

**Intuition.** Slide a fixed-width paper frame along a long curve and copy the eight curves inside each position. The cache stores those pieces, not pieces chosen after seeing the answers.

`src/ppg_frailty/data/preprocessing_cache.py:592–678`:

1. Lines 601–643: the key contains the upstream signal-view key, all `WindowPlan` parameters, and normalization configuration; the output axes are declared as `N_8_T`.
2. Lines 645–649: first execute `build_raw_windows`; results already containing `window_quality_scores` or `window_aggregation_mask` cannot serve as this layer's raw cache.
3. Lines 650–661: store `values`, `valid_mask`, and `start_samples`, plus candidate count, invalid-drop count, and processing provenance.
4. Lines 663–678: reconstruct `RawWindows` with its original structure.
5. Lines 827–842: window arrays must be float32 with shape $W\times8\times T$, masks $W\times T$, and starts $W$ increasing integers. Checking these conditions does not modify values.

`experiment.py:1548–1558` reads this cache; filtering by `routing_timeline` starts only after line 1559. Deep-model input sampling rate is handled later by `_prepare_dl_input_dataset` around `experiment.py:1950–1972`. Consequently, finalcase caches 400 Hz windows here, not already-converted 64 Hz model tensors.

## 2.7 Motion-Window Layer

**Intuition.** Detecting movement requires a differently sized paper frame. Those pieces are stored separately from classification windows, before the ruler's scale is adjusted using the training population.

`src/ppg_frailty/data/preprocessing_cache.py:680–791`:

1. Lines 698–749 construct identity from the motion-input profile, upstream views, implementation, channel structure, and sampling rate, explicitly setting `fold_imu_scaler_applied=False`.
2. Lines 751–766 store window `values`, `start_samples`, profile, and channel order.
3. Lines 768–784 restore tensors; record, participant, role/activity, and dataset identities come from the current manifest, not supervised labels in the cache.
4. `experiment.py:953–969` supplies prebuilt windows to `infer_reused_motion_windows`; later model transforms and predictions are outside this cache.

Motion detection is disabled in finalcase, whose default cache-layer selection also excludes `motion_windows`. Adding the layer name to configuration does not enable the motion model.

## 2.8 Lossless Storage, Loading, and Disk Usage

**Intuition.** To avoid repeating work, each paper piece is stored intact rather than reduced to a blurry thumbnail. This is convenient to load but uses space. Reading the same drawer repeatedly does not create 25 identical copies.

Step-by-step storage in `src/ppg_frailty/data/recording_cache.py`:

1. `_prepare_array` (236–252) converts arrays to contiguous C order while preserving dtype and shape. Content hashes include dtype, shape, and array bytes, not merely printed numbers.
2. `_load_verified` (322–426) checks metadata, the complete-commit marker, file lengths/hashes, shapes, and dtypes. Line 397 uses `np.load(..., allow_pickle=False, mmap_mode="r")` for read-only mapping; line 415 marks arrays non-writable.
3. `get_or_compute` (446–464) looks up an entry, rechecks a miss inside a process mutex, then calls the builder and publishes. The mutex only prevents simultaneous writers to one entry; it does not freeze the environment or algorithm choice.
4. `_publish` (484–533) writes arrays with `np.save` to a temporary location, writes metadata and the completion marker, and finally renames it to the published entry. There is no lossy compression or automatic float64-to-float32 conversion.

Ideal array storage is:

$$B=\left(\prod_i d_i\right)b,$$

where $d_i$ is each axis length and $b$ the bytes per element. Headers, masks, metadata, and filesystem allocation increase actual disk usage.

For example, 18,013 finalcase-style raw windows, each 5 seconds × 400 Hz × 8 channels × float32, require for `values` alone:

$$18{,}013\times2{,}000\times8\times4=1{,}152{,}832{,}000\text{ bytes}.$$

Multiple float64 PPG/IMU views of complete recordings can add several GB. This illustrates where storage comes from, not a fixed size for every input. Upstream parameter, source, or implementation changes generate new keys; old entries are not automatically removed. Disabling a namespace does not delete historical files either.

## 2.9 Configuration Entry Points, Defaults, and Inspection

| Entry point | Switches and defaults |
|---|---|
| `pipeline.py run` | `--preprocessing-cache-mode off/read_only/read_write`; generic execution defaults to `off` when unspecified |
| Cache directory | `--preprocessing-cache-root`; generic execution defaults to `artifacts/studies/cache`, within V6 |
| Layer selection | `--preprocessing-cache-namespaces` takes comma-separated names; the generic default includes all four, but entries arise only when the corresponding algorithm is called |
| `sweep.py` | Study YAML `execution.preprocessing_cache`, not a pipeline leaf parameter via `--set` |
| finalcase study | `read_write`, `cache/preprocessing`, and the calibration/signal-view/raw-window layers |
| `manual-cli` output | Explicitly generates `read_write` and `cache/preprocessing`; edit the generated command directly if needed |
| Dash | Ordinary runs use the mode dropdown; sweeps follow the selected study YAML |

Default sources: `src/ppg_frailty/study/schema.py:473–508`; CLI parameters and precedence: `src/ppg_frailty/v5/cli.py:128–130,208–232,388–390`; finalcase study: `configs/studies/finalcase.yaml:76–83`.

Each fold's `preprocessing_cache.json` records hits/new writes/skips, per-layer array sizes, and durations. `logical_array_bytes` in `preprocessing_cache.py:793–824` sums bytes accessed during this session's operation events. Summing it over 25 folds does not measure unique disk usage: multiple events may read the same entry.

Disabling the cache does not alter saved predictions, weights, or reports. Reporting reads pipeline outputs; exported-model raw inference uses its own fixed parameters and does not depend on these preprocessing cache files. Cache cleanup should be a separate action performed when no task is using it; this guide provides no automatic deletion operation.
