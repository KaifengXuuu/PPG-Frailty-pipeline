# V2 signal, SQI, artifact and feature routes

The default direct filter is 0.2–8 Hz; 0.5–5 Hz is a named ablation. SQI has
three exact modes:

## Peak detector contract

The canonical detector is the **project Aboy++-inspired detector** registered
as `aboy_project_v1`; this name does not claim exact reproduction of the
published upstream Aboy++ implementation. It operates independently on RED and
IR at 400 Hz, uses complete non-overlapping 10 s blocks, carries the persisted
HRI state between blocks, evaluates both polarities, and preserves rejected
intervals on the original event timeline. Every formal config persists
`signal.peak_detector.detector_id` and
`failure_action: fail_closed_no_fallback`.

The numerically preserved whole-record detector is available only through the
named ablation `dual_polarity_prominence_v1_ablation`. Missing or unknown IDs
are errors; canonical detector failure never invokes the ablation.

## Dual-wavelength optical contract

RED and IR are detected independently with the same persisted detector ID.
Reference selection uses detector score, then coverage, then RED as the exact
tie-break. Interior reference beats define midpoint-bounded cycles; the
secondary wavelength is paired one-to-one without reuse, and paired/unpaired/
ambiguous rows retain both ordinals, samples, lag and reasons. Each wavelength
uses its own polarity, peak and valleys. Recording medians of paired-valid AC
and DC values are computed before PI, AC ratio, absolute-DC ratio and
ratio-of-ratios. Standardized cross-correlation searches inclusive +/-0.5 s
(+/-200 samples at 400 Hz) with a deterministic lag tie-break. Coherence is not
computed or admitted to the formal predictor registry.

## Engineering feature contract

Engineering uses complete 10 s windows with 5 s hop on the 400 Hz grid. The
ordered schema has exactly 115 columns: RED/IR and A/Omega/J each receive seven
time descriptors, four spectral summaries and family-specific band powers;
the six individual IMU axes receive only the seven time descriptors. Welch
uses the longest contiguous finite run, a Hann window,
nperseg=min(N,max(64,min(2048,4*fs))), and 50% overlap (1600/800 for a
complete canonical window). File aggregation produces exactly 115 means and
115 population SDs. This engineering representation does not change the
eight-channel raw/fusion tensor.

| Mode | Computation | Effect on classifier |
|---|---|---|
| `off` | no SQI computation | none; V2 default |
| `diagnostics_only` | raw components saved separately | none |
| `route` | may keep/drop/select reducer | disabled pending supervision |

`diagnostics_only` calls the raw diagnostic evaluator and cannot emit routing
weights or normalized threshold decisions. Failures are recorded without
dropping a record.

Identity is the current artifact reducer. Named EMD-sifting-rate-only,
CEEMD-lite-NLMS legacy and DWT-A2 legacy routes remain ablations. No fictional
ANS is registered. A non-identity route is rate-only: post-reducer morphology
and amplitude-dependent fields are not applicable, and reducer failure cannot
silently fall back to the direct route.

IMU reference units are m/s² and rad/s. The calibrated roll-pitch EKF path uses
zero-phase third-order sensor low-pass filters at 20 Hz for acceleration and
40 Hz for gyroscope; its separate 0.3 Hz fourth-order gravity filter, state and
covariances are unchanged. Frailty raw/fusion and the motion 8-channel
reference scale six IMU axes on outer-training participants only; the named
11-channel motion augmentation scales its nine IMU-derived inputs. Each uses
median centering and IQR/1.349, then population SD and finally 1 for degenerate
scales. RED/IR are untouched by these fold scalers, and per-window IMU
amplitude normalization is forbidden. LPF gravity separation remains a named
ablation rather than an error fallback.
