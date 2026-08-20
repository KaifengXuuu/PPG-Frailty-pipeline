# V2 signal, SQI, artifact and feature routes

The default direct filter is 0.2–8 Hz; 0.5–5 Hz is a named ablation. SQI has
three exact modes:

## Peak detector contract

The current production default remains the historical **project
Aboy++-inspired detector** registered as `aboy_project_v1`; this name does not
claim exact reproduction of the published upstream Aboy++ implementation.
`aboy_project_v2` is a separate registered ablation candidate implementing the
authoritative seven-step project contract: it owns the 0.2 Hz high-pass,
selects polarity independently in each complete non-overlapping 10 s block,
updates HRI against the retained-Pd median, and physically removes ratio-failed
peaks before physiological/MAD interval cleaning. Every formal config persists
`signal.peak_detector.detector_id` and
`failure_action: fail_closed_no_fallback`.

The v2 candidate and two other tested comparators are ordinary parallel modules
rather than code copied into study runners: `dual_polarity_prominence_v1_ablation`
and `msptdfast_v2_3_python_port`. Stage-ablation-01 v2 and the classifier pipeline
both import the single implementation in `peaks/msptdfast_v2.py`; the latter is
an equation-level Python port bound to the reviewed ppg-beats v2.3 source hash,
not a claim of bitwise MATLAB parity. Missing or unknown IDs are errors, and a
detector failure never invokes another detector as fallback.

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

The engineering reference preset uses complete 10 s windows with a 5 s hop on
the 400 Hz internal grid. Window length, hop, alignment and cap are resolved
from the selected engineering window profile; the reference values are
defaults, not permissions or runtime gates. The
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
| `route` | fits endpoint calibration on outer-training participants, then applies the configured keep/drop/reducer state machine | changes only the explicitly selected route; executable without an authorization gate |

`diagnostics_only` calls the raw diagnostic evaluator and cannot emit routing
weights or normalized threshold decisions. Failures are recorded without
dropping a record.

Identity is the reference artifact reducer. Named EMD-sifting-rate-only,
CEEMD-lite-NLMS legacy and DWT-A2 legacy reducers are selectable parallel
modules. No fictional ANS is registered. A non-identity route is rate-only:
post-reducer morphology and amplitude-dependent fields are marked ineligible
rather than extracted from an incompatible signal, while rate/interval and
eligible matrix features remain available. Reducer failure cannot silently
fall back to the direct route.

IMU reference units are m/s² and rad/s. Internal acceleration in g uses the
Profile-B factor 9.81 m/s²; the hash-bound PTT m/s² source remains an identity
conversion. The calibrated roll-pitch EKF path uses zero-phase third-order
sensor low-pass filters at 20 Hz for acceleration and 40 Hz for gyroscope. Its
five-state roll/pitch/gyro-bias model uses forward Euler propagation and
Q=diag(5,5,.05,.05,.05)dt, while R=diag(.5,.5) is scaled by
`1 + 3*max(0, ||a||-g)/g`. Gravity is reconstructed only from roll and pitch as
`(Rx(phi)Ry(theta)).T[0,0,g]`; yaw is not corrected without a magnetometer.
Frailty raw/fusion and the motion 8-channel
reference scale six IMU axes on outer-training participants only; the named
11-channel motion augmentation scales its nine IMU-derived inputs. Each uses
median centering and IQR/1.349, then population SD and finally 1 for degenerate
scales. RED/IR are untouched by these fold scalers, and per-window IMU
amplitude normalization is forbidden. LPF gravity separation remains a named
ablation rather than an error fallback.
