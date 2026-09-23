# 3. Quality and Motion Processing

This chapter covers V6 `signal/sqi.py`, `quality/`, and `artifacts/`, following actual calls in `experiment.py` and motion models in `models/motion.py`. Source locations are relative to V6 `src/ppg_frailty/`. Line numbers locate the documented implementation; function names support lookup after code moves. Existing computations are described without presenting proposed behavior as implemented behavior.

Each section starts with nontechnical intuition, followed by source-aligned formulas, steps, and boundaries. “Default” means constructor defaults; “finalcase” means explicit choices in `configs/presets/finalcase.yaml`. They are not interchangeable. Raw optical values are not converted here into blood volume or medical quantities; their magnitude cannot directly indicate health status.

SQI internal dataclass fields do not always match public YAML keys. Use the following mapping from `signal/sqi.py:210–317` when configuring parameters:

| Internal field | YAML key and default |
|---|---|
| `q_rate_threshold` / `q_morph_threshold` | `quality.rate_threshold: 0.50` / `quality.morph_threshold: 0.65` |
| `cardiac_low_hz` / `cardiac_high_hz` | `quality.cardiac_band_hz: [0.5, 3.0]` |
| `spectral_analysis_low_hz` / `spectral_analysis_high_hz` | `quality.spectral_analysis_band_hz: [0.2, 8.0]` |
| `peak_density_min_bpm` / `peak_density_max_bpm` | `quality.peak_density_bpm_range: [30.0, 200.0]` |
| `ppi_min_s` / `ppi_max_s` | `quality.ppi_range_s: [0.3, 2.0]` |
| `calibrator_lower_quantile` / `calibrator_upper_quantile` | `quality.calibrator_quantiles: [0.10, 0.90]` |
| References, scales, shape centers, individual thresholds, and template half-width | Within `quality.component_normalization`; values are given in each section |
| `welch_max_nperseg`, the three template count fields, and `ppi_stability_min_intervals` | Directly under `quality` |

Another easily misread default concerns calibration: direct `SqiConfig()` uses fixed formulas, whereas reading a quality mapping with omitted/empty `calibrator` selects empirical quantiles. Therefore production `quality.mode=route` should explicitly specify `quality.calibrator` rather than infer YAML behavior from constructor defaults. The parser inserts fixed formulas only in the non-route denoising-recovery path; see `config.py:197–275`.

## 3.1 Actual Execution Order and Independent Switches

Intuition: first ask whether fluctuations can be counted, then whether each fluctuation's shape is trustworthy. A separate “motion observer” may inspect synchronized body movement. If a segment is too messy, try one selected cleanup method once. Recovering countable fluctuations does not imply recovering their original fine shape.

The actual main entry is `_apply_quality_motion_routing` at `experiment.py:696–743`, not the retained whole-recording `_route_records` function. When routing is needed, it calls `_route_records_window_level` at `experiment.py:888–1133`.

| Setting | Meaning | finalcase |
|---|---|---|
| `quality.mode` | `off`: no evaluation; `diagnostics_only`: retain diagnostics; `route`: use quality results for routing | `off` |
| `quality.calibrator` | Fixed score formulas or a score mapping fitted only on training participants | Disabled; constructor defaults to fixed formulas |
| `artifact.motion_detector_enabled` | Score windows with an existing trained motion detector | `false` |
| `artifact.denoiser_enabled` | Allow an ineligible window to request one waveform cleanup | `false` |
| `artifact.reducer` | Actual cleanup algorithm; parameters under `artifact.parameters` | `identity` |
| `artifact.degraded_policy` | Permitted use of recovered rate features | `drop` |
| `quality.window_selection.policy` | Separate raw-training-window selection, distinct from the SQI above | `none` |

`route_module_switches_from_config` (`quality/routing.py:158–175`) reads the three routing switches independently. Only when `denoiser_enabled` is absent is it inferred from whether the reducer is nonidentity. finalcase explicitly supplies `false` and does not rely on inference.

All three routing switches are off in finalcase. The entry retains recordings already accepted by physical loading/preprocessing without computing SQI, loading a motion network, running a denoiser, or automatically discarding R/S/W roles. Input `roles` selection still operates independently. “Retain” does not cancel the preceding chapter's missingness, nonfinite, duration, or windowing handling.

When routing is enabled:

1. If empirical mapping is selected, fit the SQI mapping on direct-waveform windows from outer-training participants.
2. If motion detection is required, load the existing model, input transform, and thresholds; do not retrain here.
3. Construct complete 8-second assessment windows with 2-second hops on the 400 Hz grid. Detect peaks on the complete direct waveform, then slice peak evidence into windows.
4. Combine SQI and motion evidence per window to obtain Excellent, Acceptable, or Unfit.
5. If any window requests denoising, run the selected reducer once on the complete recording, not separately for every overlapping window.
6. Redetect peaks on the complete cleaned waveform and recompute `Q_rate` for each recovery-requesting window. Success reaches only Acceptable, not a renewed claim of fully trustworthy morphology.
7. Convert overlapping assessment windows into nonoverlapping ownership intervals for raw/feature/matrix/fusion processing.

## 3.2 Shared SQI Inputs, Raw Observations, and Scores

### 3.2.1 Input Preparation: Selecting a Truly Valid Continuous Curve

Source: `signal/sqi.py:742–881`, `_raw_quality_observations`.

Input is `CanonicalSignalViews` or an explicit floating array shaped `[N, 1]`/`[N, 2]` for one or two wavelengths, strictly at 400 Hz. Additional inputs include peak positions, intervals, validity markers, synchronized IMU, and original quality evidence. Output consists of observations shared by subsequent small modules, not class predictions.

Intuition: do not join opposite sides of a hole in a paper strip and pretend time is continuous. Select the longest genuinely usable continuous strip for waveform observations, while separately recording the usable fraction of the entire strip.

The code proceeds as follows:

1. `canonical.analysis_signal` selects the directly filtered or aligned cleaned waveform; `canonical.route` determines which properties remain interpretable.
2. Locate True runs from False/True transitions in `rate_valid_mask` and choose the largest `stop-start`. No valid interval raises an error rather than manufacturing a zero score as an ordinary observation.
3. If invalid positions exist, crop waveform and synchronized IMU to that longest interval; coverage still uses the original validity mean. Diagnostic mode obtains peaks again after cropping rather than applying old coordinates to the new array.
4. For array input, check finiteness and column count; reject absent explicit `route` because the function cannot guess whether the waveform has been altered.
5. If no pulse result is supplied, call configured `detect_pulses` rather than silently choose another detector. Peak algorithms are explained in the next chapter.
6. Valid intervals require `pulse.valid_interval_mask`, finite values, and `ppi_min_s=0.3` through `ppi_max_s=2.0` seconds simultaneously.

The Welch, autocorrelation, template, and other kernels below are reused here without frailty labels. Fixed formulas require no cross-participant fitting.

### 3.2.2 Fraction of Fluctuation Within the Usual Heart-Rate Range

Source: `signal/sqi.py:507–532`, `_welch_metrics`; score transform at `evaluate_quality:1063–1067`.

Intuition: imagine a curve as waves of many speeds added together. Concentration around half to three oscillations per second suggests countable pulses; concentration elsewhere weakens that clue. Being within the range does not prove a fluctuation is a heartbeat.

Mathematical steps:

1. Call `signal.welch(x, fs=400, nperseg=min(2048,N))` for each wavelength to obtain frequencies `f_k` and powers `P_k`.
2. Define full analysis band `U={k:0.2≤f_k≤8.0}` and cardiac band `H={k:0.5≤f_k≤3.0}`.
3. Raw observation `C=Σ_H P_k / Σ_U P_k` sums powers at frequency-grid points; it is not `trapezoid` integration.
4. Compute RED and IR separately and average. Skip a channel with a nonpositive denominator; if none is usable, return NaN and later mark unavailable.
5. Quality score is `s_C=clip(C/0.65,0,1)`, where `0.65` is `quality.component_normalization.cardiac_concentration_reference`.

Optical amplitude squared divided by amplitude squared gives a dimensionless output. Public controls are `quality.cardiac_band_hz`, `quality.spectral_analysis_band_hz`, `quality.welch_max_nperseg`, and the reference above, with defaults shown in the steps. Not executed in finalcase.

### 3.2.3 Is Wave Energy Spread Everywhere?

Source: the same `_welch_metrics` and `normalized_spectral_entropy` in `evaluate_quality`.

Intuition: is a bowl of sand concentrated in a few small boxes or spread evenly among all boxes? Concentration suggests a few dominant fluctuation speeds; dispersion suggests many mixed speeds. This measures dispersion, not heartbeats directly.

For `M` frequency points in the preceding analysis band:

`p_k=P_k/Σ_U P_k`

`H=−Σ p_k log(p_k+10⁻¹⁵) / max(log M,10⁻¹²)`

Average `H` across both channels, then use `s_H=clip(1−H,0,1)`. Small constants only avoid `log(0)` or division by zero. There is no additional trainable parameter; this shares the same Welch calculation as cardiac concentration. Not executed in finalcase.

### 3.2.4 Does the Curve Align With a Shifted Copy of Itself?

Source: `signal/sqi.py:534–551`, `_autocorrelation_periodicity`.

Intuition: photocopy the curve and move the copy slowly rightward. If peaks and valleys align again after approximately one heartbeat, the fluctuations are fairly regular.

For each wavelength, first set `x←x−mean(x)`, then compute:

`r[l]=Σ_n x[n]x[n+l] / Σ_n x[n]²`

The code calls `signal.correlate(..., mode="full", method="fft")` and retains nonnegative lags. Search `l=round(0.3fs)…round(2.0fs)`, take the maximum, then average usable channels. There is no further adjustment for overlap count at each lag; longer lags generally have fewer overlapping samples.

Score is `clip(max_l r[l]/0.70,0,1)` with `quality.component_normalization.autocorrelation_reference=0.70`; lag bounds come from `quality.ppi_range_s`. A channel is unavailable when energy is ≤`10⁻¹²` or length does not exceed maximum lag. Not executed in finalcase.

### 3.2.5 How Many Peaks Are Found Per Minute?

Source: `signal/sqi.py:846–857, 1033–1055`.

Intuition: count the already-marked hilltops. Substantially fewer than 30 or more than 200 per minute lowers the score; a slight boundary crossing does not abruptly make it zero.

For `K` peaks over `N/fs` seconds:

`d=60K/(N/fs)`, in peaks per minute.

The default accepted interval is `[30,200]`; score is 1 inside it and outside:

`s_d=exp(−|d−clip(d,30,200)|/30)`.

The public control is `quality.peak_density_bpm_range`. Here `K` is the length of `pulse.peaks`, not a recount after separately removing every failed peak. Not executed in finalcase.

### 3.2.6 Are Adjacent-Peak Distances Physiologically Plausible?

Source: `signal/sqi.py:824–840, 1073–1077`.

Intuition: some hilltops nearly overlap while others are far apart. Count how many adjacent distances both pass the peak workflow and remain plausible.

For original intervals `Δ_i` and existing validity markers `v_i`:

`u_i=v_i ∧ finite(Δ_i) ∧ (0.3≤Δ_i≤2.0)`.

Both raw observation and quality score are dimensionless `mean(u_i)`. With no intervals, return unavailable rather than interpreting “no evidence” as “all passed.” Not executed in finalcase.

### 3.2.7 Do Adjacent Distances Alternate Between Long and Short?

Source: `signal/sqi.py:833–840, 1030–1032, 1078`.

Intuition: line up the accepted distances. Similar lengths look orderly; large jumps lower the score. Real heartbeat intervals naturally vary, so this is only a weighted clue, not a requirement of perfect spacing.

With at least `ppi_stability_min_intervals=3` valid intervals:

`CV=std(Δ_valid,ddof=0)/mean(Δ_valid)`.

`s=exp(−max(CV,0)/0.20)`, where `0.20` is `quality.component_normalization.ppi_cv_scale`.

For the `ppi_stability` component in `evaluate_quality`, `raw_value` is already the exponentially mapped stability score; separate raw diagnostics retain `rate.ppi_cv`. Do not treat these table entries as the same quantity. Not executed in finalcase.

### 3.2.8 Do Red and Infrared Rise and Fall Together?

Source: `signal/sqi.py:803–809` and the `red_ir_agreement` component.

Intuition: two curves observe the same location simultaneously. Even if one is tall and the other small, matching timing supports both. The current implementation also accepts matching fluctuations with opposite directions.

`s=|corr(RED,IR)|` uses correlation as the cosine between mean-centered vectors. Absolute value gives both −1 and +1 correlation a score of 1. The component is unavailable for a single wavelength or a constant channel. No extra default parameters. Not executed in finalcase.

### 3.2.9 Larger Body Motion Weakens the Optical-Trustworthiness Clue

Source: `signal/sqi.py:840–846, 1080–1085`.

Intuition: stronger wrist movement is more likely to disturb the optical curve, so larger motion gets a lower score. This does not claim to have separated motion from optical error.

Let finite synchronized `dynamic_magnitude` values be `a_i` in m/s²:

`R=sqrt(mean(a_i²))`

`s=exp(−R/4.0)`; the denominator is `quality.component_normalization.motion_rms_scale=4.0` m/s².

The component is unavailable if this derived quantity is absent or lengths do not align. This fixed formula is not a motion CNN and trains no classifier. In an IMU branch that retains gravity, `dynamic_magnitude` keeps its name but physically includes gravity; interpretation must account for the chosen branch. finalcase disables SQI and does not use it.

### 3.2.10 Is the Curve Entirely Flat?

Source: `signal/sqi.py:855–857, 1086–1090`.

Intuition: if either curve is almost a horizontal line, it is not a fluctuating measurement.

`u=min_c std(x_c,ddof=0)` scores 1 for `u>10⁻¹⁰`, otherwise 0. Threshold `quality.component_normalization.nonflat_std_threshold` uses the original optical amplitude units. This is minimum channel variation, not the two-channel average. Not executed in finalcase.

### 3.2.11 What Fraction Actually Contains Data?

Source: `signal/sqi.py:765–801, 1091`.

Intuition: a beautiful curve drawn in one small corner cannot make the entire canvas reliable.

Coverage normally means the valid-sample fraction. A two-dimensional source mask requires all channels valid per row; reduced routes prefer the `rate_valid_mask` fraction. The component score equals this fraction. Endpoint passing separately requires at least `quality.minimum_coverage=0.80`, so coverage acts both through weighted scoring and as a direct endpoint condition. finalcase does not apply SQI coverage assessment.

### 3.2.12 Flat Runs, Extreme-Value Pileup, Saturation, and Long Gaps: Four Separate Modules

Source: `signal/sqi.py:647–711`, `_qc_components`. These read evidence from upstream PPG checks rather than infer original ADC state from an already-filtered waveform.

**Flat runs.** Intuition: beyond overall variation, check whether a long section never changes. Take maximum constant-run length `L` across wavelengths; raw value is `L/fs` seconds and score `1−min(L/(1.0fs),1)`. `quality.flatline_duration_s=1.0`.

**Extreme-value pileup.** Intuition: many points stopping near the same minimum or maximum may have hit a measurement-range wall. Take the largest channel `min_occupancy/max_occupancy` fraction `c` and score `1−min(c/0.02,1)`; `quality.component_normalization.clipping_fraction_reference=0.02`. This is an extrema-occupancy heuristic, not confirmed hardware clipping.

**Explicit ADC saturation.** Intuition: the instrument's actual limits must be known before claiming that its wall was hit. Only finite `adc_saturation_fraction=a` evidence yields `1−min(a/0.02,1)`; `quality.component_normalization.saturation_fraction_reference=0.02`. Unknown range means unavailable, not relabeling extreme-value pileup as saturation.

**Long gaps.** Intuition: a small missing corner and a large missing block need different treatment. Take the maximum channel `longest_nonfinite_gap_samples=G`; score 1 for `G≤100`, otherwise 0, with `quality.long_gap_max_samples=100`. This assesses quality rather than fills gaps; repair occurred during preprocessing.

Without `channels` evidence, all four are unavailable. finalcase does not compute these SQI components, but upstream physical-quality checks remain.

### 3.2.13 Does Each Pulse Resemble a Shared Template?

Source: `signal/sqi.py:553–586`, `_template_correlation`.

Intuition: cut a small strip around each hilltop, stretch all strips to the same width, then overlay them. Similar ridgelines allow one template to represent most strips; inconsistent shapes yield low similarity.

Computation order:

1. Require at least `template_min_peaks=5` peaks and select the RED/IR column identified by `pulse.wavelength`.
2. For accepted peak `p`, extract `[p−w,p+w]` with `w=round(0.30fs)`; skip out-of-bounds peaks.
3. Linearly interpolate each segment to `template_resample_points=101` points.
4. Subtract each segment's mean and divide by its standard deviation; skip standard deviations ≤`10⁻¹²`.
5. Require at least `template_min_beats=3` segments, then take the pointwise median template `t_j=median_b z_bj`.
6. Correlate each segment with the template and take median correlation `r`; score `max(r,0)`, then clip to `[0,1]`.

`template_half_width_s` belongs under `quality.component_normalization`; `template_min_peaks/template_min_beats/template_resample_points` belong directly under `quality`. Nonidentity denoised output is not evaluated here and cannot claim preserved peak morphology. Not executed in finalcase.

### 3.2.14 How Strongly Does the Waveform Lean to One Side?

Source: `signal/sqi.py:861–862, 1130–1134`.

Intuition: inspect distances from average height. Does one side consistently have a longer tail? Excessive asymmetry lowers this score; spikes in a particular direction are not identified as a disease.

The code uses `stats.skew(matrix, axis=0, bias=False)` to compute a finite-sample-corrected third-order shape quantity per channel, then averages channels to `g`. Score is `exp(−|g|/3.0)` with `quality.component_normalization.morph_skewness_scale=3.0`. The basic geometric quantity is `mean((x−mean x)³)/std(x)³`, but actual values include SciPy's `bias=False` correction and must not be replaced by uncorrected hand calculations. Not executed in finalcase.

For one nonconstant column with N>2, let `m_k=mean((x−mean x)^k)`. The finite-sample correction is `G1=sqrt(N(N−1))/(N−2)·m3/m2^(3/2)`. Only afterward are wavelengths averaged and exponentially mapped; mixing both wavelengths into one column first is not equivalent.

### 3.2.15 Are There Too Many Very Tall, Sharp Points?

Source: `signal/sqi.py:862–863, 1135–1139`.

Intuition: points concentrated centrally with a few far-away outliers look more peaked or heavy-tailed than evenly spread points. The code prefers this shape measure not to stray too far from a reference center.

`k=mean_channels(stats.kurtosis(..., fisher=False, bias=False))` uses the Pearson definition centered at 3, not the excess definition after subtracting 3.

Score is `exp(−|k−3.0|/5.0)`, with `morph_kurtosis_center=3.0` and `morph_kurtosis_scale=5.0` under `quality.component_normalization`. The basic quantity uses fourth-power distances, `mean((x−mean x)⁴)/std(x)⁴`, but actual computation again includes SciPy finite-sample correction. Not executed in finalcase.

For a nonconstant column with N>3, using m2/m4 notation above, set `g2=m4/m2²−3`. Corrected Pearson value is `K=(N−1)[(N+1)g2+6]/[(N−2)(N−3)]+3`. The final +3 corresponds to `fisher=False`; omitting it changes the later score centered at 3.

## 3.3 Combining Component Scores Into Two Different Endpoints

### 3.3.1 Shared Component Clipping and States

Source: `signal/sqi.py:485–505`, `_component/_components`.

Intuition: place measurements made with different rulers onto one 0–1 ruler, but do not arbitrarily replace missing measurements with zero or full marks.

Absent/nonfinite `raw` or `normalized` gives `UNAVAILABLE` with `None` value. Otherwise `score=clip(normalized,0,1)`, marked PASS only when `score≥quality.component_normalization.component_pass_threshold`, default 0.50. An individual PASS does not mean the whole segment passes.

### 3.3.2 Q_rate: Can Pulses Still Be Counted Reliably?

Source: `signal/sqi.py:14–19, 588–621, 1062–1109`.

Intuition: several observers vote, but someone unable to observe an aspect abstains rather than votes against it. A few excellent clues cannot compensate for missing data over most of the duration.

Renormalize weights over currently available components:

`Q=Σ_(i available) w_i s_i / Σ_(i available) w_i`.

The implementation first divides weights by their maximum before summing to avoid numerical scale problems; mathematically it remains the formula above. No available positive-weight component gives UNAVAILABLE. Passing requires `Q≥0.50` and coverage `≥0.80`.

| Q_rate component | Default weight |
|---|---:|
| cardiac_concentration | 0.20 |
| autocorrelation_periodicity | 0.15 |
| normalized_spectral_entropy | 0.10 |
| peak_density_bpm | 0.08 |
| ppi_physiological_fraction | 0.15 |
| ppi_stability | 0.12 |
| red_ir_agreement | 0.08 |
| motion_energy_rms | 0.05 |
| nonflat_scale | 0.02 |
| source_coverage | 0.04 |
| flatline | 0.02 |
| clipping | 0.015 |
| saturation | 0.015 |
| long_gap | 0.01 |

Weights are under `quality.rate_component_weights`; the public endpoint threshold is `quality.rate_threshold`. Original weights need not sum to 1 because the function normalizes them. Not executed in finalcase.

### 3.3.3 Q_shape and Q_morph: Can Fine Contours Be Interpreted?

Source: `signal/sqi.py:20–24, 1111–1189`.

Intuition: locating each hill does not guarantee reliable comparisons of tip sharpness or base width. Trusting shape requires more than counting hilltops.

`Q_shape` is the weighted shape-component score, computed as above, with public default threshold `quality.morph_threshold=0.65`. Actual `Q_morph` passes when:

`Q_morph_pass = Q_rate_pass AND Q_shape_pass`.

Thus even a high shape score fails `Q_morph` if the counting endpoint fails, while retaining `Q_shape.score` as its score. Nonidentity reducer routes set `Q_shape/Q_morph=NOT_APPLICABLE` with empty scores and thresholds, not zero scores.

| Q_shape component | Default weight |
|---|---:|
| template_correlation | 0.30 |
| skewness | 0.08 |
| pearson_kurtosis | 0.08 |
| red_ir_agreement | 0.18 |
| cardiac_concentration | 0.16 |
| nonflat_scale | 0.04 |
| source_coverage | 0.12 |
| flatline | 0.03 |
| clipping | 0.02 |
| saturation | 0.02 |
| long_gap | 0.02 |

Configuration is `quality.morph_component_weights`. Not executed in finalcase.

### 3.3.4 Fixed-Formula and Training-Data-Quantile Routes

Source: `signal/sqi.py:413–483, 623–645`; call site `experiment.py:567–620`.

**Fixed formulas, `fixed_formula_thresholds_v1`.** Intuition: use the same engraved ruler every time rather than redraw its scale for each participant batch. It uses the divisors, exponentials, and endpoint thresholds above, without cross-record fitting. This is the `SqiConfig.calibrator` constructor default.

**Empirical quantiles, `outer_train_empirical_quantiles_v1`.** Intuition: use only the training group to locate typical lower and upper endpoints of the ruler. A person contributing many strips must not receive more total influence than another.

Step-by-step:

1. `_fit_quality_calibrator` selects only direct waveforms from participants in `train_ids`, computing base fixed-formula scores over 8-second/2-second assessment windows. Test windows cannot define the ruler.
2. Give each finite component observation `x_j` weight `1/n_p`, where `n_p` is that participant's finite row count for that component. Each participant's total component weight is 1.
3. Stable-sort values and assign position `t_j=(cumulative_weight−0.5w_j)/total_weight` along the ruler.
4. Linearly interpolate at `q=0.10` and `q=0.90` to obtain `L,H`. Public configuration is `quality.calibrator_quantiles: [0.10,0.90]`.
5. At application, compute `s'=clip((s−L)/(H−L),0,1)`. If `H≤L`, merely clip the original base score; a component without fitted bounds remains unchanged.
6. Save `fitted_on_participant_ids`; no test participant may appear in this list. Frailty labels are unused, and no refitting occurs during testing/inference.

One implementation detail matters: rate components first receive `rate.*` mappings. Shared morphology components such as RED/IR agreement reuse those rate components and then receive `morph.*` mappings. Shared components can therefore undergo two mappings in sequence; it is incorrect to describe every morphology component as mapping an untouched base score once.

### 3.3.5 diagnostics_only Versus Main-Workflow Recording

Source: `signal/sqi.py:883–975`, `quality/routing.py:234–288`, `experiment.py:888–1014`.

`evaluate_quality_diagnostics` produces only raw observations and availability states, with no classification decision or training-quantile fit. It is the default evaluator for `run_quality_mode(..., diagnostics_only)`.

However, the current window-level main orchestrator's `diagnostics_only` branch also calls fixed-formula `evaluate_quality` once to place Q values in the timeline while retaining raw diagnostics. Timeline grading ignores these Q values in this mode, but enabled motion evidence can still independently affect routes. This mode does not also disable motion and denoising.

Also, all routing switches off takes the direct-retention fast path, while `diagnostics_only` builds the complete assessment-window timeline. Structural handling of recordings shorter than an assessment window and tails without evidence follows the timeline. “Diagnostic scores do not select” therefore does not establish samplewise equivalence of both full paths without inspecting assessment coverage.

## 3.4 Quality and Motion Jointly Determine Routing

### 3.4.1 Excellent / Acceptable / Unfit Truth Table

Source: `quality/routing.py:54–156`, `route_quality_tier`.

Intuition: first ask whether sufficient evidence supports the segment, then whether only hilltops can be counted or their contours can also be trusted. Missing motion evidence must not be treated as low motion.

| SQI used for decisions | Q_rate | Q_morph | Motion detection | Preliminary tier |
|---|---|---|---|---|
| Off | Unused | Unused | Off or low | Excellent |
| Off | Unused | Unused | high | Unfit |
| On | Non-PASS | Any | Any | Unfit |
| On | PASS | PASS | Off or low | Excellent |
| On | PASS | Non-PASS | Off or low | Acceptable |
| On | PASS | Any | high | Unfit |
| Any | Any | Any | Enabled but no valid evidence | Unfit |

This function neither trains nor denoises, and Unfit does not automatically mean permanent deletion. The next section determines whether recovery is attempted once. Tiers are not frailty classes or medical diagnoses.

### 3.4.2 One Recovery Attempt and the Meaning of Rate-Only

Source: `quality/routing_timeline.py:86–181`, `resolve_routing_evidence`; actual call at `experiment.py:1026–1087`.

Structural failures become Excluded immediately. Otherwise a reducer is requested only when `denoiser_enabled` is true, preliminary tier is Unfit, and SQI routing or motion detection is actually enabled.

Intuition: try smoothing a crumpled sheet once. Do not repeatedly change methods until the result looks pleasing, and do not claim all tiny creases were faithfully restored merely because the sheet looks flatter.

Recovery requires algorithm success, an aligned output time axis, and post-cleanup `Q_rate` PASS together. Success reaches at most Acceptable using `x_ar_400`; cleaned amplitude/morphology cannot regain a fidelity claim.

With `quality.mode=off`, only explicitly allowed `feature_vector + artifact.degraded_policy=denoise_then_extract_rate_features` supports “high motion → cleanup → reassess Q_rate” recovery. raw/matrix/fusion do not silently enable SQI recovery via this special switch; `diagnostics_only` does not upgrade tiers using cleaned results.

Failure, unsupported recovery, or failed reassessment gives Excluded. The main workflow retains separate direct/processed views and interval ownership rather than stitching them into a supposedly uniformly faithful waveform.

### 3.4.3 Turning Overlapping Assessment Windows Into Unique Time Ownership

Source: `quality/routing_timeline.py:51–83, 234–356`.

Intuition: each observer sees 8 seconds and observers stand 2 seconds apart, so their views overlap. Divide ownership at the midpoint between neighboring observer positions so every time belongs to one observer without duplicate ownership.

`W=round(8fs)=3200` and `H=round(2fs)=800`. Complete-window starts are `0,H,2H,…≤N−W`, without zero-padding or an automatically appended right-aligned tail window. Centers are `c_i=(start_i+stop_i)/2`; ownership boundaries are `round((c_i+c_(i+1))/2)`, using `np.rint`.

Ownership begins at the first window's start and ends at the last window's stop. Uncovered leading/trailing edges are separately Excluded; no complete window means no assessment evidence for the entire recording. Ownership intervals carry the source window's Q values, motion probability, thresholds, denoising status, and provenance without averaging probabilities again.

### 3.4.4 How Representations Consume Routes

Source: `experiment.py:1099–1118`; `quality/routing_timeline.py:359–379`.

At initial recording selection, raw and fusion require an Excellent direct interval. Feature-vector and feature-matrix can contain Acceptable intervals, but feature validity remains governed by the next chapter's feature/mask rules. Acceptable does not mean every feature is allowed.

For a target matrix-row interval, `matrix_row_route` gathers every ownership interval overlapping by positive duration. Any Excluded interval makes the row unusable; all Excellent/direct yields Excellent; otherwise any Acceptable yields Acceptable. This is neither majority voting nor duration-weighted tier averaging.

## 3.5 Separate Optional Raw-Window Selection

### 3.5.1 `none`: No Calculation or Selection

Source: `quality/window_selection.py:23–67, 190–221`.

Intuition: retain every already-cut strip. There is no invented perfect quality score; this selection module simply does not execute.

Defaults are `policy=none`, `keep_fraction=1.0`, and `application_scope=outer_train_only`. Return original `RawWindows` and counts without historical score calculation or training-statistics fitting. finalcase uses this mode.

### 3.5.2 Historical Window Score: First Establish a Common Scale

Source: `quality/window_selection.py:70–166`, `legacy_window_sqi_scores`.

Input is `[K,C,T]` with temporal validity mask `[K,T]`, initially converted to float32. This scorer creates its own all-channel standardized view and does not modify the configured model-input tensor.

Intuition: center each strip and make heights comparable before judging rhythm and body motion. A sensor must not gain influence merely because its units have larger numbers.

1. Valid length is the final True mask index plus one, not concatenated True samples. Fewer than 16 samples score zero.
2. Per-channel center is `m=median(x)` and candidate scale `r=(Q75−Q25)/1.349`; use r when `r>10⁻⁶`, otherwise population standard deviation.
3. `z=clip((x−m)/(scale+10⁻⁶),−8,8)`. This constant is separate from configurable normalization in the raw module.
4. Choose the higher-standard-deviation RED/IR channel, with IR winning ties.
5. When corresponding channels exist, use `a=norm(z[2:5])` and `g=norm(z[5:8])`; otherwise use zero for the missing quantity.

finalcase does not execute this scorer.

### 3.5.3 Four Components of the Historical Window Score

Source: `quality/window_selection.py:119–152`.

**Cardiac-band fraction.** Welch on the selected optical channel uses `nperseg=min(512,T)`. `S=trapz(P_0.5…3.0,f)/[trapz(P_all,f)+10⁻¹²]`. This trapezoidal integration differs from frequency-bin summation in 3.2.2.

**Interval regularity.** Run `find_peaks` on `(ppg−median)/[std+10⁻⁸]` with minimum distance `round(0.28fs)` and prominence 0.3. For at least three peaks, `D=1/[1+std(diff(peaks))/(mean(diff(peaks))+10⁻⁸)]`, otherwise zero. More similar neighboring hilltop distances appear more regular.

**Peak-count clue.** With at least three peaks, `P=min(1,K_peaks/max(2,3T/fs))`, otherwise zero. This rewards enough peaks for the observation duration, not the 30–200/minute interval function of 3.2.5.

**Motion penalty.** `M=RMS(a)+0.25RMS(g)` and `A=1/[1+max(0,M−1)]`. Here a/g are window-standardized values, not physical-unit motion RMS.

Final `score=0.40S+0.35D+0.15P+0.10A` replaces nonfinite values with zero. Scores within each file are rescaled between their 5th and 95th percentiles as `clip((score−q5)/(q95−q5+10⁻⁸),0,1)`; identical scores are not rescaled. Scaling/clipping can create ties, so preserve the implemented sorting behavior rather than claim equivalence to arbitrary stable sorting.

These constants belong to the named historical algorithm `legacy_cardiac_motion_window_sqi_v1`. Public selection parameters configure policy, retention fraction, and scope, not all four historical coefficients individually.

### 3.5.4 Retaining the Best Fraction Within Each File

Source: `quality/window_selection.py:167–188, 244–291`.

Intuition: select strips separately for each person's recording, rather than allowing one person's good strips to displace all of someone else's strips.

For K file scores retain `max(1,ceil(K·keep_fraction))`, keeping empty input empty. `np.argsort(scores)[::-1]` determines ranking; a boolean mask then selects values, validity, starts, and scores together. Output therefore preserves chronological order, not quality rank order.

`keep_fraction` is in `(0,1]`, default 1.0. Statistics are within-file only, with no labels or cross-file/cross-fold fitting.

### 3.5.5 Three Noninterchangeable Application Scopes

Source: `experiment.py:1596–1644`; `quality/window_selection.py:293–342`.

| `application_scope` | Training windows | OOF window prediction rows | OOF aggregation |
|---|---|---|---|
| `outer_train_only` | Top fraction | Retain all | This policy removes no windows; independent quality weighting may read scores |
| `all_partitions` | Top fraction | Top fraction; unselected windows never enter prediction | Retained windows |
| `legacy_train_and_aggregation` | Top fraction | Predict and save all | Only top-fraction `window_aggregation_mask` entries are True |

The third scope uses `mark_raw_windows_for_aggregation`, which marks without deleting prediction rows. Continuous quality weighting remains independently controlled by `aggregation.quality_weighting` and `quality_weight_source=legacy_window_sqi`. Within-file top fractions differ from historical scripts that ranked pooled windows from all recordings of one participant.

## 3.6 Motion Detection Learns Activity Protocols, Not Optical-Artifact Ground Truth

### 3.6.1 Training Labels and Input Routes

Source: `motion_activity_label` at `quality/motion.py:253–255`; `quality/motion_adapters.py:41–101`; `quality/motion_reference.py:357–464, 659–773`.

Intuition: show an observer examples of sitting/static-protocol actions versus walking/activity so it can distinguish activity states. Labels come from task instructions, not samplewise manual annotations of optical corruption; this is not an optical-artifact ground-truth detector.

Internal labels are `B/R→0` and `S/W→1`; PTT uses `sit→0` and `walk/run→1`. Three-class frailty labels are not used for this binary target.

Production motion-model input is `[batch,8,3200]`: RED, IR, three gravity-removed acceleration axes, and three angular-velocity axes at 400 Hz, with 8-second windows and 2-second hops. Construction and within-training-fold IMU scaling are explained under `representations/motion.py` in the representation chapter. RED/IR come from raw optical columns, not the filtered columns used by frailty raw models; the two input routes must not be conflated.

Internal participants use their own static B for calibration; PTT participants use their own sit recording. Production motion materialization uses calibrated roll/pitch EKF, not finalcase's `sensor_filter_only_no_gravity_removal`. finalcase disables motion detection and does not require these additional network computations during classification.

### 3.6.2 Light CNN First Layer: Sliding Small Shapes Through Time

Source: `models/motion.py:110–127`, `LightCnnMotionDetector.__init__`.

Intuition: slide many small rulers along the curves, each sensitive to a particular local fluctuation pattern. Each ruler can inspect several sensor curves together rather than only one point's height.

The first layer is `Conv1d(C,12,kernel_size=9,padding=4)`:

`y[o,t]=b[o]+Σ_(c=1…C)Σ_(j=0…8) W[o,c,j]x[c,t+j−4]`.

Conv1d zero-padding at both ends preserves length 3200. Then `GroupNorm(1,12)` computes mean/population variance over all 12 channels and time positions separately for each sample and applies `z=γ(y−μ)/sqrt(σ²+ε)+β`. No explicit `ε` is supplied, so the installed PyTorch layer default applies. This is not BatchNorm: test samples do not share statistics, and there is no training moving mean.

`GELU` softly suppresses small negative responses, mathematically `zΦ(z)`, using default `nn.GELU()` behavior. `AvgPool1d(2)` then averages neighboring pairs, reducing length to 1600.

### 3.6.3 Second/Third Layers and Output: Combining Longer Motion Patterns

Source: `models/motion.py:120–140`.

Intuition: the first layer recognizes small bends; subsequent layers combine them into longer motion patterns, and the output judges which activity the whole window resembles.

Second layer: `Conv1d(12,24,7,padding=3)` → `GroupNorm(1,24)` → `GELU` → `AvgPool1d(2)`, giving `[batch,24,800]`.

Third layer: `Conv1d(24,24,5,padding=2)` → `GroupNorm(1,24)` → `GELU`, retaining 800 time positions. `AdaptiveAvgPool1d(1)` averages each feature curve over the window to 24 numbers; `Flatten` and `Linear(24,1)` output one real value `l`.

The predictor at `quality/motion_adapters.py:372–388` uses `p_active=1/(1+exp(−l))`, returning dimensionless `[batch]` probabilities. No extra Platt/isotonic calibrator is fitted after sigmoid; metadata wording `calibrated_p_active_probability` does not prove an additional calibration layer exists.

### 3.6.4 Three Named Architectures and One Development Constructor

Source: `models/motion.py:148–212`; parameter-count formula at `48–89`.

**Production eight-channel.** `build_formal_motion_cnn()` fixes two optical plus six IMU axes, base width 12, and 5,965 trainable parameters. It binds the production channel order and 3200-sample-window description.

**Eleven-channel derived augmentation.** `build_motion_derived_augmentation_cnn()` adds motion magnitude, rotation magnitude, and movement-change rate to the production eight channels, totaling 6,289 parameters. Intuitively it presents clearer aggregate curves derived from existing axes, not three new sensors. The named motion-only ablation constructor is available, but the default formal trainer still accepts only eight channels. Constructor availability does not mean the current formal runner trains eleven channels.

**Historical ten-channel backup.** `build_historical_light_cnn_backup()` uses one historical optical column, six physical axes, and three derived quantities, totaling 6,181 parameters with base width 12. The archive corresponds to 256 Hz, 8-second/2-second windows, and threshold 0.05. The constructor builds only the network, not historical weights, and does not promote historical external SIM scores into current PTT or historical V5 validation results.

**Development constructor.** `build_parameterized_light_cnn(channel_names,base_channels=12)` accepts ordered channel names and width, but the implementation still permits only convolution lengths 9/7/5. For input channel count C and base width b, parameter count is:

`(9Cb+3b)+(7b·2b+6b)+(5·2b·2b+6b)+(2b+1)`.

This includes convolution biases, GroupNorm scale/shift, and the output head. None is finalcase's main classifier, which is a separate Inception small network.

### 3.6.5 Training Sampling: Balancing Rare Classes and Participants With Many Windows

Source: `quality/motion_adapters.py:170–183, 253–266`.

Intuition: a very common class or prolific participant must not occupy the whole classroom every epoch. Reduce their relative sampling chance, without promising exactly equal counts for every person in every epoch.

Weight for window i:

`w_i=1/[n_(dataset_i,class_i)·sqrt(n_participant_i)]`.

Divide by the mean weight across all windows to make average weight 1. `WeightedRandomSampler` samples `len(examples)` windows with replacement using these weights. Each epoch has the same number of draws but may repeat some windows and omit others. Default seed is 42 and batch size 16.

Class weights are not additionally applied in the loss: `class_weighting=none_balancing_is_sampler_only`. The sampler uses counts from this fit's training windows only.

### 3.6.6 Network Learning: Ten Fixed Epochs Without OOF Early Stopping

Source: `quality/motion_adapters.py:117–145, 232–369`.

Intuition: after each example batch, adjust the small rulers to better match known activity labels. Study exactly ten times, without inspecting examination scores to choose the most flattering epoch.

Training inputs first receive IMU center/scale transforms fitted on this fold's training participants; per-window RED/IR processing occurs during representation construction. For logit l and label y, the equivalent samplewise BCE-with-logits formula is:

`L=max(l,0)−l·y+log(1+exp(−|l|))`.

The code averages batch loss, backpropagates, clips total gradient-vector length to `gradient_clip_norm=1.0`, then updates with Adam. Defaults are `learning_rate=0.001`, `weight_decay=0.0`, `fixed_epochs=10`, `dropout=0.0`, `label_smoothing=0.0`, no augmentation, and `num_workers=0`. The formal trainer checks these fixed choices; device may be CPU or CUDA. This is not an arbitrary model-tuning entry point.

Each epoch records training loss and training balanced accuracy at threshold 0.5. The latter is historical logging only, not the frozen motion threshold below or an epoch-selection criterion. Only final epoch-ten weights are saved, without OOF/external-PTT epoch selection.

### 3.6.7 Within-Fold Threshold: Midpoint of Two Typical Class Scores

Source: `quality/motion.py:307–355`, `fit_train_only_midpoint_threshold`; `quality/motion_runner.py:555–582`.

Intuition: each person first reports their typical static and active scores; then combine people using typical values. Set the boundary halfway between class centers rather than searching the test group for an attractive threshold.

After fitting the fold network, infer on that fold's training windows. For each training participant p and class c:

`m_(p,c)=median_i p_active(i)`, using only that participant's windows of that class.

`m_c=median_p m_(p,c)`; final threshold `τ=(m_0+m_1)/2`.

Every training participant must have both activity classes and `0≤m_0<m_1≤1`. Reversed/equal centers fail rather than silently replacing the threshold with 0.5. Test windows with `p≥τ` are motion=1, including equality; otherwise static=0.

### 3.6.8 Deployment Threshold: Complete OOF, Not All-Data Self-Predictions

Source: `quality/motion_runner.py:584–606, 609–724`.

Intuition: for actual deployment, the boundary should still reflect scores from models that have not seen that person, not an all-person model grading its own homework.

The internal formal protocol uses 29 participants, one participant-grouped five-fold split, and seed 42. Each network and fold threshold read only training participants; each person appears once as outer OOF. An all29 deployment model is trained afterward, but deployment threshold is recomputed from all strict OOF windows as “per-participant class medians → cross-participant medians → midpoint of two centers.” External PTT never adjusts the threshold.

Thus within-fold thresholds from training-fit predictions and deployment thresholds from strict OOF are distinct stages; neither “all train-fit” nor “all OOF” describes both.

### 3.6.9 External PTT and Reverse-Direction Training Comparison

Source: `quality/motion_reference.py:914–984`; `quality/motion_runner.py:928–1007, 1100–1204`.

PTT contains 66 sit/walk/run recordings from 22 participants. Its adapter first converts them to 400 Hz while preserving optical/IMU synchronization. Current unit evidence identifies PTT acceleration as m/s², so do not multiply by 9.81 again; angular velocity converts deg/s to rad/s. Each person's sit recording supplies static calibration.

For internal-training → PTT evaluation, model and threshold remain fixed, with no PTT fitting of center/scale/threshold. The reverse ablation trains on PTT's predefined repeat-0 grouped five folds, generates OOF and a deployment threshold, refits on all PTT, then performs inference only on Frailty29. Each direction retains its own transforms, labels, and training-set membership.

`sqi_only` and `sqi_plus_motion_override` in `MotionOptionId` describe historical protocols, not two further network algorithms. Actual classifier use of motion detection is determined by `artifact.motion_detector_enabled` and the executable bundle-adapter path.

### 3.6.10 Reusing a Motion Bundle in the Classification Pipeline

Source: `quality/motion_bundle_adapter.py:49–104, 244–385, 393–521`.

Intuition: employ an already-trained observer together with the ruler and boundary used during training. Taking only its weights while changing input scale is not equivalent.

Defaults are `enabled=false`, `device=cuda`, `batch_size=64`, `threshold_source=bundle_frozen`, and `reuse_scope=all29_smoke_or_final_only`, with empty evidence paths/hashes. Enabling requires an existing bundle; a missing model is not trained automatically.

Three reuse scopes:

- `matching_outer_fold_or_all29_final`: with outer OOF, match the Stage5 fold model and training threshold exactly by training-participant list; use all29 only without outer OOF.
- `all29_smoke_or_final_only`: final/smoke use only; all29 cannot be used within classification folds containing OOF.
- `all29_frozen_in_sample_auxiliary`: explicitly labeled in-sample auxiliary comparison is allowed, but `valid_outer_oof_claim=false`; it is not leakage-free OOF.

Window probabilities use the native 8-second/2-second grid: `p<τ` is low and `p≥τ` high. Recording median `median(p_windows)` is diagnostic only; current window routing does not substitute it for individual probabilities. Missing signals, no complete 8-second window, or invalid probabilities return unavailable/unfit evidence.

The adapter requires the same calibrated EKF inputs as training and rejects gap-repaired native PPG. Merely turning motion on in finalcase YAML while retaining no-gravity inputs does not automatically create compatible bundle use.

### 3.6.11 Motion Evaluation: Score Each Person First, Then Average Equally

Source: `quality/motion_runner.py:282–369`.

Intuition: score each person separately, then average people, rather than giving longer recordings more influence. These computations evaluate and do not modify the trained model.

**Two-class recognition.** Apply the frozen threshold to obtain 0/1 predictions. Treat static and motion in turn as the target class: `recall_c=TP_c/(TP_c+FN_c)`, `precision_c=TP_c/(TP_c+FP_c)` (zero if no predicted examples), and `F1_c=2precision_c·recall_c/(precision_c+recall_c)` (zero for zero denominator). `balanced_accuracy=mean(recall_0,recall_1)`, `macro_f1=mean(F1_0,F1_1)`, `sensitivity=recall_1`, `specificity=recall_0`. Both true activity classes are required; do not fabricate AUC for a person with only static recordings.

**Rank-based ROC AUC.** Intuition: for a random active strip and static strip, does the active strip usually score higher? `_rank_average` ranks ascending probabilities and gives ties average ranks. With positive count `n1`, negative count `n0`, and positive rank sum `R1`, `AUC=[R1−n1(n1+1)/2]/(n1n0)`; ties count as half a win.

**PR AUC is actually average precision.** Intuition: lower the cutoff gradually from the highest scores; each time more truly active strips are recovered, inspect the purity of selected strips. Stable descending sorting evaluates recall/precision at each tied-score group's end, then `AP=Σ_k(recall_k−recall_(k−1))precision_k`. This is not trapezoidal precision–recall integration; replacing it with another “PR AUC” implementation need not be numerically identical.

**Probability error ECE.** Split `[0,1]` into ten equal bins and compare each bin's mean `p_active` with observed active fraction: `ECE=Σ_b(n_b/N)|mean(p_b)−mean(y_b)|`. The first nine bins are left-closed/right-open; the last includes 1. This is active probability versus active frequency, not a classification-confidence version using `max(p,1−p)`.

**Participant aggregation and worst fold.** A participant's evaluation rows must use one frozen threshold. Compute all metrics separately per participant, then average participants equally. `worst_fold_balanced_accuracy` is the minimum existing fold summary score. `parameter_count` counts all `requires_grad` elements. Inference timing is measured separately: by default ten warm-ups, fifty measured calls, batch=1, reporting per-window milliseconds P50/P95 and `1000/mean_milliseconds` windows/second. CUDA is synchronized around timing to avoid measuring only asynchronous submission.

## 3.7 Shared Interface for All Artifact Reducers

Source: `artifacts/base.py:19–211`, `artifacts/router.py:79–107, 135–169`.

Input `ppg[N,2]` contains finite float64 RED/IR at 400 Hz, with synchronized IMU dictionary when needed. Successful output includes `x_ar[N,2]`, confidence, validity, and method diagnostics, preserving the original sampling grid. Failed/unsupported results return `x_ar=None`; routing neither silently falls back to direct nor chains two nonidentity reducers.

Intuition: every cleanup tool receives two equally long, aligned paper strips and returns aligned strips or an explicit failure. It cannot discard a section while claiming identical positions.

Except for identity, successful results are rate-only. Two output columns do not prove preservation of original amplitudes, independent physiological meaning, or pulse shape. `confidence` is an algorithm-specific diagnostic, not a universally calibrated probability of correct denoising; values are not directly comparable across methods.

Each reducer receives `artifact.parameters` through its dedicated dataclass. Within-record/segment decomposition, scaling, and adaptive weight updates are not cross-participant classifier training. They may nevertheless use future samples from the full segment, so these are offline static-data methods, not necessarily samplewise causal online algorithms.

“Aliases” below are accepted by low-level `artifacts.router.get_reducer`. Production `artifact.reducer` uses ten canonical IDs from `module_registry.py:671–703`, not every low-level alias. learned/hybrid/ONNX are explicit unsupported low-level branches, not runnable formal-registry modules.

### 3.7.1 Shared IMU References: Six Axes or Nine-Channel Augmentation

Source: `artifacts/base.py:55–123`, `imu_reference_matrix`.

Intuition: arrange three movement directions and three rotation directions as six rulers. Center and scale each ruler using genuinely valid data in this segment so unit magnitude does not determine influence.

Production `imu_axes6_reference_v2` uses `dynamic_acc_mps2[N,3]` and `gyro_rads[N,3]`. Augmented `imu_axes6_plus_derived3_augmentation_ablation_v2` adds `dynamic_magnitude`, `gyro_magnitude`, and `jerk_magnitude`.

Over valid rows j, compute per-column `μ=mean(r_valid)` and `σ=std(r_valid,ddof=0)`, dropping constant columns with `σ≤10⁻¹²`. Standardize valid rows as `(r−μ)/σ`; invalid rows receive temporary zeros but retain a separate returned mask. Require at least 32 valid rows and one nonconstant column, otherwise fail.

Invalid-row zeros are numerical placeholders, not observed zero motion; downstream algorithms must propagate the mask or fail. NLMS, spectral masking, and PCA/FastICA source selection use this reference. SSA, EMD, CEEMD-lite, DWT, and NMF do not.

## 3.8 Identity: Unmodified Direct Comparison

Source: `artifacts/identity.py:13–41`; registered `identity`, aliases `none/direct`.

Intuition: photocopy both strips unchanged without cleaning or selection.

`output=source.copy()` gives `output[n,c]=source[n,c]`, confidence=1, empty parameters, and `max_absolute_change=0`. Copying isolates array mutations; it is not a numerical conversion. Invalid input still fails.

Identity preserves the direct/identity route rather than becoming rate-only. finalcase configures identity, but with all routing modules off, the fast path need not call it again for every recording to reproduce the same signal.

## 3.9 NLMS: Estimating the Curve to Subtract From Body Motion

Source: `artifacts/nlms.py:29–136`, `NlmsReducer.reduce`; registered `nlms_imu_anc`, alias `nlms`.

Intuition: body motion often leaves a shadow on the optical curve. Mix current and slightly earlier motion curves in different amounts to estimate that shadow, then subtract it. Adjust the mixture slightly after each new point.

Defaults: `taps_per_delay=8`, `delay_taps=[0,4,8,16]` samples, `step_size=0.15`, `epsilon=1e-6`, `leakage=1e-5`, `update_gate_reference_rms=0.10`, and six-axis IMU reference.

Step-by-step mathematics:

1. Obtain standardized reference `r[n]` as above. Delay set `D=sorted({d+t:d∈delay_taps,t=0…7})` has default union `0…23`, not 32 distinct taps. With all six axes nonconstant, dimension is `24×6=144`, not `4×8×6=192`; dropping constant axes reduces it.
2. Assemble `v_n=[r[n−d] for d∈D]` and initialize separate wavelength weights `W[2,dim]=0`.
3. If any delayed IMU position is invalid, neither predict nor update; temporarily retain the original numerical output but mark its mask False. The first 23 points likewise lack complete delayed context.
4. Estimate motion shadow `ŷ_n=Wv_n`; output residual `e_n=x_n−ŷ_n` and mark the position valid.
5. If `sqrt(mean(v_n²))≥0.10`, update `W←(1−leakage)W+step_size·e_n v_nᵀ/[epsilon+v_nᵀv_n]`; otherwise retain current weights.
6. Diagnostic explained variance is per-channel `clip(1−var(output)/max(var(source),1e-12),0,1)`, with confidence the channel mean. It is not manually annotated cleanliness.

`step_size` must lie in `(0,2)`. Missing IMU, all-constant references, and similar conditions fail. Motion and real heartbeat may vary together, so subtracting a motion shadow can also remove physiology; morphology fidelity is not promised. Not executed in finalcase.

## 3.10 SSA: Decomposing and Selecting Repeated Patterns

Source: `artifacts/decomposition.py:22–139`; registered `ssa_decomposition`, aliases `ssa/decomposition`.

Intuition: cut many slightly shifted strips from one curve and arrange them as a rectangle. Repeated fluctuations form shared patterns within it. Decompose those patterns and add back only ones with suitable speeds.

Defaults: `embedding_samples=160`, `max_components=12`, `minimum_cardiac_concentration=0.45`, `cardiac_low_hz=0.5`, `cardiac_high_hz=3.5`.

1. Process wavelengths independently. Actual strip length is `L=min(160,N//3)`; construct trajectory matrix `H[i,j]=x[i+j]` with shape `[L,N−L+1]`.
2. `np.linalg.svd(H,full_matrices=False)` yields `H=U diag(s) Vᵀ`. This is complete thin SVD, not a solver computing only twelve components; `max_components` limits subsequent inspection only.
3. Component matrix `E_i=s_i·outer(U[:,i],Vᵀ[i,:])` is averaged over entries sharing original time `i+j=n` to recover a length-N component. `_diagonal_average` implements rowwise accumulation divided by counts.
4. Apply Welch to each component with `nperseg=min(1024,N)`, then divide summed `0.5–3.5 Hz` power by summed `0.2–8 Hz` power.
5. Sum components with concentration `≥0.45`. If none qualify, fail rather than force selection of the best one.
6. Reconstruct wavelengths separately and place side by side. Confidence is the mean of each wavelength's highest candidate concentration, not the fraction of retained components.

No IMU or cross-record fitting is required. Insufficient matrix length, decomposition failure, or no qualifying components fails. Not executed in finalcase.

## 3.11 Spectral Mask: Suppressing Optical Frequencies With Strong Motion

Source: `artifacts/spectral.py:25–222`; registered `spectral_mask`, aliases `spectral/stft/stft_imu_mask`.

Intuition: draw a picture with time horizontally, fluctuation speed vertically, and strength as color. Where motion is strong at a particular time/speed, dim the matching optical region. Inside the retained band, do not erase it entirely; clear everything outside the band.

Defaults: `stft_window_s=4.0`, `stft_hop_s=1.0`, `imu_mask_quantile=0.75`, `mask_strength=0.80`, `preserve_band_hz=[0.5,3.0]`, and six-axis IMU reference.

1. Convert 4 seconds to 1600 samples and requested 1-second hop to 400. Short input uses `nperseg=min(1600,N)`, actual hop `min(400,nperseg)`, and overlap equal to their difference. Require at least 32 samples. Use a `Hann` window, zero-extended boundaries, and tail padding to complete STFT frames.
2. Compute each IMU-axis STFT magnitude `|R_c(f,t)|` and combine as `M=sqrt(mean_c |R_c|²)`.
3. For each time frame, obtain the 75th percentile across all frequencies, `q_M(t)`, then `M'=M/max(q_M,1e-12)`.
4. Compute each optical-channel STFT similarly: `P'=|X|/max(percentile_95_f(|X|),1e-12)`.
5. Relative contamination is `C=M'/max(M'+P',1e-12)`.
6. In-band gain is `G=clip(1−0.8C,0.2,1)`; 0.2 equals `1−mask_strength`, not another hidden parameter. Outside `0.5–3.0 Hz`, `G=0`.
7. Multiply complex STFT directly by G, preserving surviving phase; reconstruct with ISTFT and trim to N samples. Reconstruction shorter than N fails.
8. Conservatively expand invalid IMU rows by the full window length. Output remains aligned, but affected positions are unusable; fewer than 32 valid samples fails.
9. Confidence is the two-channel mean of nonnegative-clipped input/output correlations over valid positions. `1−mean_gain` is separately recorded as suppression fraction; more suppression does not imply greater reliability.

The retained band must contain at least two STFT bins and stay within Nyquist. Output is narrow-band and rate-only. Not executed in finalcase.

## 3.12 PCA: Rotating Shared Dual-Optical Variation Into Two New Directions

Source: `artifacts/bss.py:85–93, 177–224, 241–330`; registered `pca_bss`, alias `pca`.

Intuition: treat RED/IR at each time as a point in a plane. The point cloud may stretch diagonally; rotate the coordinate sheet to describe variation along two new directions. Select the direction most like heartbeat and least like motion.

1. Require synchronized wavelengths and rank 2 after mean-centering. Single-channel or perfectly linearly dependent channels fail.
2. `PCA(n_components=2,svd_solver="full").fit_transform(X)` applies SVD to `X−μ`, yielding components `S=(X−μ)V` and mixing `V`.
3. For each component calculate `C_i`, the Welch `0.5–3.5 Hz / 0.2–8 Hz` power ratio.
4. Over valid IMU rows, compute absolute correlation with every nonconstant reference axis and take maximum `R_i`.
5. Select `argmax_i(C_i−0.25R_i)`. There is no separate minimum-concentration threshold, unlike SSA's required 0.45.
6. Back-project one direction into two channels: `X_hat=outer(S[:,i],V[:,i])+μ`. Output validity follows the IMU mask.

The only dedicated setting is `imu_reference_profile`, default six axes. PCA does not use random_state, iteration counts, or NMF rank; supplying them does not make them effective. Confidence is the selected component concentration clipped to `[0,1]`. Not executed in finalcase.

## 3.13 FastICA: Seeking Two Less-Dependent Sources

Source: `artifacts/bss.py:95–107, 241–320, 332–339`; registered `fastica_bss`, aliases `ica/fastica`.

Intuition: two microphones each record two speakers. Beyond rotating toward maximum variation, try to find sources whose fluctuations are less dependent. Two optical channels resemble microphones, but whether they can truly be separated this way depends on the data.

Input requirements, source-selection score `C_i−0.25R_i`, and single-source back-projection match PCA. The difference is `_fit` calling `FastICA(n_components=2,whiten="unit-variance",random_state=42,max_iter=1000,tol=1e-5)`.

The standard FastICA kernel centers data and equalizes directional scales, then uses a nonlinear contrast to seek independent directions. This call leaves sklearn algorithm/nonlinearity defaults unchanged. Details belong to the installed sklearn implementation, not a hand-written equivalent in this repository. Mathematically `S=(X−μ)Wᵀ`; the code reads `model.mixing_` as A and retains column i via `X_hat=outer(S[:,i],A[:,i])+μ`.

For dependency-call review, the following expands the locally installed sklearn 1.8.0 `_fastica.py` matching the dependency list. R denotes the iterative direction matrix, not the final complete unmixing matrix:

1. `Z=(X−μ)ᵀ` has shape `[2,N]`. Apply SVD `Z=U diag(d)Vᵀ` and fix U's signs; `K=(U/d)ᵀ` is the preprocessing map.
2. `Z_w=KZ·sqrt(N)` is the equal-scale input used during iteration.
3. Seed 42 generates an initial 2×2 normal random R. Apply `R←(RRᵀ)^(-1/2)R` to make directions nonredundant. Eigen decomposition clips very small eigenvalues to dtype tiny to avoid division by zero.
4. Defaults are `algorithm="parallel"`, `fun="logcosh"`, and `alpha=1`. For `Y=RZ_w`, `g(Y)=tanh(Y)` and `g'_i=mean_n(1−tanh(Y_i,n)²)`.
5. `R_new=mean_n[g(Y)Z_wᵀ]−diag(g')R`, followed by the same symmetric decorrelation.
6. Stop when `max_i ||dot(R_new_i,R_i)|−1| < tolerance`. Absolute values prevent sign flips from appearing to be distinct solutions. Otherwise run at most 1000 iterations; nonconvergence raises a warning that this reducer treats as failure.
7. Original-scale sources are `S=(RKZ)ᵀ`. `whiten="unit-variance"` then rescales S and R by each S column's standard deviation. Final `components_=RK` and `mixing_=pinv(components_)`.

No library source is copied into the pipeline and no call is changed here. After dependency upgrades, check these default-kernel details against the newly installed source.

Actual controls are `random_state=42`, `max_iter=1000`, `tolerance=1e-5`, and `imu_reference_profile`. `ConvergenceWarning` is caught and returned as failure, not success or an automatic PCA fallback. Not executed in finalcase.

## 3.14 NMF: Building Two Optical Spectra From Additive-Only Blocks

Source: `artifacts/bss.py:109–128, 342–444`; registered `nmf_bss`, alias `nmf`.

Intuition: construct two time-versus-speed images from nonnegative color templates that only add brightness, never cancel through opposite signs. Select the template most concentrated at heartbeat speeds and restore its original temporal fluctuations.

Defaults: `random_state=42`, `max_iter=1000`, `tolerance=1e-5`, `nmf_rank=2`, `nperseg=512`, `overlap_fraction=0.75`.

1. Apply Hann STFT to each wavelength with `nperseg=min(512,N)`, overlap `min(nperseg−1,round(0.75nperseg))`, and boundary/tail zero-padding.
2. Concatenate magnitude images along the frame axis: `M=[|X_RED| |X_IR|]`, shape `[F,2T_frames]`.
3. Effective rank is `min(requested_rank,F,2T_frames)`. `NMF(init="nndsvda",solver="cd")` fits `M≈WH` with nonnegative W/H and default squared reconstruction-error objective. `basis=model.fit_transform(M)` gives W; `model.components_` gives H.
4. Score each basis as `C_i=Σ_(0.5…3.5) W[f,i]/max(Σ_(0.2…8) W[f,i],1e-12)`. This sums nonnegative spectral-basis amplitudes, not Welch power, and does not square first.
5. Select maximum `C_i` and reconstruct single-basis magnitude `outer(W[:,i],H[i,:])`.
6. Split the frame axis back into RED/IR, multiply each by original STFT phase `exp(j·angle(X_c))`, then ISTFT and trim to N samples.

IMU and the PCA/ICA motion-correlation penalty are not used. Single wavelength, short windows, NMF nonconvergence, or insufficient reconstruction length fails. Confidence is the clipped selected-basis ratio; output remains rate-only. Not executed in finalcase.

The two called kernels in sklearn 1.8.0 `_nmf.py`/`_cdnmf_fast.pyx` are expanded below so “fit NMF” does not substitute for mathematical steps:

**NNDSVDA initial blocks.** Apply fixed-seed truncated randomized SVD to M. Take absolute values of the first vector pair and multiply by `sqrt(s0)`. For subsequent pairs, split positive/negative parts, compare the corresponding products of norms, choose and normalize the larger pair, then multiply by `sqrt(s_j·selected_norm_product)`. Set elements below `1e-6` to zero; `nndsvda` finally fills zeros in W/H with `mean(M)`. The a suffix does not add random noise to zeros.

**Alternating coordinate updates.** Without explicit regularization, minimize `0.5||M−WH||²` subject to nonnegative W/H. With H fixed, let `A=HHᵀ` and `B=MHᵀ`. For each W[i,k], compute `gradient=(WA−B)[i,k]` and `hessian=A[k,k]`; if hessian is nonzero, set `W[i,k]←max(W[i,k]−gradient/hessian,0)`. Update H identically after transposing M and swapping W/H roles. Default `shuffle=false` preserves fixed column order.

Each full update accumulates absolute projected gradients: use `min(0,gradient)` for a zero current element, otherwise gradient. The W/H absolute sum is violation. Stop if first-round violation is zero or subsequently `violation/initial_violation≤tolerance`. Thus `tolerance=1e-5` does not guarantee optical-output maximum error ≤1e-5.

## 3.15 EMD Sifting: Peeling Away Successive Midlines

Source: `artifacts/legacy.py:29–34, 67–120, 229–281`; registered `emd_sifting_rate_only`.

Intuition: draw a smooth line across hilltops and another across valleys, then subtract their midline. Repetition extracts a thin texture oscillating around zero. Continue peeling the residual. Finally add extracted layers back without the remaining slow background.

Defaults: `max_imfs=6`, `max_sift=10`, `sd_threshold=0.2`.

1. `_local_extrema` detects maxima from positive-to-negative first-difference transitions and minima from negative-to-positive transitions, excluding positions immediately next to the ends. Flat tops are not supplemented by another peak algorithm.
2. `_sift_mean` requires at least two maxima and two minima, adds both endpoints as knots, and fits natural-boundary cubic upper/lower envelopes `u(t),l(t)`. Their mean is `m(t)=[u(t)+l(t)]/2`.
3. Update the current candidate `h` as `h_new=h_old−m`.
4. Change measure is `SD=Σ(h_old−h_new)²/[Σh_old²+1e-18]`. End the layer when below 0.2 or after ten sifts.
5. Saving a candidate requires at least two extrema in total (112–115), unlike envelope construction's requirement of at least two of each type. Thus insufficient envelopes may end the inner loop early while a not-fully-sifted candidate is still saved as an IMF. Then `r←r−h`. Stop when residual maxima or minima count drops below one, up to six layers.
6. Each wavelength outputs the sum of all IMFs, explicitly excluding residual. This neither selects particular cardiac-band IMFs nor restores residual for a complete inverse decomposition.

No IMF means failure; IMU is unnecessary. Fixed confidence 0.5 is a historical diagnostic default, not estimated success probability. Outputs include `residual_norm_fraction` and mark morphology unguaranteed. Not executed in finalcase.

## 3.16 CEEMD-lite + NLMS: Repeated Small Perturbations, Then Adaptive Subtraction

Source: `artifacts/legacy.py:37–52, 122–227, 282–331`; registered `ceemd_lite_nlms_legacy`.

Intuition: sprinkle slight random sand on an image, then opposite sand, repeating and averaging to stabilize decomposition. Combine non-heartbeat-like layers into an interference reference, then learn sample by sample how much to subtract. Here the reference comes from optical data itself, not IMU.

### 3.16.1 Paired Positive/Negative Perturbations and Averaged Decomposition

Defaults: `pairs=6`, `noise_ratio=0.2`, `random_seed=2025`; EMD still uses six layers, ten sifts, and stopping threshold 0.2.

Reinitialize fixed-seed `default_rng` for each wavelength. Generate `η~N(0,[0.2(std(x)+1e-12)]²)` and apply the preceding EMD to `x+η` and `x−η`. Accumulate IMFs by index; realizations missing a layer contribute no value for it.

Important code detail: the denominator remains `2*pairs=12`, not the count of successful realizations. `successful_realizations` is recorded but never replaces the denominator. Averaged residual is `x−Σaveraged_IMF`. No IMF in any realization means failure.

### 3.16.2 Protect Heartbeat-Like Layers and Combine Suspected Motion Layers

`_welch_peak` uses Hann Welch up to 8 seconds and selects the maximum-power bin in the specified band. Original dominant frequency `f_H` is sought in `0.6–3.5 Hz`; each IMF dominant frequency in `0–8 Hz`.

Defaults are `protect_bandwidth_hz=0.25` and `protect_harmonics=2`. An IMF dominated by `h·f_H±0.25` for h=1,2 is protected. Otherwise a dominant frequency `≤low_motion_hz=0.4` or `≥high_motion_hz=6.0` marks a motion layer.

Other layers are correlated with the original curve after second-order `0.6–3.5 Hz` forward-backward Butterworth filtering; `|corr|≥0.2` is protected, otherwise motion. The reference sums motion IMFs, additionally including residual if its `0–2 Hz` dominant frequency is below 0.4.

Correlation threshold 0.2, residual-frequency threshold 0.4, and protection-filter order/band are internal constants of this historical implementation. The latter 0.4 does not read `low_motion_hz`. Continue only with a finite nonconstant reference.

### 3.16.3 Historical Leaky NLMS Differs From Modern IMU-NLMS

Defaults are `nlms_length=32`, `nlms_mu=0.1`, and `nlms_leak=1e-4`. At each sample insert the new reference at the front of a 32-point buffer and shift old entries rightward; buffer and weights start at zero.

`estimate[n]=wᵀv_n`

`clean[n]=x[n]−estimate[n]`

`w←(1−1e-4)w+0.1·clean[n]·v_n/[1e-6+v_nᵀv_n]`.

Update every sample, without modern IMU-NLMS's RMS update threshold or multiple delay groups. Process wavelengths separately, with fixed confidence 0.5. Missing usable reference or nonfinite output fails rather than substituting a zero reference and claiming success. Not executed in finalcase.

## 3.17 Historical DWT A2: Stretching Coarse-Contour Coefficients to Original Length

Source: `artifacts/legacy.py:55–59, 333–400`; registered `dwt_a2_legacy`.

Intuition: use a fixed small ruler twice to separate fine detail from slower contour. Keep only the remaining coarse-contour numbers, then stretch their horizontal positions back to the original length.

1. For each wavelength call `pywt.wavedec(x,"db4",level=2)`, returning coarse coefficients and two detail levels.
2. Retain only `coefficients[0]`, A2, requiring at least two finite values.
3. Create uniform `[0,1]` coordinates for both A2 and the original N points.
4. Use `np.interp(query,knots,approximation)` to interpolate linearly to N points.

This is not standard inverse wavelet reconstruction using zeroed details and `waverec`, and it does not compensate A2 coefficient amplitude scale. Interpret amplitude and morphology cautiously; output is therefore rate-only. This named historical branch accepts only `wavelet=db4` and `level=2`.

Missing PyWavelets returns unsupported; no IMU is required. Confidence is fixed at 0.5. Not executed in finalcase.

## 3.18 Learned / Hybrid / ONNX Denoisers: Named but Without Runnable Weight Paths

Source: `artifacts/router.py:37–59, 105–106`, `UnsupportedReducer`.

`learned`, `learned_denoiser`, `hybrid_denoiser`, and `onnx_denoiser` all return unsupported with `x_ar=None` because no corresponding auditable model artifact exists. They have no network-layer definition, training procedure, or actual inference here; no imagined algorithm should be supplied for them.

Intuition: labels exist on the tool cabinet, but no usable tools are inside. Selecting a label does not pretend to clean a curve. Recording `parameters` does not mean a learning algorithm used them. Not executed in finalcase.

## 3.19 Stage5 Research Modules: Comparing Motion Detection and Denoising

This is a separate study workflow, not a hidden prerequisite of each frailty finalcase run. Main entry is `quality/stage5_pre.py:843–1039`. Default `include_denoiser=True` applies only when the corresponding study plan is explicitly run.

### 3.19.1 Six Sequential Steps

1. Internal Frailty29 five-fold motion OOF and all29 final model.
2. Cross-dataset PTT motion evaluation with fixed internal model/threshold.
3. PTT repeat-0 five-fold motion-training comparison and all-PTT final model.
4. Reverse Frailty29 evaluation with fixed PTT model/threshold.
5. Collect corresponding model bundles into a motion-model comparison.
6. Optional PTT denoiser benchmark; `--no-denoiser` skips it.

These steps reuse the algorithms above. Source evidence, schemas, and hashes record which data/parameters were used; they are not another signal transform. Resuming reuses completed stages rather than retraining in reporting.

### 3.19.2 Segment Generation for the PTT Denoiser Comparison

Source: `quality/stage5_pre.py:195–204, 610–678`, `run_ptt_denoiser_benchmark`.

Intuition: give every cleanup tool the same strips and evaluate with the same peak counter. No tool gets easier strips or a more favorable ruler.

Each person's sit calibrates IMU; each sit/walk/run recording produces filtered PPG and corresponding IMU. Study YAML supplies `segment_s`, without an implicit signature default. Starts advance by segment length. If the final complete regular-grid start does not align with the right edge, append a right-aligned segment, potentially overlapping its predecessor.

Skip segments shorter than 8 seconds or with fewer than three reference beats. Call each reducer once on two channels; preserve separate RED/IR failure rows when it fails. After success, evaluate wavelengths separately using the same configured detector and the lag correction/beat matching below. Ground-truth peaks do not select a best reducer for each segment.

### 3.19.3 Peak-Time Alignment: Fixed or Slowly Varying Lag

Source: `quality/stage5_pre.py:206–324`, `_matched_pairs`, `_best_lag_grid`, `_piecewise_shift_reference`, `align_and_score_beats`.

Intuition: recording clocks may differ, and electrical and optical changes arrive at different times. Shift reference flags to align with predicted flags before counting misses and extras. Long recordings may allow lag to vary slowly between segments.

Function defaults are `max_lag_s=10.0`, `lag_step_s=0.02`, `tolerance_s=0.2`, and `lag_window_s=None`. Actual studies may override them; defaults are not every experiment's settings.

Search equal-step candidate shifts from `−max_lag_s` to `+max_lag_s`. Interval accumulation first gives an upper bound on potential matches, then `match_events` counts exactly. Comparison key `(TP,−|lag|,lag)` prefers more TP, then smaller absolute lag, then more positive lag. Pruning only skips shifts unable to win and does not alter the objective.

If `lag_window_s` is nonnull, split references into fixed-duration blocks and search shifts independently, extending prediction candidates by maximum lag and tolerance on both sides. Finally match all shifted references together, preventing different blocks from claiming the same predicted peak.

`_matched_pairs` walks reference times in order and takes the nearest unused prediction within tolerance, using each prediction at most once. Interval-error calculation additionally requires successive matches to be consecutive in both reference and prediction indices; missed peaks cannot be bridged into apparently normal intervals.

### 3.19.4 Peak-Comparison Metrics and Equal-Participant Aggregation

Source: `quality/stage5_pre.py:296–324, 416–495`.

`Recall=TP/(TP+FN)`, `PPV=TP/(TP+FP)`, and `F1=2TP/(2TP+FP+FN)`. Zero denominators follow the called matching/aggregation implementation. `timing_mae_s` measures time differences of matched peaks.

For consecutive matches, use original reference IBI and predicted PPI, not shifted intervals that may jump across blocks: `e_i=PPI_i−IBI_i`; `RMSE_ms=1000sqrt(mean(e_i²))` and `MAE_ms=1000mean(|e_i|)`. No matchable intervals gives empty values.

First sum TP/FP/FN within each participant. Interval errors accumulate squared sums as `RMSE_segment²×matched_count` before taking the square root. Then average participant metrics arithmetically so longer recordings do not gain weight merely through more windows; standard deviation uses `ddof=1` and at least two people. Failed segments do not enter passed-metric means as invented F1=0. Pass/fail participant counts, segment counts, and coverage are reported separately, so always inspect coverage alongside performance.

### 3.19.5 Static Peak-Detector Ablation and Testing

Source: `quality/stage5_pre.py:497–609, 680–727`.

Static peak comparisons run configured detectors on complete PTT sit recordings; the plan determines the actual list compared with default `aboy_project`. This path neither processes dynamic segments nor selects denoisers. Detector mathematics appears in the peak chapter.

Original reports can provide recording-level medians, 25th/75th percentiles, IQR, and 10th/90th percentiles. `_static_peak_rank_sum_comparisons` first intersects participant/recording lists for two algorithms, then calls `scipy.stats.ranksums(...,alternative="two-sided")`. Despite identical lists, the test is unpaired rank-sum, not paired signed-rank; the code explicitly records `pairing_used_by_test=false`.

For multiple comparisons, `_holm_sidak_step_down` sorts ascending p values. At step k with m−k+1 remaining, candidate adjustment is `1−(1−p_k)^(m−k+1)`, taking the running maximum for monotonicity. Significance threshold is `1−(1−α)^[1/(m−k+1)]`. After the first nonrejection, no later hypothesis is rejected. Results retain both the full comparison family and prespecified within-plan families.

This explains existing behavior: sharing recording lists does not make the statistical test paired, and the explanation does not change the algorithm.

## 3.20 Manual-Review Coverage Checklist

| File/algorithm unit | Chapter section | Executed in finalcase |
|---|---|---|
| `signal/sqi.py` raw observations and all rate/morph/QC components | 3.2 | No |
| SQI fixed formulas, training empirical mapping, Q_rate/Q_shape/Q_morph, raw diagnostics | 3.3 | No |
| `quality/routing.py` modes and truth table | 3.1, 3.4 | All-off fast-path semantics only |
| `quality/routing_timeline.py` assessment windows, recovery, ownership intervals, matrix-row routing | 3.4 | No |
| `quality/window_selection.py` none, historical scores, top fraction, three scopes | 3.5 | none |
| `quality/motion.py` labels, five-fold protocol, training thresholds, historical options | 3.6 | No |
| `models/motion.py` 8/11/10-channel and development constructors | 3.6.2–3.6.4 | No |
| `quality/motion_adapters.py` materialization, sampling, training, inference, weight loading | 3.6 | No |
| `quality/motion_runner.py` OOF, deployment thresholds, internal/PTT bidirectional comparisons | 3.6.7–3.6.9 | No |
| `quality/motion_reference.py` same-person static calibration and source adaptation | 3.6.1, 3.6.9 | No |
| `quality/motion_bundle_adapter.py` matched-fold/all-person reuse and window decisions | 3.6.10 | No |
| `artifacts/base.py` shared I/O and six-axis/nine-channel references | 3.7 | No extra reducer computation |
| Identity, NLMS, SSA, spectral mask | 3.8–3.11 | Identity configured as comparison; others no |
| PCA, FastICA, NMF | 3.12–3.14 | No |
| EMD, CEEMD-lite + NLMS, DWT A2 | 3.15–3.17 | No |
| Learned/hybrid/ONNX unsupported registration | 3.18 | No; no runnable algorithms |
| `quality/stage5_pre.py` study sequence, denoiser benchmark, alignment, aggregation, tests | 3.19 | No |

Noncomputational `__init__.py` files, dataclass fields, serialization, hash bindings, and path/shape checks are not presented as extra numerical algorithms. Checks affecting algorithm inputs, failure behavior, or fitting scope are explained in the relevant sections. Called SciPy, NumPy, sklearn, and PyTorch internals are not copied into repository algorithms; calls, configuration, and mathematical meaning are stated explicitly, while actual values depend on installed dependencies and source.
