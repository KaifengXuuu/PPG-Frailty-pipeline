# 4. Features and representations

This chapter is a source-code companion for `final_pipeline_v6`. Unless stated
otherwise, source paths are relative to `src/ppg_frailty/`. Line numbers refer
to the source snapshot used when the chapter was written; function names are
more stable navigation points after code changes. `N` is the sample count,
`P` the detected peak count, `W` the window count, `D` the feature count,
and `K_i` the number of complete windows in recording `i`. Sampling rate
`fs` is in Hz; divide sample positions by `fs` to obtain seconds.
`median` is the middle ordered value, `IQR=Q75−Q25`, and
`MAD=median(|x−median(x)|)`; IQR and MAD are different quantities.

Read in sequence: locate pulses, measure their spacing, describe changes in
spacing, pulse shapes, and relationships between the two optical channels.
A parallel branch describes waveforms directly in each time window.
Finally, arrange these results into the arrays required by models.
“Intuition” explains shapes and actions; “Code and mathematics” introduces
formulas and technical terms.

## 4.1 Where these modules sit in finalcase

Source entry: fold preparation at `experiment.py:2648–2666`.
Configuration: `configs/presets/finalcase.yaml`.

| Module | finalcase selection | Part of the current classifier input? |
|---|---|---|
| Representation | `representation_mode: raw` | Yes: short eight-channel windows |
| Windows | `windows.raw_dl.length_s: 5`, `hop_s: 2.5`, at most 128 per recording | Yes |
| Within-window transform | `signal.normalization.raw_ppg: per_window_robust` | Yes: despite the PPG name, it applies to all eight channels |
| IMU post-transform | `signal.normalization.raw_imu: none` | No additional fold-local six-axis scaler is fitted |
| Model-input sampling rate | `signal.dl_resampling.target_fs_hz: 64` | Yes: resampling follows 400 Hz windowing and normalization |
| Peak-detector configuration | `msptdfast_v2_3_python_port` | Retained configuration; this does not add PPI/PRV to raw classifier input |
| File-feature groups | All seven groups | Retained configuration; the raw branch does not run `_extract_vector` |
| Quality/motion/denoising | `quality.mode: off`, `artifact.reducer: identity`, motion/denoiser disabled | Corresponding branches are not enabled |

`experiment.py:2656–2666` explicitly separates branches:
only `feature_vector` and `fusion` call `_extract_vector`;
`feature_matrix` calls `_extract_matrix_features`;
`raw` and `fusion` construct raw windows. A module listed in YAML is not
necessarily executed in every run. All alternatives below remain available,
but finalcase does not enable all of them simultaneously.

## 4.2 Shared peak-detection inputs, outputs, and wavelength selection

Intuition: mark the hilltops of two curves on the same time ruler. A questionable
section can be crossed out, but the paper must not be shortened so that distant
sections appear adjacent. Either curve can provide markers; choose the more
complete, trustworthy set as the main timeline.

Sources: `detect_pulses_per_wavelength` and `detect_pulses` in
`peaks/resolver.py:21–31,84–150,152–207`;
`select_reference_wavelength` in `peaks/pairing.py:121–133`.

Input is a finite-valued `[N,1]` or `[N,2]` waveform, or
`CanonicalSignalViews` containing a valid signal view. The pipeline adapter
uses the original 400 Hz time grid, with default minima of 8 seconds and
5 peaks. Output `PulseResult` contains `[P]` peak sample positions and
times in seconds, plus `[P−1]` intervals, their endpoint peak indexes,
validity, adjacency, and detection-run identifiers. An invalid interval is
not a numerical zero.

Code and mathematics:

1. `t_j = peak_j / fs`; `PPI_j=t_{j+1}−t_j`, in seconds.
2. `valid_interval_mask` indicates whether an interval is usable;
   `adjacency_mask` indicates whether it still connects originally adjacent
   events. They are not interchangeable.
3. `detector_coverage=sum(valid PPI)/(last_peak_time−first_peak_time)`.
   The denominator is not the complete recording duration.
4. The general `AUTO` ordering is
   `(detector_score, detector_coverage, is_RED)`; RED wins exact ties.
   The historical prominence single-result entry has its own candidate
   ordering; see 4.6.
5. For artifact-rate-only waveforms, the detector selects the longest continuous
   valid segment and retains its starting offset. It does not join across
   invalid gaps. Insufficient duration/peaks returns failure, without inventing peaks.

Configuration fields are `signal.peak_detector.detector_id`,
`min_observation_sec`, `min_peaks`, and `parameters`.
Only the MSPTD branch currently accepts detector-specific `parameters`.
The other three branches keep their numerical constants in source code;
these are not already exposed as separate CLI parameters.

## 4.3 MSPTDfast v2: from coarse hilltops to precise time positions

Intuition: shrink the curve and check whether each point is higher than points
at several distances on both sides. Tiny spikes usually look like hilltops
only at very short distances; larger rises remain prominent at multiple
viewing distances. After locating an approximate hilltop, return to the
original curve and find its nearby highest point.

Sources: defaults at `peaks/msptdfast_v2.py:29–34`;
`_msptd_window_peaks:70–95`; `detect_msptdfast_v2:97–142`;
`_prepare_input:144–191`;
`detect_pulses_per_wavelength_msptdfast_v2:193–282`.

### 4.3.1 Input and windows

The direct/identity path reads repaired `x_native`, not the 0.2–8 Hz-filtered
`x_filter`, because this detector removes a linear trend within its own
windows. The artifact-rate-only path reads `analysis_signal` and its valid
sample mask. Peak positions remain on the original 400 Hz grid.

Defaults are `target_downsample_hz=20`,
`minimum_heart_rate_bpm=30`, `window_s=6`, and
`overlap_fraction=0.2`.

Let `nominal=round(window_s*fs)`. Actual window length is `nominal+1`,
following the reference implementation's inclusive-right-endpoint convention.
Hop is `round(nominal*(1−overlap))`. A right-aligned final window covers the
tail. The internal decimation factor is
`q=max(1,floor(fs/target))`, so the actual coarse sampling rate is `fs/q`;
arbitrary target values are not guaranteed to be achieved exactly.

### 4.3.2 Comparisons inside each coarse-grid window

1. Line 78, `signal.detrend(..., type="linear")`, removes the least-squares
   line: `x'_i=x_i−(a i+b)`.
2. Lines 82–88 set `L=ceil(n/2)−1` and window duration
   `T=n/fs_coarse`; only distances `k` satisfying
   `(L/k)/T >= min_bpm/60` are retained.
3. Lines 90–93 construct the Boolean table
   `M[k,i]=(x'_i>x'_{i−k}) AND (x'_i>x'_{i+k})`.
   Positions without both neighbors are false. Flat plateaus also fail
   the strict-greater-than test.
4. Line 94 selects the row with the most hits:
   `λ=argmax_k sum_i M[k,i]`, not simply the largest allowed distance.
5. Line 95 accepts `i` as a coarse peak only if all rows 1 through `λ`
   are true.

### 4.3.3 Returning to the original grid

Lines 124–141 take `segment[::q]`. This is direct internal subsampling,
not the anti-alias resampling of the model input. A coarse peak `p` maps
initially to `(p+1)q−1`; this must not be casually changed to `p*q`.
Search for a local maximum in the original waveform around that position.
The radius is 0.2 seconds when the coarse rate is `<10`, 0.1 seconds when
`<20`, otherwise 0.05 seconds. The sample radius is
`ceil(fs*seconds)`. Results from all windows are sorted and deduplicated
using `np.unique`.

Final validity uses only `60/210 <= PPI <= 60/35`; peaks are retained.
Confidence values are all 1 and must not be interpreted as calibrated
probabilities. Detector score is `number_of_valid_intervals + 0.5*coverage`.
The default detects upward peaks only, with `selected_polarity=1`;
it does not also run the inverted curve.

A short window with fewer than 5 coarse samples has no coarse peaks.
Invalid complete input, insufficient duration, or too few final peaks fails.
The source identifies this as a Python port of the reference MATLAB equations,
not a promise of bitwise MATLAB equivalence.

## 4.4 Aboy project v1: choose one direction per recording, adapt every ten seconds

Intuition: inspect the curve upright and upside down. Every ten seconds,
find approximate hilltops, then adjust the minimum separation and required
prominence according to the stability of their spacing. Pass the measuring
rule from one section to the next. Finally retain one direction for the
entire recording.

Source: `peaks/aboy_project.py:27–32,72–196,198–369,420–517,519–560`,
especially `_block_parameters`, `_block_candidate`,
`_clean_intervals`, and `_polarity_candidate`.

Input is a private copy of valid `analysis_signal`. By default, only complete
ten-second blocks are used; shorter tails are dropped. Despite the shared
8-second minimum, this branch needs at least one complete ten-second block.
Each wavelength runs independently.

### 4.4.1 Adaptive filtering and preliminary peaks

For polarity `s∈{+1,−1}`, set `z=s*x`. Initial `HRI=0` is an internal
window-adjustment quantity, not heart rate in beats per minute.

1. `d210=max(1,round(fs*60/210))` sets the minimum preliminary peak distance.
2. Upper cutoff is `f_hi=min(8,max(1.5,3*(1+HRI_in)))`.
3. `_bandpass_block` applies second-order Butterworth SOS forward/backward
   filtering at 0.5–`f_hi` Hz to the current block. It does not write the
   result back into the original analysis view used for morphology.
4. `find_peaks(filtered,distance=d210)` returns preliminary peaks;
   differencing yields preliminary intervals.

### 4.4.2 Updating the rule and detecting peaks again

Lines 224–247 implement:

1. Retain only intervals strictly above the 30th percentile of preliminary
   intervals; call this set `R`.
2. If `|R|>=2` and
   `0.5*median(all preliminary PPI) <= mean(R) <= 1.5*median(all preliminary PPI)`,
   set `HRI_out=10*std_population(R)/mean(R)`; otherwise keep the old value.
3. `HRwin=fs/(3*(1+HRI_out))`; final distance is
   `max(round(2*HRwin),d210)`.
4. Sort preliminary-peak heights and average from index `floor(0.70*n)`
   to the end. If fewer than 3 peaks exist, use all of them.
5. Prominence threshold is
   `max(0.25*max(the above mean, std_population(filtered)),1e−12)`.
6. Call `find_peaks` again using both the distance and prominence threshold.

SciPy implements the detailed prominence-base search and equal-height-peak
handling. This repository sets parameters and consumes returned prominence;
the procedure cannot be reduced to “peaks above zero.”

### 4.4.3 Interval cleaning and final polarity

`_clean_intervals:139–196` has three stages:

1. Physiological range `[60/210,60/35]` seconds, including endpoints.
2. With at least 5 in-range intervals, compute their median `m` and
   `σ_r=1.4826*median(|PPI−m|)`. For `σ_r>0`, retain
   `|PPI−m|<=4σ_r`. With `σ_r=0`, do not add this rejection rule.
3. Compute median `r` from intervals surviving the first two stages and
   require `0.5r<=PPI<=1.8r`. Mark the later peak of an ineligible interval
   unaccepted and invalidate the intervals on both sides of that peak.
   Do not delete peaks or reconnect the two sides into a new interval.

For each polarity, concatenate ten-second blocks in their original time
order, removing only peaks at exactly duplicate positions. Overall score is
`S=n_clean+0.5*coverage−min(CV,2)`, where `CV` is valid PPI population
standard deviation divided by mean. With fewer than two valid intervals,
`CV=2`. Candidates are compared by score, valid interval count, total peak
count, then polarity value, so positive polarity wins the final tie.
Confidence is `clip(prominence/(2*median(prominence)),0,1)`,
not calibrated probability.

The selection ID is `aboy_project_v1`. The ten-second duration, frequencies,
and thresholds are source constants defining this version. It is the
project's Aboy-inspired implementation, not a claimed bitwise copy of the
original authors' implementation.

## 4.5 Aboy project v2: choose polarity per block, delete peaks before cleaning intervals

Intuition: instead of forcing the whole recording to face one way, choose the
clearer direction separately every ten seconds. Erase clearly inconsistent
hilltops first, then measure new distances between those remaining.
“Erase and remeasure” produces different distances from “keep the markers
but cross them out” in the previous section.

Source: `peaks/aboy_project_v2.py:58–155,170–260,263–373,376–410`,
especially `_prepare_input`, `_candidate`,
`_remove_ratio_outlier_peaks`, and `_result_for_channel`.

1. Direct/identity reads repaired native PPG; artifact-rate-only reads reducer
   output. Lines 110–124 first apply a second-order 0.2 Hz high-pass filter
   to the entire selected continuous segment, then divide it into complete
   ten-second blocks.
2. Within-block 0.5-to-adaptive-upper-cutoff filtering, preliminary peaks,
   and the `HRI_out` formula resemble v1. However, the update check uses
   `median(retained PPI)`, not v1's median of all preliminary intervals.
3. The highest 30% uses the top `ceil(0.3*n)` peaks. V1 uses all peaks
   when fewer than 3 exist, so small-sample behavior also differs.
4. Each block selects polarity by
   `(score,n_clean_ppi,peak_count,polarity)`; the next block inherits only
   the winning polarity's `HRI_out`. V1 instead runs each polarity through
   the complete recording before choosing.
5. Concatenate winning peaks across blocks. With at least 3 peaks, compute
   median PPI `r` **before physiological/MAD cleaning**. Physically delete
   each later endpoint peak of an interval `<0.5r` or `>1.8r`.
   This is one simultaneous deletion pass, not iteration to convergence.
6. Difference the remaining peak times again, then apply
   `[60/210,60/35]` range and `4*1.4826*MAD` checks.
   Retained peaks all have `accepted_peak_mask` set to true;
   interval validity is stored separately.
7. `selected_polarity` is the sign of the sum of winning block polarities,
   with positive ties. It summarizes the recording; it does not mean every
   block used that direction. Current downstream morphology reads this
   single direction, an important actual implementation detail.

Finally, `peak_ordinals` are reassigned after deletion and
`adjacency_mask` is entirely true. V1's “never delete original candidate
peaks” description does not apply to v2. Score and confidence formulas follow
v1, but valid PPI is defined as above. Defaults require a complete ten seconds
and 5 final peaks. ID `aboy_project_v2` accepts no additional detector parameters.

## 4.6 Historical dual-polarity prominence ablation

Intuition: find prominent hilltops with the whole curve upright and inverted.
Rather than changing the measuring rule over time, enforce a fixed minimum
distance and compare which result is more orderly and resembles continuous
pulsation.

Sources: `_robust_scale`, `_candidate`, and
`_detect_pulses_dual_polarity_ablation` in
`signal/peaks.py:35–72,75–225`.
ID: `dual_polarity_prominence_v1_ablation`.

Use `scale=(Q75−Q25)/1.349`, falling back to
`max(std_population,1e−12)`. For every polarity/wavelength candidate,
detect peaks using `distance=round(0.30*fs)` and
`prominence=max(0.15*scale,1e−12)`.

Score components:

- `plausible`: fraction of intervals in `[0.30,2.00]` seconds.
- `density=exp(−((peak_count/duration−1.4)/1.4)^2)`.
- `prominence_score=tanh(median(prominence)/scale)`, or 0 without peaks.
- `CV=std_population(PPI)/mean(PPI)`; with fewer than two peaks,
  set `plausible=0,CV=1`.
- `S=0.55*plausible+0.20*density+0.20*prominence_score+0.05*exp(−CV)`.

Whole-record candidates are compared by
`(score,peak_count,−channel_index,polarity)`; RED's smaller index gives
it preference. When `detect_pulses_per_wavelength` runs separately,
only the two polarities of that wavelength compete. Final interval validity
uses only `[0.30,2.00]` seconds, without Aboy MAD or reference-ratio
cleaning. A peak is accepted if it borders at least one valid interval.
Confidence is prominence-normalized as in v1.

## 4.7 The actual boundary of PPI-cleaning options

Intuition: the four hilltop-finding methods come with their own rules for
trustworthy distances. Changing the detector also changes cleaning.
The current code has no separate cleaner switch that can be freely mixed
with any detector.

| Detector | Range | Additional cleaning | Physically deletes peaks? |
|---|---|---|---|
| MSPTD | Seconds corresponding to 35–210 bpm | None | No |
| Aboy v1 | 35–210 bpm | MAD first, then ratio to the cleaned median | No; invalidates later endpoint peaks and adjacent intervals |
| Aboy v2 | 35–210 bpm | Delete peaks by ratio to the original interval median, then MAD | Yes; PPI is recomputed afterward |
| Prominence ablation | 0.30–2.00 seconds | None | No |

`features.prv_primary_backend` and `prv_library_comparison_scope`
also provide no new cleaner. Library comparisons receive exactly the same
fixed PPI vector; see 4.16.

## 4.8 Assembling route-eligible PPI while preserving real discontinuities

Intuition: differently colored time sections can share one long ruler,
but a color change must retain a break. Putting sections in one table does
not turn the gap between them into a normal heartbeat.

Sources: `_eligible_event_segment` and
`build_route_eligible_rate_pulse` in `features/window_matrix.py:114–263`.

For each direct and optional processed PPI, multiply its endpoint times
by 400 to recover sample positions. Both ends must belong to one routing
cell, with matching signal source and final quality excellent or acceptable.
The code uses `stop−1e−9` at the right endpoint for half-open intervals.

Split eligible intervals into segments with consecutive original ordinals,
the same cell, shared endpoints, and the same detection run; sort by
starting time. Within each segment retain original PPI, validity,
and adjacency. Between segments insert a connection row with
`PPI=NaN`, `valid=False`, `adjacency=False`, and
`run_id="routing_boundary"`. All segments must use the same wavelength;
a RED section is not joined to an IR section. No eligible segment means failure.

This assembly fits no whole-dataset statistics. It consumes upstream routing
decisions, which must not be treated as deterministic cache shared across folds.

## 4.9 Basic intervals and beats per minute

Intuition: describe the usual distance between neighboring hilltops and its
variation, then convert “how long to wait for each beat” into “how many beats
per minute.” Averaging waiting times before taking a reciprocal is not the
same as converting each one first and averaging afterward.

Source: `compute_prv` in `signal/prv.py:332–398`;
default configuration `PrvConfig:18–41`.

Set
`valid = valid_interval_mask & adjacency_mask & isfinite(PPI) & (PPI>0)`.
Call valid intervals `r_1…r_m`. Record count `m`, valid duration
`sum(r)`, and coverage `sum(r)/(last_peak−first_peak)`.
These three diagnostics can be available even when other PRV is ineligible.

Default basic eligibility requires at least 8 seconds of observation,
5 total peaks, and 4 valid intervals, with
`features.rate_prv_min_duration_s` and `rate_prv_min_peaks`.

| Output | Code formula | Unit |
|---|---|---|
| `ppi_mean_s` / `ppi_median_s` | `mean(r)` / `median(r)` | Seconds |
| `ppi_sd_s` | Sample SD `sqrt(sum((r−mean(r))²)/(m−1))` | Seconds |
| `ppi_iqr_s` / `ppi_mad_s` | `Q75−Q25` / `median(abs(r−median(r)))` | Seconds |
| `ppi_cv` | `std_sample(r)/mean(r)` | Dimensionless |
| `hr_mean_bpm` / `hr_median_bpm` | Compute `h_i=60/r_i` first, then mean/median | Beats/minute |
| `hr_sd_bpm` | `std_sample(h)` | Beats/minute |

Ineligible fields remain NaN/false. Zero is not used to stand for missing variability.

## 4.10 Time-domain changes: SDNN, RMSSD, SDSD, NN50, and pNN50

Intuition: distances may be widely spread overall, or may simply grow slowly.
To identify abrupt changes between consecutive beats, compare only genuinely
adjacent intervals. Never subtract across a break.

Source: `signal/prv.py:400–432`. Defaults:
`features.time_prv_min_duration_s=60`,
`time_prv_min_coverage=0.8`, and `time_prv_min_intervals=30`.

1. Compute `sdnn_s=std_sample(r)` only when all three eligibility criteria hold.
2. Pairs require both intervals valid, both adjacency flags true, and
   `stop_index[i]==start_index[i+1]`.
3. Compute `d_i=PPI_{i+1}−PPI_i` only for this pair set.
4. `RMSSD=sqrt(mean(d²))`; `SDSD=std_sample(d)`.
   With only one difference, SDSD is 0.
5. `NN50=count(abs(d)>0.050)`, strictly greater than 50 ms.
   `pNN50=NN50/len(d)` is a 0–1 fraction, not a percentage.

Without eligible adjacent differences, SDNN can remain available while the
difference-based metrics are missing with a recorded reason.
Standard deviations here use `ddof=1`, unlike the short-window matrix's
population standard deviations below.

## 4.11 Poincaré widths in two directions

Intuition: plot “this interval” against “the next interval.”
Spread along the lower-left to upper-right direction means both intervals
lengthen or shorten together. Perpendicular spread means alternating
long-short behavior is more pronounced.

Source: `signal/prv.py:417–427`;
group `features.enabled_groups: hrv_nonlinear`.

Using the same eligible differences as above:
`SD1=sqrt(0.5)*SDSD`;
`SD2=sqrt(max(2*var_sample(r)−0.5*SDSD²,0))`;
then `SD1/SD2`. Both widths are in seconds; the ratio is dimensionless
and missing when its denominator is zero. The implementation calculates
directly from interval/difference variances; it does not separately fit an ellipse.

## 4.12 Sample entropy: do similar short sequences stay similar when extended?

Intuition: find two short sequences whose first two steps look alike.
Look one step farther in each and ask whether they still match.
Frequent divergence makes local behavior harder to predict.
Do not cross a broken interval to construct a sequence.

Sources: `_sample_entropy` in `signal/prv.py:272–308` and eligibility
at lines `434–458`. Defaults:
`features.sample_entropy: {m: 2, r_sd_fraction: 0.2, min_intervals: 200}`.

1. Among continuous `valid & adjacency` segments, choose the one with the
   greatest cumulative PPI duration, not simply the most elements.
   Use only this segment `x`.
2. Require at least 200 consecutive intervals by default.
   Tolerance is `r=0.2*std_sample(x)`; nonpositive or nonfinite tolerance
   means missing.
3. Length-`m` templates are `u_i=(x_i,…,x_{i+m−1})`.
   Count a match when `max_j|u_i[j]−u_k[j]|<=r`.
   Exclude self-matches and count `(i,k)`/`(k,i)` only once.
4. Count matches `C_m,C_{m+1}` for lengths `m` and `m+1`,
   with corresponding comparable-pair counts `A_m=n_m(n_m−1)/2`.
5. Output `−log((C_{m+1}/A_{m+1})/(C_m/A_m))`.
   This must be a **ratio of match probabilities**, not a ratio of raw
   match counts at the two lengths.

If either match probability is zero, return NaN, not infinity.
The nested template-comparison loops can approach quadratic cost for
long sequences; this is not classifier training.

## 4.13 Spectral PRV: slow and fast variation on the real timeline

Intuition: put each measured interval back at the time it actually ended,
then draw a curve on an evenly spaced time grid. Examine whether slow or
faster variation contributes more. Do not cut out long gaps; doing so
changes both the apparent slow and fast behavior.

Source: `signal/prv.py:311–317,460–518`.

Default eligibility: role family B or R; direct, identity, or
artifact-rate-only route; `q_rate_qualified=True`; total observation
at least 300 seconds; the longest continuous valid PPI segment covers
at least 300 cumulative seconds and contains at least 200 intervals;
overall coverage at least 0.8.

Configuration:
`features.spectral_prv_min_duration_s=300`,
`spectral_prv_min_intervals=200`, `spectral_prv_min_coverage=0.8`,
`tachogram_fs_hz=4`, with three bands in `spectral_bands_hz`.

1. Interval timestamps are their right-end peak times, not cumulative
   sums of the filtered valid PPI.
2. The regular grid starts at the first valid right-end time and stops
   before the last, with spacing `1/4` second.
3. Linearly interpolate PPI, then remove a least-squares linear trend.
4. With at least 256 grid points and strictly increasing times, call Welch
   using `nperseg=min(1024,N_grid)` and `detrend=False`.
   Window/overlap arguments not explicitly supplied follow the installed
   SciPy defaults.
5. Integrate VLF `[0.003,0.04]`, LF `[0.04,0.15]`, and HF
   `[0.15,0.40]` Hz by the trapezoidal rule over in-band bins.
   Both endpoints are included, retaining shared boundary bins as coded.
   A band with fewer than two bins yields NaN.
6. Power units are seconds². `LF/HF`, `LF/(LF+HF)`, and
   `HF/(LF+HF)` are dimensionless ratios, missing for nonpositive denominators.

Passing eligibility does not guarantee enough integration bins in every
band. W/S roles are excluded from formal spectral PRV in this implementation,
even for long recordings.

## 4.14 Pulse morphology: one rise between two valleys

Intuition: find a valley on each side of a hilltop and stretch a straight
string between them. The height above the string, climbing and descending
times, width halfway up, and area above the string describe the curve's shape.

Sources: `extract_morphology` and `_crossing_time` in
`signal/morphology.py:13–21,33–61,63–166`.

Input is `[N]` or `[N,2]` `x_filter` and the corresponding wavelength's
`PulseResult`, fixed at 400 Hz. Only direct/identity routes are allowed.
A nonidentity artifact-reduced curve does not preserve the required shape
meaning and cannot supply these metrics.

Code and mathematics:

1. Line 93 multiplies the waveform by detector polarity `s`, measuring
   upward rises.
2. Consider only the second through penultimate peaks, requiring the central
   peak to be accepted. Left/right search boundaries are integer midpoints
   between the central peak and its neighbors.
3. Find minimum positions `l,r` from the left boundary to the peak and
   from the peak to the right boundary; require `l<p<r`.
4. The valley line is
   `b(n)=x_l+(x_r−x_l)*(n−l)/(r−l)`;
   `y(n)=s*x(n)−b(n)`, with valley values also taken from the polarity-adjusted curve.
5. Amplitude is `A=y(p)>0`. Half-height crossings use linear interpolation
   between neighboring samples: the last crossing before the peak and
   first crossing after it.

| Independent morphology quantity | Formula | Unit |
|---|---|---|
| Height `amplitude` | `A` | Original PPG counts |
| Half-height width `width_half_s` | `(right_cross−left_cross)/fs` | Seconds |
| Rise time `rise_s` | `(p−l)/fs` | Seconds |
| Decay time `decay_s` | `(r−p)/fs` | Seconds |
| Rise slope `rise_slope_per_s` | `A/rise` | PPG units/second |
| Decay slope `decay_slope_per_s` | `−A/decay` | PPG units/second; negative sign retained |
| Positive area `positive_area` | Trapezoidal integral of `max(y,0)` with step `1/fs` | PPG units·seconds |

Each quantity has independent beatwise validity. Whole-record median and MAD
give 14 summary fields, each valid only with at least 3 eligible beats.
Missing half-height crossings affect width only; they do not automatically
invalidate measurable rise/decay times or other quantities.

## 4.15 Two optical paths: pairing, local heights/baselines, and waveform similarity

### 4.15.1 Peak pairing

Intuition: choose the more reliable marker timeline. Around each hilltop,
draw vertical boundaries halfway to neighboring hilltops, creating
nonoverlapping cells. Pair it with the closest hilltop from the other curve
inside that cell. Each hilltop can be used only once.

Source: `pair_dual_wavelength_beats` in `peaks/pairing.py:148–315`.

Both wavelengths must share detector and route. Reference wavelength follows
the score/coverage/RED ordering in 4.2. For accepted reference peak `p_j`,
use neighbors in the **accepted reference peak sequence**, forming
`[(p_{j−1}+p_j)/2,(p_j+p_{j+1})/2)`. First and last reference peaks
lack complete cells and are not paired.

Select an accepted, unused peak of the other wavelength inside the cell.
Break ties by absolute time distance, earlier position, then smaller original
ordinal. No additional fixed millisecond tolerance is imposed. Unmatched
and rejected peaks also generate audit rows. Pair lag is always
`IR_position−RED_position`, divided by `fs` for seconds;
positive means IR is later.

The auxiliary `match_events:318–342` supports diagnostics with reference
events: in reference-time order, select the nearest unused predicted event
within tolerance, producing TP/FP/FN, precision, recall, F1, and mean absolute
matched timing error. This is not the formal dual-wavelength pairing algorithm.

### 4.15.2 AC, DC, and PI

Intuition: even when two curves represent the same pulse, each needs its own
hilltop and valleys. Height describes how much this pulse rises; baseline
describes the brightness level beneath it. Dividing height by baseline
magnitude gives a relative rise less dependent on overall brightness.

Sources: `_wavelength_local_ac_dc` and `extract_dual_optical` in
`signal/optical.py:128–189,191–331`.

Inputs are native and filtered `[N,2]` arrays with independent peak results
for both wavelengths; only direct/identity at 400 Hz is allowed.
For each pair, repeat valley searches within each wavelength.
AC is the polarity-adjusted filtered peak minus the valley line.
DC is the native-waveform line through the same valley positions,
evaluated at the peak. **DC is not inverted with peak polarity.**

1. Only pairs with positive AC and finite DC for both wavelengths enter
   the common set.
2. Per-pair values are `PI_R=AC_R/(abs(DC_R)+ε)` and
   `PI_I=AC_I/(abs(DC_I)+ε)`, with `ε=1e−12`.
3. AC ratio is `AC_R/(AC_I+ε)`; DC ratio is
   `abs(DC_R)/(abs(DC_I)+ε)`; ratio-of-ratios is `PI_R/PI_I`.
   Denominator magnitude no greater than ε yields missing.
4. Formal file-level predictor fields first take
   `median(AC_R)`, `median(AC_I)`, `median(DC_R)`, and
   `median(DC_I)` over the common set, **then calculate ratios from
   those four numbers**. Despite `_median` in their names, these are
   not medians of beatwise ratios.
5. Four base medians plus five derived ratios give 9 fields.
   At least 3 jointly valid pairs are required. Beatwise ratios are
   diagnostic only and do not enter formal prediction.

This is not an oxygen-saturation formula; the code does not map
ratio-of-ratios to SpO₂.

### 4.15.3 Zero-lag and shifted similarity

Intuition: bring both curves to comparable centers and sizes, then see
whether they rise and fall together. Also shift one curve a little left
or right to find the closest match; the shift direction indicates which
curve comes later.

Source: `signal/optical.py:66–117,333–345`.

Standardize each complete filtered waveform once:
`z=(x−mean(x))/std_population(x)`.
Zero-lag correlation is `dot(z_R,z_I)/N`.
For every integer-sample lag in `[-0.5,+0.5]` seconds, compare
overlapping parts with
`ρ(lag)=dot(a,b)/(norm(a)*norm(b))`.
Vector norms are recomputed for each overlap, but **overlap means are
not subtracted again**.

Select the largest signed correlation, not the largest absolute value.
For ties, prefer smaller absolute lag, then smaller signed lag.
Output zero-lag correlation, maximum correlation, and its lag in seconds:
3 fields. Positive lag means IR is later. A constant curve that cannot
be standardized makes all three missing. No coherence is calculated.

## 4.16 External PRV backend comparisons: not classifier replacement modules

Intuition: give exactly the same sequence of distances to different
calculators and compare their answers. Changing the calculator and the
input numbers simultaneously makes the source of differences unclear.

Source: `features/prv_backend_compare.py:47–107,115–204,206–249`.
Options are `local`, `aura_hrv_analysis`, and `rhenan_hrv`.
These are fixed-PPI diagnostics only, without cleaning or classifier integration.

Input is milliseconds, with at least 4 finite positive values.
Local calculations include mean, median, range, sample SDNN/SDSD,
RMSSD, `abs(diff)>50` count, percentage
`100*mean(abs(diff)>50)`, `CVNN=SDNN/mean(PPI)`,
`CVSD=RMSSD/mean(PPI)`, and the mean, minimum, maximum, and sample
standard deviation of `60000/PPI_ms`.
Here `pnni_50` is a percentage, whereas `pnn50` in 4.10 is a fraction.

Aura calls `hrvanalysis` time-domain, geometrical, frequency-domain,
CSI/CVI, Poincaré, and SampEn functions in sequence.
Rhenan constructs `hrv.rri.RRi`, then calls
`hrv.classical.time_domain/frequency_domain/non_linear`.
Their internal equations are not defined in this repository's adapters;
this chapter does not claim line-by-line review of their internals.
Missing dependencies return `unavailable_optional_dependency`;
other exceptions return `backend_failed`.
No automatic fallback to another library occurs.

Fixed test vectors include constant 800 ms, alternating 760/840 ms,
two superimposed periodic components, a slow 740→880 ms change,
and one 1400 ms outlier. Each has 512 intervals.
Input hashes confirm all backends receive the same sequence.

## 4.17 Engineering-feature inputs: one description per complete time window

Intuition: without finding hilltops first, write a small card describing
a curve segment: its usual height, spread, sharp excursions, and slow/fast
variation. Use the same rulers for both optical channels and body-motion curves.

Sources: `engineering_feature_names`, `_imu_columns`, and
`extract_engineering_features` in
`features/engineering.py:20–41,199–275,278–356`.

Inputs are 400 Hz amplitude-preserving PPG and SI-unit IMU.
The `windows.engineering` plan defaults to 10-second windows/2-second hops.
Output is `[W,115]`, with same-shape per-value validity,
`[W]` start positions, and row flags. Only complete, unpadded windows
are accepted. Length/hop are configurable; the schema name's
`10s_hop2s` does not mean duration is immutable.

| Channel group | Fields per channel | Channels | Subtotal |
|---|---|---:|---:|
| `ppg_red`, `ppg_ir` | 7 time statistics + 4 spectral summaries + 3 band powers | 2 | 28 |
| Dynamic acceleration magnitude, angular-velocity magnitude, jerk magnitude | 7 + 4 + 4 band powers | 3 | 45 |
| Dynamic acceleration x/y/z, angular velocity x/y/z | 7 time statistics | 6 | 42 |
| Total | Fixed order | 11 | 115 |

Acceleration is m/s², angular velocity rad/s, and jerk m/s³.
Magnitudes are the square root of the sum of squared axes.
The engineering branch does not use normalized eight-channel DL-window copies.
For a nonidentity rate-only route, the 28 PPG fields are NaN/false,
while IMU fields can still be calculated. Entry into the 146-channel matrix
has stricter row rules; see 4.23.

## 4.18 Engineering time statistics: seven different rulers

Intuition: one ruler measures average height, another spread around the
average, and another overall size including negative excursions.
Two rulers resist extreme spikes. The last two describe asymmetry and
the tendency to produce far-out excursions.

Source: `_one_channel_features` in `features/engineering.py:124–181`.

First check the fraction of finite samples. Below 80%, all fields for
the channel are missing. Time statistics use all finite values `x_1…x_n`,
which need not be contiguous. Spectral calculations separately use the
longest continuous segment.

| Small algorithm | Lines | Formula and meaning |
|---|---|---|
| mean | 175 | `μ=sum(x)/n`; retains the waveform baseline |
| population_sd | 176 | `sqrt(sum((x−μ)²)/n)`, `ddof=0` |
| rms | 177 | `sqrt(sum(x²)/n)`; no prior mean subtraction |
| iqr | 178 | `Q75−Q25`; width of the middle half |
| mad | 179 | `median(abs(x−median(x)))`; no factor 1.4826 |
| skew_bias_corrected | 172,180 | `scipy.stats.skew(x,bias=False)`; sample-size-corrected standardized third central moment |
| pearson_kurtosis | 173,181 | `scipy.stats.kurtosis(x,fisher=False,bias=False)`; Pearson definition, with normal reference 3 rather than 0 |

The first five use the source channel's units; the last two are dimensionless.
Let `m_k=mean((x−μ)^k)`. With sufficient nondegenerate samples,
skewness is `sqrt(n(n−1))/(n−2)*m_3/m_2^(3/2)`;
Pearson kurtosis is
`3+((n²−1)*m_4/m_2²−3(n−1)²)/((n−2)(n−3))`.
The SciPy calls determine exact small-sample/near-constant behavior.
Nonfinite returned values are marked missing. The code suppresses expected
precision warnings but does not replace returned values.

## 4.19 Engineering spectra: the shared Welch basis

Intuition: split a short curve into pieces, gently narrow each piece
toward its ends, and ask how waves of different speeds combine to form it.
Average the pieces' answers to reduce the influence of isolated sharp events.

Sources: `engineering_welch_parameters` and `_one_channel_features`
in `features/engineering.py:86–103,141–161`.

1. If gaps exist, use only the longest continuous finite run.
   Do not join both sides of a gap before spectral analysis.
2. Segment length is
   `L=min(N_run,max(64,min(2048,round(4*fs))))`.
   At the default 400 Hz, complete windows generally use 1600 samples,
   or 4 seconds.
3. Adjacent segments overlap by `L//2`. Window is explicitly `hann`,
   and the spectrum is one-sided. `detrend` is not supplied explicitly,
   so the SciPy Welch default applies.
4. Abstractly, for segment `b`, transform the weighted samples `y_b[n]`:
   `Y_b[k]=sum_n w[n]*y_b[n]*exp(−j2πkn/L)`.
   Normalize squared magnitude by sampling rate and window energy,
   then average. SciPy handles folding symmetric negative/positive
   frequencies into the one-sided spectrum.

Output frequency `f` is in Hz and density `P(f)` in source-units²/Hz.
The five spectral algorithms below reuse this result rather than
re-filtering/transforming for every field.

## 4.20 Engineering spectral summaries and band powers

### 4.20.1 Total power

Intuition: add contributions from all speeds of variation to describe
overall signal strength in the segment.

Source: `features/engineering.py:183–190`.
`trapezoid(power,frequencies)` integrates the area below the spectrum
using frequency spacing. Units are source-units².
It is not a direct `sum` of PSD values.

### 4.20.2 Normalized spectral entropy

Intuition: contribution concentrated at one speed gives a focused answer;
similar contributions from many speeds give a dispersed answer.

Source: `features/engineering.py:106–114,191`.
Set `p_k=P_k/sum(P)`, then for positive `p_k` calculate
`H=−sum(p_k log(p_k))/log(number_of_bins)`.
Nonpositive total power or fewer than 2 bins gives NaN.
This normalizes the power array, not trapezoidal area assigned to each bin.

### 4.20.3 Dominant frequency

Intuition: find the speed of variation with the greatest contribution.

Source: `features/engineering.py:185–186,192`.
Return `f[argmax(P)]`, requiring a finite spectrum and `sum(P)>0`.
Units are Hz; no extra interpolation improves peak-frequency resolution.

### 4.20.4 Spectral centroid

Intuition: place weights at different speeds on a ruler and find its
balance point.

Source: `features/engineering.py:187–188,193`.
`centroid=sum(f*P)/(sum(P)+float64_machine_epsilon)`, in Hz.
Unlike the highest peak, it depends on all frequency bins.

### 4.20.5 Band powers

Intuition: collect the contributions from very slow, slower, faster,
and very fast variation into separate bags.

Source: `features/engineering.py:35–36,117–121,195`.
PPG bands are `[0.2,0.5]`, `[0.5,3]`, and `[3,8]` Hz.
The three IMU-magnitude channels use `[0.1,0.5]`, `[0.5,3]`,
`[3,8]`, and `[8,20]` Hz. Each includes both endpoints
and requires at least two bins for trapezoidal integration.
Units are source-units². The six individual-axis channels do not receive
these band powers or spectral summaries.

## 4.21 File-feature vectors: composition of 282 fields

Intuition: a recording may have many window cards and pulse-derived cards.
Reserve fixed positions for each type, then fill trustworthy values.
Keep absent measurements blank instead of pretending they were measured
as exactly zero.

Sources: `features/registry.py:43–78,128–151,278–405,407–434,436–504`;
`representations/feature_vector.py:8–34`.

| Independently selectable complete group | Fields | Contents |
|---|---:|---|
| `ppi_basic_rate` | 11 | Valid interval count/duration, 6 PPI descriptions, 3 HR descriptions |
| `hrv_time_domain` | 5 | SDNN, RMSSD, SDSD, NN50, pNN50 |
| `hrv_spectral` | 6 | VLF/LF/HF, LF/HF, two normalized powers |
| `hrv_nonlinear` | 4 | SD1, SD2, their ratio, sample entropy |
| `morphology` | 14 | Median/MAD of each of 7 morphology quantities |
| `dual_optical` | 12 | 9 AC/DC/ratio fields + 3 waveform-agreement fields |
| `engineering_summary` | 230 | Across-window mean/population SD for each of 115 fields |
| All | 282 | File-level `[282]` values and `[282]` validity |

For each column, `summarize_engineering` takes all valid finite window
values `v_w` and computes `mean(v)` and `std_population(v)`.
One window can form a valid summary with SD=0.
No values means NaN/false.

`features.enabled_groups` enables all seven groups by default.
Inputs can be unordered or use aliases, but output follows fixed registry
order. Selection operates on whole groups, not arbitrary unordered individual
fields. Older `PPI`, `HRV`, and `morphology_ppi_hrv` names are
compatibility mappings to these groups.

The registry excludes technical/quality metadata such as `prv.coverage`,
SQI, motion probabilities, and routing. `build_feature_vector` also uses
route and validity to decide whether a position can be filled.
In `experiment._extract_vector:1329–1480`, acceptable quality supplies
only pulse/PRV, leaving nonpulse fields missing. Under mixed routing,
optical features require every recording cell to be excellent/direct.
Missing information remains separate from measured values.

## 4.22 Feature scaling and imputation: similar names, different formulas

Intuition: put card values onto comparable rulers. Use training participants
to decide the usual center and the size of one ruler unit, then reuse that
ruler for new participants. Do not inspect the new participant first to
decide the ruler's scale.

### 4.22.1 Transforming 115-column engineering sequences

Sources: `fit_fold_feature_transform` and `transform_engineering`
in `features/engineering.py:359–414`.

Stack only declared outer-training extractions rowwise.
For valid finite values in each column,
`center=median` and `scale=max(IQR,1e−8)`.
An entirely missing training column keeps center=0 and scale=1.
Output `(value−center)/scale`; invalid values remain NaN.
The function relies on the caller supplying actual training extractions;
it does not independently filter each row by participant ID from
the `extractions` objects.

### 4.22.2 File-vector transforms for fusion

Sources: `fit_fold_feature_vector_transform` and
`transform_feature_vector` in `features/vector_transform.py:133–241`.

Input vectors include participant IDs. First actually select recordings
from the fitting roster. Per column, use `center=median`, `scale=IQR`.
If `IQR<=1e−12`, try `1.4826*MAD`; if still no greater than the
threshold, use scale=1. Entirely missing columns use center=0, scale=1,
and record valid sample counts. Valid entries become `(x−center)/scale`;
invalid entries remain NaN/false.

### 4.22.3 Numerical encoding of fusion vectors

Source: `transform_feature_vector_batch` in
`features/vector_transform.py:243–272`.

Apply the training-fold transform above, set invalid values to neutral zero,
then append each column's validity as 0/1:
`[z_1,…,z_D,m_1,…,m_D]`. With all seven groups, width is 564.
Missingness indicators explicitly become additional fusion inputs.
By contrast, the 146-channel matrix below does not concatenate validity
as predictor channels. These representations are not interchangeable.

### 4.22.4 Imputation and scaling for classical feature-vector classifiers

Intuition: when a whole-record card lacks a measurement, fill it with
the usual value learned from training cards. Classifiers that need
comparable scales then recenter and resize the values. Others use the
filled card directly. Test cards cannot choose imputation values.

Sources: `FeatureVectorBaseline._make_pipeline` and `fit` in
`models/feature_baselines.py:134–178,188–209`.
This is the actual representation-to-classical-classifier boundary,
**not universal application of the IQR transform in 4.22.2**.

`logistic_regression` and `rbf_svm` use
`SimpleImputer(strategy="median",keep_empty_features=True)` followed by
`StandardScaler()`. Training-column medians fill NaN, then the imputed
training columns' means and standard deviations define
`(x−μ_train)/σ_train`. Entirely missing columns retain their width;
scikit-learn implements the exact degenerate imputation behavior.
`extra_trees` uses the same median imputation without `StandardScaler`.
All steps are fitted together on supplied training rows; OOF only
transforms/predicts. The function receives training data from its caller;
train/test splitting occurs upstream.

## 4.23 The 146-channel window matrix: detailed descriptions ordered in time

Intuition: instead of compressing a recording into one summary card,
arrange each window's small card in chronological order along a strip.
This retains both content and change over time. Longer recordings
naturally have more cells.

Source: `features/window_matrix.py:33–76,282–383,386–486`,
especially `extract_window_features`.

Each row has 115 engineering fields, 14 local morphology fields,
and 17 local interval fields: 146 total.
Engineering windows come from the same complete-window plan.
The 282-field file vector is not repeated in every window,
and file context is not appended.

### 4.23.1 Assigning events to windows

Assign PPI by whether the **time midpoint** of its two endpoint peaks
falls in `[window_start,window_stop)`. Both endpoints must also lie
within one eligible routing cell. Adjacent differences require both
original intervals in the current selection, the same detection run,
a shared endpoint, and true adjacency; the complete span across three
peaks must not cross a routing boundary. Morphology is assigned by its
central peak time, requiring the complete left/right midpoint boundaries
to stay inside one excellent/direct cell.

### 4.23.2 Nine local interval fields

With at least 4 valid intervals, calculate PPI mean, median,
population SD, IQR, MAD, and CV, plus mean, median, and population SD
of `60/PPI`. Units match 4.9, but SD uses `ddof=0`.
These are short-window descriptions, not claims of long-duration PRV eligibility.

### 4.23.3 Six local adjacent-change fields

With at least 3 eligible differences `d`, calculate mean(d),
median(d), population SD(d), MAD(d), mean(abs(d)), and median(abs(d)).
All are in seconds. Rejected PPI is not deleted to create new adjacency.

### 4.23.4 Two local PRV descriptions

On the same set of at least 3 differences, calculate `mean(d²)`
and `mean(abs(d)>0.050)`. The first has units seconds² and
**has no square root, so it is not RMSSD**; the second is a fraction.
Names are `local_prv.mean_squared_delta_ppi_s2` and
`local_prv.pnn50_fraction`.

### 4.23.5 Fourteen local morphology fields and row-level routing

For each morphology quantity, take median/MAD over at least 3 eligible
beats within the window. Row logic in `extract_window_features:452–471`:

- Excellent: fill 115 engineering + 14 morphology fields, and extract
  17 rate fields from direct pulses.
- Acceptable: fill only the 17 rate fields using direct/processed pulses
  eligible for their cells. The first 129 fields remain missing;
  **this does not merely mask PPG while retaining IMU engineering fields**.
- Ineligible row: row flag false and all values unavailable,
  while retaining the original time position.

## 4.24 Training-fold scaling and storage of the 146-channel matrix

Intuition: choose a training-data ruler for each card field and place
empty fields at the ordinary center. A separate marker tells the model
to disregard an entirely ineligible card. The number of cards follows
recording duration instead of being forced to one storage length.

Sources: `fit_fold_window_feature_transform`,
`transform_window_features`, and `build_ordered_window_matrix`
in `features/window_matrix.py:489–606`;
`representations/feature_matrix.py:9–43`.

Fit only valid fields in valid rows of training extractions, using
`center=median`, `scale=IQR/1.349`.
If scale is nonfinite or `<=1e−8`, fall back to
`std_population`, then 1 if still invalid.
This differs from both engineering-sequence and file-vector formulas
in 4.22.

After `(x−center)/scale`, invalid fields and rows are set to 0.
The transformed `WindowFeatureExtraction` still carries per-value
validity (line 556), but it is not added as predictor channels.
Final matrix construction transposes `[K_i,146]` to
`[146,K_i]` and carries a separate `[K_i]` row mask.
Lines 588–591 save only a packed-bit summary of per-value validity
in provenance; the final matrix object does not carry a complete
per-field mask. At least one usable row is required.
Recording storage neither crops nor pads; padding may occur only
when assembling batches. Recording length `K_i` is variable;
the old fixed `matrix_k` cap is no longer a current parameter.

## 4.25 Raw eight-channel representation: waveform copies, not handcrafted features

Intuition: instead of writing summary cards, stack eight short curves
neatly for the model. Each segment moves its own center near zero
and scales by its own size. The original recording stays unchanged,
so other modules can still measure actual pulse heights and movement size.

Sources: `_normalize_dl_window` and `build_raw_windows` in
`representations/raw.py:59–112,115–244`;
`normalization.py:15–25,74–89`.

Inputs are amplitude-preserving PPG `views.x_analysis` and physical
`dynamic_acc_mps2`/`gyro_rads`, concatenated into `[N,8]`
in order RED, IR, A_dyn_x/y/z, GX/GY/GZ.
Nonidentity `x_ar` is rate-only and does not automatically replace
raw inputs. Each planned window is processed, transposed, and converted
to float32, producing `[W,8,T_400]` and a `[W,T_400]` mask.

### 4.25.1 No scaling: `raw_ppg: none`

Copy all eight channels directly, skipping centering, scaling, and clipping.
This is not limited to disabling PPG scaling. Data still undergoes upstream
gap repair, filtering, and IMU processing.

### 4.25.2 Ordinary per-window scaling: `per_window_standard_zscore`

For each channel, `center=mean(x)` and
`scale=std(x,ddof=standard_ddof)`, default ddof=0.
If scale is nonfinite or no greater than `scale_epsilon=1e−8`,
use 1. Output `(x−center)/scale`.

### 4.25.3 Robust per-window scaling: `per_window_robust`

For each channel, `center=median(x)` and
`scale=(Q75−Q25)/robust_iqr_divisor`, with divisor 1.349 by default.
For a tiny/nonfinite scale, `iqr_fallback` selects:

- `standard_deviation_then_finite_one`: ordinary SD, default ddof=0;
  then 1 if still unusable.
- `median_absolute_deviation_then_finite_one`:
  `MAD/mad_consistency_divisor`, default divisor
  0.6744897501960817; then 1 if still unusable.
- `finite_one`: immediately use 1.

Both scaling strategies finally set nonfinite outputs to 0 and can clip
using `clip_after_scale`, default `[-8,8]`; `null` disables clipping.
Windows containing invalid original values or IMU initialization samples
without estimates are already dropped before this step. Training does
not rely on this zero fallback to retain such windows.

### 4.25.4 Complete windows, padding, and sampling rate

By default, only complete windows are used. If short-record/tail padding
is explicitly enabled, compute centers/scales only on the real prefix.
The padded suffix is 0 with mask=false and does not contribute to
statistics. No eligible window means failure.

`signal.normalization.raw_ppg` is a historical compatibility name;
the current implementation scales **all eight channels** per window.
`raw_imu:none` only disables the subsequent six-axis fold transform;
it does not mean IMU was never scaled.

`experiment.py:1950–1994` resamples model inputs after raw/fusion
datasets have been constructed. Finalcase's 5-second windows initially
contain 2000 samples, then after normalization are converted to the
64 Hz model grid, approximately 320 samples. MSPTD's internal 20 Hz,
400 Hz feature processing, and the model's 64 Hz are three separate rates.

## 4.26 Optional fold-local six-axis post-transform for raw/fusion

Intuition: after scaling each small picture individually, a ruler chosen
jointly from training participants can additionally standardize the six
body-motion curves. This is a second transform; it does not restore
physical magnitude already removed earlier.

Sources: `fit_fold_imu_channel_transform` and
`apply_fold_imu_channel_transform` in
`representations/imu_transform.py:151–240,243–290`.

Inputs are `[W,8,T]`, per-window participant IDs, and valid-sample masks.
Select only real samples from fitting-roster windows and process
channels 2–7 separately:

- `raw_imu:none`: center=0, scale=1; values ultimately remain unchanged.
- `raw_imu:outer_train_mean_std`: training-sample mean and SD with configured ddof.
- `raw_imu:outer_train_robust`: training-sample median and `IQR/1.349`;
  use the three fallback choices above for degenerate scale, finally 1.

Apply `(x−center)/scale`, leave RED/IR unchanged, retain padding at 0,
and convert back to float32. When raw_ppg per-window scaling is enabled,
this fits **already window-scaled IMU copies**, not upstream SI arrays
directly. Some historical source descriptions are less precise than the
actual call order. The none option above describes the low-level
transform's capability; ordinary pipeline execution skips fitting and
application for none at `experiment.py:1726–1732`.
Finalcase defaults to none, adding no second scaling step.

## 4.27 Motion-detector representation: eight-channel reference

Intuition: to reveal how strong movement is, optical curves may be resized
within each short segment, but body-motion curves first retain their
relative sizes. A common ruler is chosen from training participants
afterward. This differs from the frailty classifier's per-window scaling
of all eight curves.

Source: `representations/motion.py:19–46,50–91,135–196,259–368`.

`motion_8ch_axes_reference_v2` uses RED, IR, and six IMU axes.
Windows are fixed at 8 seconds/2-second hops, 400 Hz:
`[W,8,3200]`. Short recordings are not padded and incomplete tail
windows are excluded. Start samples are `0,800,1600,…`.

1. RED/IR use per-window `(x−median)/IQR`, falling back to
   `1.4826*MAD`, then 1, with threshold `1e−12`.
   **There is no division by 1.349 and no [-8,8] clipping here.**
2. Copy six IMU axes in SI units into float32 windows without
   per-window amplitude scaling.
3. On every window sample from outer-training participants, fit each
   axis's median and `IQR/1.349`; for scale no greater than
   `1e−12`, fall back to population SD, then 1.
4. During application, move one axis at a time into a float64 working
   array, subtract center/divide by scale, then return to float32.
   PPG retains its already scaled values. Axiswise processing reduces
   peak memory without changing numerical operation order.

These tensors feed the motion model, not the frailty raw eight-channel
input directly. External-model training/inference and thresholds are
described in the quality/motion chapter. Finalcase disables the motion
detector and does not execute this branch.

## 4.28 Motion-detector representation: eleven-channel derived-signal ablation

Intuition: add three curves to the six directional curves: overall
movement size, rotation size, and abruptness of change. These are
explicit additional information, not hidden additions to a supposedly
six-axis input.

The source is the same as above; profile is
`motion_11ch_derived_augmentation_ablation_v2`.
Input `[W,11,3200]` adds `|dynamic_acc|`, `|gyro|`,
and `|jerk|`, each the square root of summed squared axes,
with units m/s², rad/s, and m/s³. Physical-signal construction
is described in the preprocessing chapter.

PPG still uses only per-window median/IQR→MAD→1.
All nine IMU/derived curves use training-fold
median/(IQR/1.349)→population SD→1.
This has a different schema and scaler from the eight-channel version;
a six-axis scaler cannot directly transform nine curves.
The added three belong only to this motion ablation and do not become
frailty raw-model channels 9–11.

## 4.29 Fusion: combine two information sources once per file

Intuition: a recording has many small pictures and one summary card.
First combine the pictures into one overall impression, then combine
that impression with the card once. Repeating the same card for each
picture would make long recordings submit identical information
unnecessarily many times.

Sources: `masked_file_mean` in `representations/fusion.py:5–14`;
`FileBagFusionClassifier` in `models/fusion.py:21–117`;
dataset assembly at `experiment.py:1938–1947`.

Inputs are window bags `[batch_files,W_i,8,T]`, window masks,
optional sample masks, and one `[batch_files,2D]` feature vector
per file (4.22.3). Only real windows enter the signal encoder;
padded bag positions are not encoded. Let each valid window's
embedding be `e_w`.

### 4.29.1 Mean pooling

The NumPy `masked_file_mean` computes
`mean(embeddings[window_mask],axis=0)`.
Model `pooling: mean` uses `sum(m_w*e_w)/sum(m_w)`.
Every file needs at least one valid window.

### 4.29.2 Attention pooling

`pooling: attention` computes `a_w=uᵀe_w+b`,
sets invalid-window values to `−∞`, and forms weights
`α_w=exp(a_w)/sum_valid exp(a)`.
The pooled representation is `sum(α_w*e_w)`.
`u,b` are learned parameters, not SQI scores or additional
deterministic preprocessing.

### 4.29.3 File-vector encoding and concatenation

Line 47 computes `h=ReLU(W_f*v+b_f)`, with default
`feature_hidden_dim=32`.
Lines 105–106 concatenate `pooled_signal` and `h` along
the feature axis, then apply `ReLU(W_c*[pooled_signal,h]+b_c)`
and training-time dropout, default 0.2.
Fusion width defaults to 64.
A final linear classifier produces per-file class scores.

Choose signal encoders with `FileBagFusionCompact`,
`FileBagFusionInception`, or composable `FileBagFusion`.
Their network architectures are outside this chapter.
The four representations are not merely data-format aliases:
raw learns window waveforms; feature_vector uses file statistics;
feature_matrix learns time-ordered window statistics;
fusion combines file-level waveform representations with file statistics.

## 4.30 Inputs/outputs, fitting scope, and missing-value semantics

| Module | Main output | Fitted from training cohort? | Missing/short-data handling |
|---|---|---|---|
| Four peak detectors | Original-grid peaks + PPI + validity/adjacency | No; within-recording algorithms | Fail for insufficient duration/peaks; valid segments do not cross gaps |
| PRV | Field dictionary + validity | No; within-recording statistics | Independent eligibility for each category; NaN when insufficient |
| Morphology/optical | Beatwise arrays + file statistics | No | Route restrictions, at least 3 valid beats; fieldwise missingness |
| 115-field engineering sequence | `[W,115]` | Extraction no; separate scaler yes | Complete windows; below 80% finite samples makes channel fields missing |
| File vector | `[D]` + validity | Extraction no; fold transform yes | Unmeasured values are NaN/false |
| Fusion vector | `[2D]` | Uses training-fold vector transform | Neutral 0 + separate 0/1 channels |
| Window matrix | `[146,K_i]` + row mask | Yes for the 146-column transform | Ineligible entries become transformed 0; validity is not a predictor channel |
| Frailty raw | `[W,8,T]` + sample mask | Per-window no; optional IMU post-transform yes | Invalid real samples drop the whole window; explicit padding is 0/false |
| Motion reference/ablation | `[W,8/11,3200]` | Per-window PPG no; IMU scaler yes | Complete eight-second windows required; no short-window padding |
| Fusion pooling/encoding | One representation per file | Yes for attention/network | Pools only valid windows |

All training-cohort-fitted artifacts must accompany the model and only be
applied, not refitted, to OOF/new participants. Label-free within-recording
operations do not make all downstream outputs safe to cache across folds.
Once results depend on fold scalers, quality calibration, or routing,
they belong to that fold's state.

## 4.31 Coverage checklist and verification boundaries

This chapter follows the numerical core of:

- `signal/peaks.py`: historical prominence, polarity scoring, interval validity.
- `peaks/`: resolver, MSPTDfast, Aboy v1/v2, dual-wavelength pairing,
  and event matching; `__init__.py` only exports symbols and adds no algorithm.
- `signal/prv.py`: basic rate, time PRV, Poincaré, sample entropy,
  spectral PRV, and eligibility.
- `signal/morphology.py`, `signal/optical.py`: beat shapes,
  AC/DC/PI, and dual-optical agreement.
- `features/engineering.py`: 115 features, Welch, and fold-local engineering transforms.
- `features/registry.py`: seven-group 282-field file vectors,
  summaries, and route/missing encoding.
- `features/vector_transform.py`: file-vector fold transforms and fusion 2D encoding.
- `features/window_matrix.py`: eligible-segment assembly,
  146 window features, fold transforms, and variable-length matrices.
- `features/prv_backend_compare.py`: three fixed-PPI comparison adapters
  and local equations.
- `representations/`: raw, six-axis post-transform, two motion inputs,
  feature-vector/matrix structures, and fusion pooling.
- Necessary call boundaries: `normalization.py`;
  representation branching and DL-resampling order in `experiment.py`;
  input imputation/scaling in `models/feature_baselines.py`;
  file pooling/fusion in `models/fusion.py`.

The following boundaries must not be mistaken for completed verification:

1. This is a companion to current repository source, not a numerical
   regression report from retraining every optional configuration.
   Writing the documentation did not rerun training.
2. SciPy filtering, Welch, peak prominence, and statistics are identified
   by function and parameters, but the full third-party implementations
   are not copied line by line into this chapter. Exact library versions
   can affect numerical details.
3. Aura/rhenan internal formulas are absent from adapter source.
   This chapter checks inputs, calls, output units, and error handling,
   not equivalence of all their internals to local PRV.
4. ShapeFormer/Inception/classical-classifier training, losses, and
   optimizers belong to the model chapter; they are not claimed as
   feature algorithms fully expanded here.
5. The current detector interface has no independent PPI cleaner;
   matrix validity is not a predictor channel, whereas fusion validity
   is input; Aboy v2 physically deletes peaks.
   These are implementation semantics that generalized tutorial
   descriptions must not obscure.
