# 1. Signal Preprocessing

This chapter explains how a recording becomes arrays for downstream analysis or model input. Paths are relative to the V6 root; `file:start–end` locates code in this version and does not indicate required edits. For line-by-line review, consecutive parameter checks and result packaging are explained together, while numerical operations follow execution order. External-library calls are described using the parameters actually passed by this project; library internals not implemented in the repository are not presented as project code.

## 1.1 Reading Route, Data Shapes, and finalcase

Intuition: a recording resembles eight curves drawn side by side on one time sheet. Processing must not secretly shift one curve in time or confuse “points per second” with curve height. Organize the curves before cutting them into segments. Segments supplied to models may be rescaled separately, while curves used to measure actual amplitude, intervals, and motion magnitude remain available.

The main call path is `_fit_imu_calibrations` and `_preprocess_records` at `src/ppg_frailty/experiment.py:308-433`, followed by `build_signal_views` at `src/ppg_frailty/signal/preprocess.py:632-806`. The raw path then reaches `src/ppg_frailty/representations/raw.py:115-179` and `src/ppg_frailty/experiment.py:1703-1757`, and finally `src/ppg_frailty/experiment.py:1950-1994`.

| Stage | Inputs and outputs | Actual order or meaning |
|---|---|---|
| CSV loading | `N×8`, float64 | RED, IR, AX, AY, AZ, GX, GY, GZ columns; 400 Hz; no timestamp column |
| Static calibration | Same-participant B recording; ACC/Gyro each `N_B×3` | Extract B's `[5,100)` seconds, filter, then estimate fixed biases |
| PPG preparation | `N×2` → `N×2` | Check/fill short gaps → remove linear trend → forward-backward bandpass |
| IMU preparation | ACC/Gyro each `N×3` | Convert units → subtract B biases → forward-backward lowpass → selected gravity processing |
| Original analysis views | PPG `N×2`; several IMU `N×3`/`N` arrays | Retain 400 Hz and physical units; no per-window standardization of these upstream arrays |
| Raw windowing | `N×8` → `W×8×2000` | finalcase: 5-second windows, 2.5-second hop, at most 128 per recording |
| Raw per-window scaling | Eight channels over each window's valid rows | finalcase scales every channel independently using that window's median and interquartile span |
| Optional fold post-transform | `W×8×T` → same shape | Six IMU channels only; fitted on training participants only; not used in finalcase |
| DL resampling | `W×8×2000` → `W×8×320` | finalcase changes 400 Hz to 64 Hz only before the model; feature-analysis grid remains unchanged |

`W` is the retained window count, `N` the recording sample count, and `T` the samples per window. At the normal entry point, row index determines time: `t[n]=n/400` seconds. The caller first builds B-calibration objects for this run's participants, then constructs PPG/IMU views recording by recording. The raw branch from windowing onward follows quality/motion routing, so this chapter does not imply that every preprocessing step executes before every quality step.

Actual finalcase choices come from `configs/presets/finalcase.yaml:63-154`: third-order PPG 0.2–8 Hz; third-order ACC 20 Hz/Gyro 40 Hz; `sensor_filter_only_no_gravity_removal`; B calibration at 5–100 seconds; per-window robust scaling with `raw_imu:none`; 64 Hz model input. `quality.mode:off` at lines 155–167 does not disable loading-time physical checks, PPG input checks, or static calibration.

## 1.2 File Loading, Synchronized Time, and Missing Segments

### 1.2.1 Production CSV Entry: Checking Is Not Repair

Intuition: first confirm that all eight columns form a complete shared recording, then decide whether to pass them to subsequent preparation. This step checks the original sheet for missing writing; it does not fill it in.

`src/ppg_frailty/pipeline.py:275-332`, `_load_record`:

1. Lines 278–285 read numbers from the same file bytes, skip the header, and specify float64.
2. Lines 286–288 require two dimensions and exactly eight columns.
3. Lines 291–300 apply physical checks to the entire recording. Even if only an initial segment is requested, cropping at lines 301–307 follows the full-record check.
4. Lines 311–315 assign the first two columns to PPG, the next three to ACC, and the final three to Gyro; input units are declared as `g` and `deg/s`.

Important implementation boundary: `physical_recording_qc_thresholds_v2` at `src/ppg_frailty/data/qc.py:296-313` sets a 5-second minimum and `maximum_nonfinite_gap_s=0.0`. Consequently, NaN/Inf at the normal finalcase CSV entry are rejected first. The existence of a “100-sample interpolation” function below does not mean the production entry automatically repairs missing data. The repair algorithm is a real reusable capability available through another eligible loader or direct calls; this explanation does not change entry-point behavior.

### 1.2.2 Explicit Timestamp Validation

Intuition: neighboring points should have equal spacing. A small number of minor deviations may be acceptable; time reversal or substantial speed changes are not. This module checks the positions but does not rearrange uneven samples.

`src/ppg_frailty/signal/preprocess.py:88-109`, `validate_timestamp_grid`:

1. Without timestamps, return immediately; otherwise flatten to a finite float64 vector of length `N`.
2. Require every `Δt_i=t_{i+1}-t_i` to be positive.
3. For ideal spacing `d=1/f_s`, compute `r_i=|Δt_i-d|/d`.
4. Reject when `percentile(r,99)>0.05`. This is not a requirement that every individual interval differ by less than 5%.

The production CSV loader supplies no timestamps, relying on the manifest's 400 Hz declaration. Historical chunked IMU `_timestamp_ok` also compares the first sample of the current chunk with the last sample of the previous one; see `src/ppg_frailty/signal/imu.py:389-411`. Cross-chunk boundary error must not exceed 5%.

### 1.2.3 Short Interior PPG Gaps: Linear Interpolation

Intuition: when both ends of a gap are visible, connect them with a ruler and fill missing points along that line. Do not guess the original curve when the gap reaches a paper edge or is too long.

Input is `N` or `N×C`; outputs are repaired `N×C` data, the original boolean-validity matrix, a repair-position matrix, and check results. Production PPG calls use `C=2`. `inspect_and_repair` is at `src/ppg_frailty/signal/preprocess.py:130-203`:

1. Lines 138–150 convert to float64 and make one-dimensional input a single column. `valid=isfinite(source)` preserves the original missingness; successful repair does not make every original sample valid.
2. Line 151 obtains `L_max=round(max_gap_sec×400)`. Production calls convert `signal.gap_repair.max_gap_samples`; the default is 100 samples, or 0.25 seconds.
3. Lines 153–178 identify contiguous `[start,stop)` gaps column by column. `_true_runs` at 111–121 pads the boolean array with False at both ends, then differences it to locate 0→1 and 1→0 boundaries.
4. Lines 180–185 execute `np.linspace(left,right,L+2)[1:-1]`. For endpoints `a` and `b` and `L` missing samples, repaired point `j=1…L` is `a+j(b-a)/(L+1)`.
5. Line 186 sets `repair_mask=True` only at newly filled positions; the original `source_valid_mask` remains unchanged.
6. Edge gaps, excessive gap length, or an entirely missing column produce `status='failed'`. This function has no 1% total-missingness limit.
7. Lines 191–200 check strictly equal plateaus after repair. `_longest_constant_run` at 123–128 counts consecutive equal samples; by default, `round(1.0×400)=400` samples cause rejection. Fractions at the minimum/maximum are recorded as metrics but are not thresholded here.

Configuration is `signal.gap_repair.method:linear_inside_only`, `max_gap_samples:100`, and `edge_extrapolation:false`; flatline duration comes from `quality.flatline_duration_s:1.0`. Lines 672–676 of `build_signal_views` require matching quality long-gap and signal-repair lengths. `max_gap_samples` is a nonnegative integer; zero repairs no nonempty gap.

### 1.2.4 Historical IMU Short Interior Gaps: A Different Rule

The ruler-between-endpoints intuition is the same, but this older route also limits total missingness per column. The two sets of rules are not identical.

`_repair_chunk` at `src/ppg_frailty/signal/imu.py:424-446` is called at line 552 of `CausalImuProcessor.process_chunk`:

- Input is the current `N×6` chunk, repaired column by column; maximum gap length is fixed at `round(0.25×f_s)`.
- Lines 433–434 reject an entirely missing column or a nonfinite fraction strictly greater than 0.01.
- Line 437 rejects edge gaps and excessive length; lines 439–443 use the same linear interpolation formula as the preceding section.
- The three calibrated IMU routes do not call this function; `motion_imu.py:340-341,544-545` directly require fully finite inputs. Changing PPG `max_gap_samples` does not change this historical IMU function's fixed 0.25 seconds.

## 1.3 PPG Preparation and Two Bandpass Options

### 1.3.1 Removing a Linear Trend

Intuition: straighten a sheet that gradually slopes upward or downward so its fluctuations sit around a flatter ground. This does not force all fluctuations to have the same height.

Input `native:N×2` has repaired gaps but retains raw-count amplitudes. In `preprocess_ppg_pair` at `src/ppg_frailty/signal/preprocess.py:227-267`, line 253 sets `native=qc.repaired` and line 262 executes `signal.detrend(native,axis=0,type='linear')`.

Mathematically, each column receives a best-fit line `a+bn` and outputs `u_n=x_n-(a+bn)`; “best” means the least-squares interpretation of the linear-detrend call. The project calls SciPy rather than hand-writing an `a,b` solver loop. The raw-count `x_native` view is preserved, not overwritten by `working`. Columns are detrended independently without mixing RED and IR.

### 1.3.2 Default Bandpass: Third-Order Butterworth, 0.2–8 Hz

Intuition: retain moderately paced fluctuations while weakening slow drift and fast spikes. Process left-to-right and then right-to-left to avoid shifting all peaks later. Edge handling is still required, and very short curves cannot support it.

`design_ppg_sos` is at `src/ppg_frailty/signal/preprocess.py:205-225`; line 264 performs the filtering:

1. `signal.butter(order,[low,high],btype='bandpass',fs=400,output='sos')` produces section coefficients; defaults are `order=3,low=0.2,high=8.0`.
2. `signal.sosfiltfilt(sos,working,axis=0)` processes both columns of the complete recording forward and backward, before any 5-second model windows are cut.
3. For one second-order coefficient row `[b0,b1,b2,a0,a1,a2]`, the difference equation is `a0*y[n]=b0*x[n]+b1*x[n−1]+b2*x[n−2]−a1*y[n−1]−a2*y[n−2]`. Sections are cascaded; SciPy implements the forward-backward operation.
4. The combined effective magnitude response is the square of the one-direction magnitude response. The configured “third order” does not mean that only one third-order filtering pass occurs. Edge-extension length and initial state use library defaults; the project adds no numerical constants for them.
5. A forward-backward `ValueError` becomes `zero_phase_filter_insufficient_length`; there is no automatic switch to one-direction filtering.

Configuration keys are `signal.ppg_filter.order/low_hz/high_hz`. Order is an integer from 1–20; `0<low<high<200`. `family='butterworth_sos'`, `phase='zero_phase'`, `short_signal_policy='reject'`, and `notch_enabled=false` describe the current implementation, not an additional unexplained notch option; see lines 543–557.

### 1.3.3 Alternative Bandpass: Third Order, 0.5–5 Hz

Intuition: narrow the range of permitted fluctuations, weakening more slow and fast changes. This can also remove genuine pulse detail, so it is a comparison option, not universally better cleaning.

`src/ppg_frailty/signal/preprocess.py:21-76` registers `butterworth_0p5_5hz_ablation` as `(0.5,5.0,3)`. Computation still uses the preceding section's detrending and forward-backward Butterworth, with corresponding `signal.ppg_filter.low_hz/high_hz/order` values. Registering a name neither runs another experiment automatically nor selects the closest band for the user. finalcase uses 0.2–8 Hz, not this branch.

## 1.4 Same-Participant Static B Calibration

### 1.4.1 Selecting the Calibration Recording and Interval

Intuition: each person first supplies a seated, stationary segment to reveal the device's baseline acceleration and rotation offsets. Subtract these offsets from that person's other recordings rather than correcting them with another person's data.

`_fit_imu_calibrations` at `src/ppg_frailty/experiment.py:308-359` serves all three calibrated gravity branches. Lines 328–336 require the same participant, exactly role B, and QC pass or pass_with_warnings, selecting the first recording by descending duration then ascending record_id. Without an eligible B recording, that person's preprocessing fails; it does not silently disable calibration.

`src/ppg_frailty/signal/motion_imu.py:312-394`, `fit_motion_imu_calibration`:

1. Lines 332–337 first convert physical units: default ACC `g` is multiplied by 9.81 and Gyro `deg/s` by `π/180`.
2. Lines 342–345 extract `[round(5×400),round(100×400))=[2000,40000)`, totaling 38,000 samples. The source must cover this interval, which must contain at least 16 samples.
3. Lines 346–351 apply third-order 20 Hz and 40 Hz forward-backward lowpasses to the already-extracted static segment, rather than filtering all of B before cropping. Thus this segment determines the edge conditions.

Configuration is `signal.imu.calibration_start_s:5.0` and `calibration_stop_s:100.0`, with `0≤start<stop` and sufficient actual recording length. The general function also supports explicitly declared `PTT_SIT_STATIC_CALIBRATION`, but ordinary frailty-experiment callers select only B.

### 1.4.2 Mean After Outlier Removal

Intuition: find the central location, set aside a few points that are far away, and average the rest. Each of the three axes decides independently which points are far away; rejected times need not match across axes.

`_robust_mean` at `src/ppg_frailty/signal/motion_imu.py:235-246` computes `m=median(x)` and `d=median(|x−m|)` per column. If `d≤1e−12`, use all-sample mean; otherwise retain `|x−m|/d<3.5` and average. Here `d` is not multiplied by 1.4826 and is not the interquartile span used in later model normalization. Thresholds 1e−12 and 3.5 are fixed function constants, not YAML controls.

### 1.4.3 Initial Tilt and Fixed Biases

Intuition: the three stationary mean-axis values indicate where “down” lies in device coordinates. Draw a fixed-length arrow in that direction. The difference between the mean reading and this arrow becomes the device's fixed offset. This is not a universal calibration that independently identifies every device error.

At `src/ppg_frailty/signal/motion_imu.py:248-264` and lines 352–365, let the robust three-axis mean be `(ax,ay,az)`:

`φ=atan2(ay,sqrt(ax²+az²))`

`θ=atan2(−ax,sqrt(ay²+az²))`

`g_B=g*[−sin(θ)cos(φ), sin(φ), cos(θ)cos(φ)]`, with default `g=9.81 m/s²`.

Then `b_acc=mean_robust(acc)−g_B` and `b_gyro=mean_robust(gyro)`. The calibration object stores `φ,θ`, both three-dimensional biases, and source identity. Quality metrics merely record gravity-magnitude error, per-axis Gyro RMS, and sample count. Line 365 explicitly sets `quality_threshold_applied=False`; the module does not automatically prove that the segment was truly stationary.

Fitting uses this person's B without class labels or cross-person statistics. It may use a held-out participant's own static calibration recording, which is also required at deployment. This differs from fitting a cross-person scaler using held-out-person data.

## 1.5 Shared Unit Conversion, Bias Subtraction, and Filtering for Calibrated IMU

Intuition: standardize the ruler's units, subtract its stationary offset, and smooth fine rapid jitter. Removing the “downward” component is a separate choice afterward.

`_prepare_si_inputs` at `src/ppg_frailty/signal/motion_imu.py:515-563` takes ACC and Gyro each shaped `N×3`, at least 16 fully finite rows, and same-participant calibration, returning same-shaped float64 SI-unit arrays:

1. Lines 536–541 convert units. `_convert_profile_acceleration` at 266–282 supports `g→×g`, `mg→×g/1000`, and `m/s2`/`m/s^2→×1`; default gravity constant is 9.81. `convert_gyro` at `signal/imu.py:88-94` supports `deg/s→×π/180` and `rad/s→×1`.
2. Lines 546–547 compute `a_c=a_SI−b_acc` and `ω_c=ω_SI−b_gyro`, broadcasting three-dimensional constants across the full recording.
3. Lines 548–549 call `_zero_phase_lowpass` (225–233): `butter(order,cutoff,lowpass,fs,sos)` followed by `sosfiltfilt(axis=0)`. Defaults are 20 Hz for ACC, 40 Hz for Gyro, order 3.
4. Outputs are `a_f,ω_f`. Failure of forward-backward filtering on short input raises an error; there is no hidden branch that skips filtering or substitutes one-direction filtering.

Configuration is `signal.imu.sensor_lowpass_acc_hz:20.0`, `sensor_lowpass_gyro_hz:40.0`, and `sensor_filter_order:3`. Cutoffs are strictly inside `(0,200)` and order is an integer 1–20. Reordering bias subtraction and filtering or replacing forward-backward with one-direction filtering changes values; this chapter only describes the current implementation.

## 1.6 Five Mutually Exclusive IMU Gravity/Orientation Algorithms

`signal.imu.gravity_method` is resolved at `src/ppg_frailty/signal/preprocess.py:356-491`. Its implicit default is `profile_a_lowpass_0p3hz`, unlike finalcase's explicit `sensor_filter_only_no_gravity_removal`.

### 1.6.1 finalcase: sensor_filter_only_no_gravity_removal

Intuition: correct device errors and rapid spikes, but retain the component caused by Earth's constant pull. Zero-filled orientation fields are interface placeholders, not measurements proving that the device remains level.

`_preprocess_motion_profile(mode='no_gravity')` is at `src/ppg_frailty/signal/motion_imu.py:648-740`; its wrapper entry is at 790–812:

1. Lines 661–670 perform the preceding section's B-bias calibration and sensor filtering normally.
2. Line 706 sets `gravity=zeros_like(acc_filtered)`; line 707 gives roll and pitch length-N zero-placeholder arrays.
3. Line 613 of `_motion_result` computes `dynamic=a_f−gravity`, so the array named `dynamic_acc_mps2` actually retains gravity and equals `a_f` exactly.
4. Metadata at 709–712 explicitly disables gravity estimation/subtraction and identifies the output as calibrated, filtered acceleration containing gravity.

Gravity-lowpass and EKF numerical parameters are not read; no gravity filter silently operates in this branch. finalcase uses this branch. `no_gravity` means neither `no_calibration` nor completely raw acceleration.

### 1.6.2 profile_a_lowpass_0p3hz: Treating Slow Variation as Gravity

Intuition: regard acceleration as a slowly moving base plus faster actions. Draw that slow base, then subtract it. Slow movements may also be absorbed into the base, so this is not exact physical separation for arbitrary movement.

In the same function, lines 689–703, with wrapper at 766–788:

1. Apply another per-axis forward-backward lowpass to shared `a_f`: `g_hat=LPF(a_f,0.3 Hz,order=4)`.
2. Lines 696–697 insert each `g_hat` into the two angle formulas of 1.4.3, solely to describe output orientation.
3. Line 613 of `_motion_result` computes `a_dyn=a_f−g_hat`. This lowpass does not constrain the vector's length to 9.81.

Configuration is `gravity_lowpass_hz:0.3` and `gravity_filter_order:4`. This fourth-order forward-backward algorithm differs from the historical second-order one-direction method in 1.6.5; a shared 0.3 Hz name does not make them interchangeable. It has no trained parameters or labels and is deterministic within the recording.

### 1.6.3 calibrated_roll_pitch_ekf: Static Initialization and Five-State Tracking

Intuition: rotation readings tell you how far the device just turned but drift over time; acceleration suggests “down” but movement distorts that clue. At each step, predict orientation from rotation, then move the prediction toward the acceleration clue according to each source's uncertainty, while correcting residual rotation bias.

`_run_roll_pitch_ekf` is at `src/ppg_frailty/signal/motion_imu.py:399-513`, connected to outputs at 671–688. Inputs are shared-preprocessed `a_f,ω_f:N×3`. State `x=[φ,θ,bx,by,bz]` contains two angles in radians and three residual angular-velocity biases in `rad/s`.

Step A: initialize (409–426). `φ0,θ0` come from B; residual biases start at zero. `P0=diag(1,1,0.5,0.5,0.5)`; `Q=diag(5,5,0.05,0.05,0.05)/f_s`; `R0=diag(0.5,0.5)`; `dt=1/f_s`; `H=[[1,0,0,0,0],[0,1,0,0,0]]` directly observes only the angles.

Step B: rotation-based prediction (427–446). Starting at `i=1`, compute `ω=ω_f[i]−b[i−1]`, denoted `(gx,gy,gz)`, using the previous angles:

`dφ/dt=gx+gy*sinφ*tanθ+gz*cosφ*tanθ`

`dθ/dt=gy*cosφ−gz*sinφ`

`φ_pred=φ+dt*dφ/dt`, `θ_pred=θ+dt*dθ/dt`, `b_pred=b`. Stop when `|cosθ|<1e−6` to avoid division by zero near the vertical singularity of this angle representation.

Step C: propagate uncertainty (447–469). Let `s=sinφ,c=cosφ,t=tanθ,u=1/cos²θ`. The nonidentity components of F are:

```text
F row 1 = [1+dt*(gy*c*t-gz*s*t), dt*(gy*s*u+gz*c*u), -dt, -dt*s*t, -dt*c*t]
F row 2 = [dt*(-gy*s-gz*c), 1, 0, -dt*c, dt*s]
F rows 3–5 = corresponding identity-matrix rows for the biases
P_pred = F @ P @ F.T + Q
```

These entries are local rates of change of the prediction with respect to each state component, explicitly listed in the source. Replacing F with identity would change the algorithm.

Step D: obtain an acceleration orientation clue and adjust its weight (470–482). `z=[roll_from_acc,pitch_from_acc]` uses the formulas in 1.4.3. Compute

`d=max(0,||a_f[i]||−g)/g`, `scale=1+α*d`, `R=diag(R0_diagonal*scale)`, with default `α=3`.

This is one-sided `max(0,magnitude−g)`, not `abs(magnitude−g)`; a magnitude below g does not increase R under this rule.

Step E: fusion (483–499). `wrap(v)=atan2(sin v,cos v)` brings angle differences into `[-π,π]`, avoiding treating +π and −π as a full turn apart.

```text
e = wrap(z - H @ x_pred)
S = H @ P_pred @ H.T + R
K = solve(S, H @ P_pred).T
x_new = x_pred + K @ e
P_new = (I-KH) @ P_pred @ (I-KH).T + K @ R @ K.T
P_new = (P_new + P_new.T)/2
```

Wrap both updated angles again. A nonfinite state or minimum eigenvalue of P below −1e−9 raises an error. The source uses the Joseph form, not simply `(I−KH)P`. The residual-bias trajectory is only for internal-state tracking and diagnostics; final Gyro channels remain `ω_f` from shared `_prepare_si_inputs`, without subtracting that residual trajectory again sample by sample.

Step F: convert orientation back to gravity (681). Use `φ,θ` in the three-dimensional gravity formula of 1.4.3, with magnitude set by g, then compute `a_dyn=a_f−g_hat`. There is no magnetometer or absolute heading/yaw correction.

Controls under `signal.imu` are `process_covariance_diagonal_per_second:[5,5,0.05,0.05,0.05]`, `observation_covariance_diagonal_rad2:[0.5,0.5]`, `initial_covariance_diagonal:[1,1,0.5,0.5,0.5]`, `dynamic_observation_scale:3.0`, and `gravity_mps2:9.81`. Covariance diagonal entries must be positive and finite, the dynamic coefficient nonnegative; vector lengths are respectively 5, 2, and 5. Defaults are at 57–78. This algorithm does not train on class labels, but full-record forward-backward prefiltering means the complete chain is not purely causal real-time processing.

### 1.6.4 quaternion_error_state_ekf: Historical Orientation Tracking Without B Precalibration

Intuition: without an initial stationary segment, infer device orientation from current readings and correct it after each rotation. Confidence takes time to develop. If clues conflict too strongly, temporarily trust rotation alone or declare the estimate invalid instead of inventing a smooth-looking direction. Apparent stability does not prove that every orientation and bias is identifiable.

`preprocess.py:752-766` maps the public name to internal `no_precalibration_ekf`. `NoPrecalibrationEskf` is implemented at `src/ppg_frailty/signal/imu.py:193-387`. Its chain first uses this file's unit conversion at 75–94 and one-direction filtering at 170–191, not B calibration or forward-backward filtering from 1.5. Here `STANDARD_GRAVITY=9.80665` differs from the calibrated routes' 9.81.

Geometry helpers (96–168):

- `skew(v)` constructs a matrix satisfying `skew(v)u=v×u`.
- `quat_normalize(q)=q/||q||` fails for nonfinite norm or norm below 1e−15. Quaternion order is `[w,x,y,z]`.
- `quat_multiply` at 113–121 implements the Hamilton product component by component to compose rotations. For left `(w₁,x₁,y₁,z₁)` and right `(w₂,x₂,y₂,z₂)`, outputs are `w=w₁w₂−x₁x₂−y₁y₂−z₁z₂`, `x=w₁x₂+x₁w₂+y₁z₂−z₁y₂`, `y=w₁y₂−x₁z₂+y₁w₂+z₁x₂`, and `z=w₁z₂+x₁y₂−y₁x₂+z₁w₂`. Swapping operands generally changes the result; the code always uses the stated order.
- `quat_exp(v)`: let `α=||v||`; below 1e−12 normalize `[1,v/2]`, otherwise use `[cos(α/2),(v/α)sin(α/2)]`.
- `quat_to_rotation` at 135–143 first normalizes q, then expands the full 3×3 rotation matrix R. Rows are `[1−2(y²+z²),2(xy−wz),2(xz+wy)]`, `[2(xy+wz),1−2(x²+z²),2(yz−wx)]`, and `[2(xz−wy),2(yz+wx),1−2(x²+y²)]`. R rotates device coordinates into reference coordinates. Predicting direction back in device coordinates uses its transpose; these directions are not interchangeable.
- `quat_from_two_vectors` at 145–157 normalizes both directions and generally returns `normalize([1+dot,cross])`; nearly opposite directions use a perpendicular axis for a half-turn.
- `tangent_basis` at 159–168 chooses the coordinate axis least parallel to the direction, then uses two successive cross products to produce two perpendicular unit axes, giving `B:2×3`.

Per-sample mathematics:

1. Initialization (212–229, 261–272). On the first `0.5g≤||a||≤1.5g` sample, rotate acceleration direction to `[0,0,1]` and set bias to zero. Initial uncertainties are 20° tilt, 180° about gravity, and 5° angular-velocity bias. `P_angle=tilt²(I−ddᵀ)+yaw²ddᵀ` and `P_bias=I*bias²`, where d is the initial acceleration unit direction.
2. Prediction (274–291). Do not advance q/P on the just-initialized sample (line 275, `if not initialized_now`); this step applies only to subsequent samples. `ω=gyro−bias`; `q←normalize(q⊗exp(ωdt))`. The six-dimensional error state contains three small angles and three biases. `F=[[-skew(ω),−I],[0,0]]`; `Φ=I+Fdt+(Fdt)²/2`. With `v_g=gyro_noise_density²` and `v_b=gyro_bias_random_walk²`: `Qaa=I(v_g dt+v_b dt³/3)`, `Qab=Qbaᵀ=−I v_b dt²/2`, `Qbb=I v_b dt`; `P←ΦPΦᵀ+Q`.
3. Update cadence (295–319). Observe acceleration direction only when the global sample index is divisible by 4. `d_pred=R(q)ᵀ[0,0,1]`, `d_obs=a/max(||a||,1e−15)`; `ρ=abs(||a||/g−1)`; `η=||a−a_previous_sample||/(g dt)`.
4. Observation-noise factor `s=clip(1+(ρ/0.05)²+(η/2)²,1,100)`; s≥25 merely marks downweighted status. `e=B(d_obs−d_pred)`; `H=[B*skew(d_pred),0]`; `R_obs=(5° converted to radians)²*s*I₂`; `S=HPHᵀ+R_obs`; `NIS=eᵀ solve(S,e)`.
5. Update only while magnitude remains in `[0.5g,1.5g]` and `NIS≤13.8155` (320–333). `K=solve(S,HP)ᵀ`, `δ=Ke`; `q←normalize(q⊗exp(δ_first_three))`; `bias←bias+δ_last_three`. Apply Joseph form to P, then `P←JPJᵀ` with `J=block_diag(I−skew(δ_first_three)/2,I)`.
6. Numerical cleanup (334–341). Symmetrize P; eigenvalues below −1e−10 fail. Eigenvalues between that threshold and 1e−15 are raised to 1e−15 before reconstructing the matrix.
7. Validity (235–240, 343–387). Tilt uncertainty is `σ=degrees(sqrt(max_eigenvalue(B P_angle Bᵀ)))`. Tracking requires at least 0.5 seconds, 20 accepted updates, and σ≤10°. Nonfinite q/P or any bias axis exceeding 0.35 rad/s in absolute value invalidates the estimate. Before tracking starts, status remains `initialization_pending`; after tracking has started, more than 2 seconds of prediction-only operation or σ>20° also invalidates it. Failure enters `no_estimate` until an explicit reset. Other valid states are tracking or prediction-only. `g_hat=R(q)ᵀ[0,0,9.80665]`.

Defaults at `imu.py:19-32` include angular-velocity noise density 0.002, bias random walk 0.0002, observation-angle noise 5°, and the thresholds above. These are Python-level `EskfConfiguration` settings. Ordinary `signal.imu` YAML currently exposes sensor-lowpass parameters, not CLI keys for all these values.

Full-chain state management is at `imu.py:467-727`. Filter state, orientation state, previous dynamic acceleration, and timestamps persist across chunks. The estimator is copied before processing and committed at 710–719 only if the entire chunk succeeds. Invalid samples during initial/reinitialization remain NaN/False; raw windows touching them are discarded, not zero-filled and called valid. Lines 604–608 additionally output `gravity_confidence=exp(−σ/10)` only for valid orientation and finite σ, otherwise zero. This is not a correctness probability calibrated against class labels. The existence of this historical function does not prove completion of finalcase deployment/comparison protocols for the “no-B silent-calibration ablation.”

### 1.6.5 low_pass_0p3hz: Historical One-Direction Lowpass Gravity

Intuition: follow the slowly moving base using only the curve already seen, without future information. This supports chunk-by-chunk processing, but tracking lags and differs from looking forward and backward over the complete curve.

`preprocess.py:759` maps the public name to internal `lpf_0p3`. See `src/ppg_frailty/signal/imu.py:170-191,625-639`:

1. Convert units using 9.80665, then apply third-order one-direction ACC 20 Hz/Gyro 40 Hz filtering.
2. `_causal_filter_axes` uses Butterworth SOS. The first chunk initializes `zi=sosfilt_zi(sos)*source[0]`; later chunks reuse the previous end state. `sosfilt(...,zi=state)` returns both output and next state.
3. `g_hat=causal_LPF(a_f,0.3Hz,order=2)` and `a_dyn=a_f−g_hat`. A finite gravity vector marks that position estimable.
4. There is no B calibration, orientation fusion, or fixed-magnitude constraint. Dynamic-acceleration differences still require adjacent valid samples.

Configuration is `signal.imu.gravity_lowpass_hz:0.3` and `gravity_filter_order:2`. Standalone `estimate_gravity_lpf` at 790–819 can supply a zero-Gyro placeholder by default because this branch does not infer orientation from Gyro; the standard eight-channel pipeline does not thereby omit Gyro input.

## 1.7 Dynamic Acceleration, Angular-Velocity Magnitude, and Rate of Change

Intuition: three axes form a spatial arrow whose length expresses overall magnitude regardless of direction. The larger the change between successive arrows, the more abrupt the movement change.

Calibrated outputs are built by `_motion_result` at `src/ppg_frailty/signal/motion_imu.py:602-646`:

`a_dyn[n]=a_f[n]−g_hat[n]`

`j[0]=0` and `j[n]=(a_dyn[n]−a_dyn[n−1])*f_s` correspond to line 614, `diff(...,prepend=dynamic[:1])*fs`.

`A_mag=sqrt(ax_dyn²+ay_dyn²+az_dyn²)`; `Omega_mag=sqrt(gx²+gy²+gz²)`; `J_mag=sqrt(jx²+jy²+jz²)`. The nine-channel order is three dynamic-acceleration axes, three angular-velocity axes, A_mag, Omega_mag, J_mag; units are respectively m/s², rad/s, and m/s³. Production raw classification takes only the first six IMU axes; it does not automatically insert the three magnitudes into the eight-channel model.

Historical causal `_jerk` (`imu.py:448-465`) differs: the current chunk's first sample uses a cross-chunk difference only if a previous final sample exists and both are valid. The initial first sample has no predecessor and remains NaN. No difference is filled when either neighbor is invalid. Thus `imu_valid_mask` requires both orientation and difference validity, not merely finite raw values on all six axes.

`compare_ekf_lpf_gravity` (`imu.py:831-879`) is descriptive only: on mutually finite positions it computes `RMSE=sqrt(mean(d²))`, `MAE=mean(abs(d))`, per-axis RMSE, and mean gravity magnitude. It does not automatically select an algorithm with the smaller metric; `selection_performed=False`.

## 1.8 Three PPG Views and an Optional Artifact-Reduced View

Intuition: retain several copies of a curve for different purposes: original heights, a version without slow drift and spikes, and a potentially further-processed rhythm view. If the last copy changes the hill's shape, it can no longer measure the original hill's height and width.

Created at `src/ppg_frailty/signal/preprocess.py:794-804`; `CanonicalSignalViews` is at `src/ppg_frailty/signal/views.py:22-157`:

- `x_native`: original counts after PPG input repair only, without detrending/bandpass; used by quantities requiring the baseline.
- `x_filter`: detrended, bandpass-filtered count waveform; the `x_analysis` property at 41–43 always returns it.
- `x_analysis_rate`: initialized as `x_filter.copy()`; lines 45–59 require elementwise equality with x_filter. Despite its name suggesting dynamically processed analysis, this field is not overwritten by a nonidentity reducer.
- `x_ar`: an optional reducer's separate output, attached by `with_artifact_result` at 118–157. Identity must equal x_filter; nonidentity is marked `ARTIFACT_RATE_ONLY`, `rate_only=True`, and `q_morph_state='not_applicable'`, with a samplewise validity mask.
- Rhythm waveform retrieval uses `analysis_signal` (78–87): x_ar for nonidentity, otherwise x_filter. Eight-channel raw windows use `x_analysis` and therefore still use x_filter.

Reducer failure raises an error rather than presenting the direct curve as a successful reduced result. These are routing rules; individual reducers are explained in the quality/motion chapter. All views remain on the same 400 Hz grid; no additional resampling or filtering occurs here.

## 1.9 Windowing: Time Slices, Not Random Sampling

### 1.9.1 Converting Seconds to Samples and Complete Windows

Intuition: place a fixed-width frame on the time sheet and move it by the same distance each time. Both width and movement must correspond exactly to integer sample counts; no hidden half-sample can accumulate per step.

`WindowPlan._sample_count/plan` at `src/ppg_frailty/data/windows.py:59-67,106-163` computes `L=round(window_seconds*f_s)` and `H=round(hop_seconds*f_s)`. Each original product must differ from its integer by at most 1e−9, and L/H must be positive.

Default profiles at `src/ppg_frailty/module_registry.py:1080-1099` are engineering 10 seconds/2 seconds and raw 5 seconds/2.5 seconds. finalcase raw uses L=2000, H=1000 at 400 Hz, not initial windowing at 64 Hz.

### 1.9.2 Left-Aligned Regular Grid

Intuition: start the first frame at the paper's left edge and keep only positions where the complete frame remains on the paper.

`windows.py:126-130` uses starts `0,H,2H,…≤N−L`. Configuration `windows.<profile>.end_alignment:left_start_regular_grid` maps to internal `start` at `module_registry.py:1192-1195`. Unless tail padding is enabled, a final incomplete section does not become a separate window.

### 1.9.3 Append the Rightmost Complete Window

Intuition: first lay frames from the left, then check whether the last frame reaches the right edge exactly. If not, add a frame flush with that edge. The last two frames may therefore be closer together than usual.

`windows.py:131-137` first generates the preceding grid, then appends `N−L` if it is not already the final start. finalcase selects `include_right_aligned_if_distinct`. This does not shift the entire grid rightward.

### 1.9.4 Right Alignment: Low-Level API Capability

Intuition: place a frame at the rightmost edge, work backward to the left, then restore left-to-right order.

Internal `end` at `windows.py:142-145` implements `sorted(range(N−L,−1,−H))`. Production `normalize_window_config` accepts only left-start and append-right-window names (`module_registry.py:1119-1120`), so this is a `WindowPlan` Python API capability, not an existing YAML option.

### 1.9.5 Short Recordings and Right-Side Tail Padding

Intuition: only when explicitly allowed, fill an incomplete frame's right side with blank space. Tell downstream modules which positions are blank; blank space is not a real measurement.

`windows.py:114-124,138-141,165-185`:

- For a recording shorter than one window, `reject` raises `ShortRecordError`; `pad_right` creates one start-zero window with true length N, padded to L.
- For normal-length input with tail padding enabled, add H to the final regular-grid start. If still below N, append only that one window; do not repeatedly append arbitrary tail windows.
- Retain a candidate only when `valid_length/L≥min_valid_fraction`.
- `padding_mask` is False on the real prefix and True on the padded suffix; raw output `valid_mask` is its inverse.

Four YAML `padding` mappings appear at `module_registry.py:1196-1204`: `none_complete_windows_only`, `right_zero_pad_short_records`, `right_zero_pad_tail`, and `right_zero_pad_short_records_and_tail`. Tail padding supports only a left-start regular grid; engineering does not support padding. Production `min_valid_fraction` is in `(0,1]`, default 1.0. Enabling padding while retaining 1.0 still removes incomplete windows. Low-level `WindowPlan` defaults to 0.0 for API compatibility; that is not the production default.

`extract_window` (201–221) copies `[start,end)` and right-pads with explicit `pad_value`. The raw builder does not simply pad before scaling: it first normalizes the valid prefix and leaves the suffix zero, so padding cannot affect the median.

### 1.9.6 Count and Fraction Caps

Intuition: if a recording is too long, select frames evenly across its whole duration rather than taking only the beginning or choosing attractive segments using labels.

At `windows.py:154-163,188-198`, for M candidates and cap K, indices are `round(j*(M−1)/(K−1)),j=0…K−1`; K=1 selects index 0. Fraction cap r first becomes `K=max(1,ceil(M*r))`. Nothing is removed when M≤K. Order is retained and duplicates checked.

`cap_per_file` is a positive integer or null; `cap_fraction_per_file` is `(0,1]` or null. They are mutually exclusive. Raw defaults/finalcase use 128; engineering defaults to null. The even cap is applied before raw windows with invalid IMU are discarded; discarded windows are not replaced to refill 128.

## 1.10 Model-Window Scaling: Three Pre-Transforms and Three Post-Transforms

### 1.10.1 Shared Location and Defaults

Intuition: change the ruler only for the model's copy so tall and short curves can be compared by shape. The original sheet used to measure actual amplitude remains untouched.

`src/ppg_frailty/representations/raw.py:138-174` combines two x_filter columns, three dynamic_acc axes, and three Gyro axes into `N×8`. Each window first checks valid IMU rows and finite values, then calls `_normalize_dl_window`. The result is transposed to `8×T` and converted to float32; padding is excluded from fitting.

Defaults are at `src/ppg_frailty/normalization.py:73-89`. A frequently misread detail: `raw_ppg` actually controls all eight DL channels, not only PPG; `raw_imu` is the optional subsequent six-axis post-transform.

| Full configuration key (prefix `signal.normalization.`) | Default | Allowed range/form |
|---|---|---|
| raw_ppg | per_window_robust | per_window_robust / per_window_standard_zscore / none |
| raw_imu | none | outer_train_robust / outer_train_mean_std / none |
| iqr_fallback | standard_deviation_then_finite_one | standard_deviation_then_finite_one / median_absolute_deviation_then_finite_one / finite_one |
| clip_after_scale | [-8,8] | null or two finite increasing numbers |
| robust_iqr_divisor | 1.349 | Positive finite number |
| mad_consistency_divisor | 0.6744897501960817 | Positive finite number |
| scale_epsilon | 1e−8 | Positive finite number |
| standard_ddof | 0 | Nonnegative integer; insufficient samples follow degeneracy handling |

### 1.10.2 per_window_robust: Per-Window Median and Interquartile Span

Intuition: move the segment's middle height to zero, then use the spread of its middle half as a ruler. A few extreme spikes have little influence on that ruler. Finally, clip excessive heights/depths to fixed limits.

`raw.py:90-111`, for valid samples `x` of each channel:

1. `m=median(x)`; `q25,q75=percentile(x,[25,75])`.
2. `s=(q75−q25)/1.349`. No epsilon is added to the denominator here.
3. If s is nonfinite or no greater than 1e−8, compute fallback s as in 1.10.5; if still unusable, set s=1.
4. `z=(x−m)/s`; replace nonfinite z with zero. If clip is nonnull, clip elementwise to its bounds, default `[-8,8]`.

Each channel and window is independent; centers/scales are not shared across windows. The same original point in overlapping windows may receive different z values because its windows have different centers/scales. This does not leak labels because only each window's own signal is used, and the result can be stored in the deterministic cache. finalcase uses this option.

### 1.10.3 per_window_standard_zscore: Per-Window Mean and Standard Deviation

Intuition: shift the segment's average height to zero and use the usual spread about that average as the ruler. Extreme points influence this ruler more strongly.

`raw.py:86-89,103-111` computes `m=mean(x)` and `s=sqrt(sum((x−m)²)/(n−ddof))`, default ddof=0. When `n≤ddof`, s becomes NaN and then falls back to 1; other too-small/nonfinite values also become 1. Finally compute `(x−m)/s`, replace nonfinite values with zero, and optionally clip. This option does not call the IQR fallback algorithms.

### 1.10.4 none: No Per-Window Scaling

Intuition: retain the model copy's original numerical heights rather than changing its ruler, while still cutting windows and converting to float32 as required.

`raw.py:83-85` immediately returns a float64 copy. This early return also bypasses clipping, so none does not still clip to ±8. Earlier filtering/calibration has already processed the physical signals; this does not restore original CSV values.

### 1.10.5 Three Fallbacks for a Degenerate Robust Scale

Intuition: when the middle half of a curve lies on one line, the original ruler has zero length. Choose another ruler instead of dividing by zero.

`raw.py:59-73` and 95–107 offer the following three separate choices.

#### 1.10.5.1 standard_deviation_then_finite_one

Intuition: if the middle half has no spread, inspect the usual spread of all points. Lines 71–73 use the channel's standard deviation with configurable ddof. Insufficient samples produce NaN, then lines 103–107 fall back to 1. This is finalcase's default fallback.

#### 1.10.5.2 median_absolute_deviation_then_finite_one

Intuition: measure each point's distance from the middle height, then use the middle of those distances as a new ruler. Lines 68–70 compute `s=median(|x−median(x)|)/0.6744897501960817`; if nonfinite or ≤epsilon, lines 103–107 set it to 1.

#### 1.10.5.3 finite_one

Intuition: do not estimate another ruler; use fixed unit length. Lines 66–67 directly return s=1 for every channel.

The shared fallback to 1 does not make the entire output zero: output remains `x−center`. Only an originally constant curve matching its center becomes all zero.

### 1.10.6 outer_train_robust: Six-Axis Training-Fold Post-Scaling

Intuition: if this additional layer is enabled, pool all available segments from training participants to prepare six common motion-axis rulers. Apply them to held-out people without remeasuring rulers on held-out curves.

`src/ppg_frailty/representations/imu_transform.py:151-240`, `fit_fold_imu_channel_transform`:

1. Lines 164–168 require fitting participants to belong to outer_train and not outer_oof.
2. Lines 174 and 182–184 select training windows and then only per-axis samples within valid masks. Samples repeated in overlapping windows count repeatedly; there is no deduplication or equal-person weighting.
3. Lines 196–210 compute per-axis median center and IQR/1.349 scale, with the same three degeneracy fallbacks; line 213 sets any remaining unusable scale to 1.
4. Store six-dimensional center, scale, valid counts, and fitting participants. `apply_fold_imu_channel_transform` at 243–260 uses `(tensor[:,2:,:]−center)/scale`. The two PPG channels remain unchanged, padding is zeroed, and the float32 output is not clipped again.

The caller at `experiment.py:1734-1747` passes raw windows already processed by 1.10.2/1.10.3/1.10.4, not physical-unit IMU. Enabling this may therefore perform a second scaling. It cannot belong to the deterministic preprocessing cache shared across folds.

### 1.10.7 outer_train_mean_std: Six-Axis Training-Fold Mean/Standard Deviation

The intuition matches the preceding section, but the shared ruler uses the mean height and spread of all training points rather than the middle half.

`imu_transform.py:192-195,213,243-260` uses `center=mean` and `scale=std(ddof)` over the same training/valid-mask scope, falling back to 1 for insufficient or degenerate data. Application matches the preceding section. There is no extra per-participant mean followed by averaging across participants.

### 1.10.8 raw_imu:none: No Six-Axis Post-Transform

Intuition: after preparing the eight curves individually, do not give the six motion curves an additional new ruler.

`experiment.py:1726-1732` skips the fitter entirely and records not applicable; it does not call an object that secretly fits training data. finalcase uses this option, while all eight channels still undergo per-window robust scaling. A direct none call to the low-level fitter has an identity center=0, scale=1 form (`imu_transform.py:188-191`), distinct from the production caller not calling it at all.

## 1.11 Resampling: Distinct Entry Points Are Not Interchangeable

### 1.11.1 finalcase Model Input: Each Valid Window from 400 to 64 Hz

Intuition: redraw the curve with fewer points. First weaken rapid oscillations that a sparse grid could mistake for another rhythm, then sample at the new spacing. Each window is processed separately; this is not thinning the full recording before windowing.

`prepare_configured_dl_input` at `src/ppg_frailty/signal/resample.py:76-151` takes float32 `W×C×T` and boolean `W×T`. Configuration comes from `signal.dl_resampling`; the resolved config supplies the default scheme. finalcase explicitly sets `enabled:true,target_fs_hz:64,method:polyphase_anti_alias,preserve_feature_grid_hz:400`.

1. Line 100 calls `_audited_ratio` (15–20), converting `target/source` to rational u/d with denominator ≤10000 and requiring reconstructed-frequency error ≤1e−12. 400→64 gives 4/25.
2. Lines 102–107 derive valid length v from each mask, which must contain v leading True values followed by False, with v≥2.
3. Lines 109–113 fix output length `T'=round(T*target/source)`, allocating zero float32 output and all-False masks. finalcase maps T=2000 to T'=320.
4. Lines 114–126 call `signal.resample_poly(valid,up=u,down=d,axis=-1,window=('kaiser',5.0),padtype='constant')` on each true prefix. Conceptually this inserts a denser grid, lowpass-shapes it, and retains regularly spaced points. SciPy generates the actual coefficients; the code does not hand-design all taps.
5. `v'=min(T',round(v*target/source))`; copy `min(v',actual_output_length,T')` points and set corresponding mask entries True. The library length and rounded target may differ by at most one; remaining positions stay zero/False.
6. When u=d, reuse the valid prefix without resampling. For `enabled:false` without case_id, `experiment.py:1953-1956` does not call this function at all and requires the target to remain 400 Hz.

The formal range is `0<target≤source=400`, with at least two output points and sufficient rational precision; it is not limited to 100/160/200. PPG peak times, morphology, and engineering features still use 400 Hz. Normalization and resampling generally do not commute: resampling changes each window's quantiles.

### 1.11.2 Named Fixed-Kernel Comparison Resampling

Intuition: change points per second while keeping the model's ruler for neighboring points the same number of samples long. That ruler then spans a different physical time, which is the intended comparison.

Registry and `prepare_fixed_kernel_dl_input` at `src/ppg_frailty/models/time_scale.py:60-67,147-224` name reference 400 Hz/5 seconds, context 10 seconds, 100/160/200 Hz, and dilation2 conditions. Valid prefixes, Kaiser5, constant boundaries, float32, and rounded lengths follow 1.11.1. Differences are fixed case-defined input/output lengths, integer-frequency ratios, and convolution-kernel sample counts that do not scale with frequency.

Routing uses `signal.dl_resampling.case_id`; `experiment.py:1957-1970` permits raw only. Registered conditions do not imply completed training, and finalcase 64 Hz is not one of these named cases: finalcase has no case_id.

### 1.11.3 Standalone One-/Two-Dimensional DL View: Linear Edge Extension

Intuition: make a separate copy at a different point density, temporarily extending each edge along its local direction instead of zero-padding outside the frame.

`resample_dl_view` at `src/ppg_frailty/signal/resample.py:163-194` takes nonempty finite one-/two-dimensional float64 input with required source rate 400 Hz. It calls `resample_poly(...,axis=axis,padtype='line')`, default axis −1. No window is explicitly passed, so the library's default filter window applies; output is float64. This standalone API permits any positive target frequency, without the production DL configuration's ≤400 limit. It is not finalcase's three-dimensional window function; the shared word “resample” does not make them substitutes.

### 1.11.4 External Synchronized Multichannel Resampling

Intuition: move eight or more parallel curves onto the same new time sheet. Every column generates points at shared positions, avoiding misalignment from independently changing column lengths.

`resample_synchronized_channels` at `src/ppg_frailty/signal/resample.py:226-275` takes float64 `N×C` with unique ordered column names. Source/target rates must be positive and finite; target defaults to 400 Hz. It applies the same u/d along axis 0 using `resample_poly(...,padtype='line')`. Expected rows are `ceil(N*u/d)`, with time `arange(N')/target_fs` reset to start at zero. This preserves row synchronization, does not repair pre-existing channel delays, and rejects NaN. The ordinary finalcase 400 Hz CSV route needs no such adapter.

## 1.12 Preprocessing Differences in the Historical Bridge

This section prevents applying current-mainline formulas to retained historical comparisons. finalcase does not enable the bridge. The bridge reproduces a separate protocol rather than aliasing the parameters above.

### 1.12.1 Historical Filtering and Resampling Before Windowing

Intuition: the old route first redraws the whole sheet on a sparser grid, then cuts frames; the new route first cuts frames and changes each grid independently. Different outside-frame content is visible near the edges, so the two processes are not equivalent.

`src/ppg_frailty/legacy_bridge.py:538-634`, `build_legacy_bridge_raw_windows`:

1. Lines 557–562 linearly detrend the whole PPG recording, then apply third-order 0.2–8 Hz forward-backward bandpass. `_filter_sos` is implemented at 421–423.
2. `legacy_filtered_axes` at 563–577 applies third-order forward-backward ACC 20 Hz/Gyro 40 Hz filtering to original axes, without SI conversion, B-bias subtraction, or gravity estimation. The alternative at 578–596 uses prebuilt canonical calibrated IMU while preserving time-row count.
3. Line 599 first converts the assembled eight-channel matrix to float32. If target frequency differs, 603–608 apply `resample_poly` with default edge/filter settings to the full recording along axis 0, then convert to float32 again. Any mask is sampled at nearest source positions `round(target_index*source_fs/target_fs)` (609–614), not lowpass-filtered.
4. Only then do lines 615–619 call `_raw_windows_from_matrix`. Removing float32 roundings or moving resampling after windowing would break historical numerical equivalence.

### 1.12.2 Historical Per-Window Scaling, Caps, and Padding

Intuition: each small frame still receives a new ruler, but the old zero-length safeguard differs, and old padded records mark blank positions usable. This reproduces the old program; it is not a recommendation for new calls.

`_robust_scale_all_channels` at `legacy_bridge.py:426-435` first converts to float32, uses median center and `s=IQR/1.349`, substitutes standard deviation when s≤1e−6, then computes `z=(x−m)/(s+1e−6)`, zeroes nonfinite values, clips to ±8, and returns float32. This differs from the standard mainline, where epsilon is only a test, not added to the denominator, and the final fallback scale is 1.

`_window_starts` at 438–469 windows at the current sample rate using profile seconds/hop and appends a right-aligned final window. Caps use evenly selected `linspace(...).round()` positions, optionally deriving a ceil count from historical retention fraction. `_raw_windows_from_matrix` at 491–518 scales all eight columns per window or only the two PPG columns before later training-fold six-axis post-transforms, according to profile; short records are right-padded only when allowed. However, the returned mask at 517 is entirely True, including historical zero-padded positions. Standard raw False-suffix semantics do not apply.

`build_v2_window_scaled_bridge_raw_windows` at 637–654 takes PPG, dynamic, and Gyro from canonical views but still uses the historical window scaler above. Different bridge profiles may replace only one stage. Follow resolved profile behavior rather than assuming standard-mainline processing because a filename contains v2.

## 1.13 Coverage Checklist and Scope

| Source file | Numerical/workflow modules explained |
|---|---|
| `signal/preprocess.py` | Time grids, short PPG gaps/flatlines, detrending, two registered bandpasses, IMU dispatch, canonical-view construction |
| `signal/motion_imu.py` | Robust mean, B static biases, units, forward-backward ACC/Gyro filtering, five-state orientation, forward-backward 0.3 Hz gravity, retained gravity, magnitudes, jerk |
| `signal/imu.py` | Historical units, six-axis short gaps, stateful one-direction filtering, quaternion geometry, orientation without precalibration, one-direction 0.3 Hz gravity, chunk continuity, comparison metrics |
| `signal/views.py` | Purposes of native/filter/rate/artifact views and identity/nonidentity validity masks |
| `data/windows.py` | Seconds-to-samples, left/right/append-right alignment, short records, tail padding, valid fraction, even count/fraction caps, window copying |
| `normalization.py`, `representations/raw.py` | Three per-window scalings, three degenerate-scale fallbacks, masks, float32 boundaries |
| `representations/imu_transform.py` | Three six-axis post-transforms, training-participant-only fitting, valid-sample selection and application |
| `signal/resample.py`, `models/time_scale.py` | Generic DL windows, named fixed-kernel conditions, standalone DL view, external synchronized resampling |
| `pipeline.py`, `data/qc.py`, `experiment.py`, `module_registry.py` | Actual entry points, B selection, production defaults/accessibility, execution order |
| `legacy_bridge.py` | Historical differences in filtering, order, floating-point precision, scaling formulas, and padding |

This chapter explains source code, not a rerun-based validation report. SciPy filter design/polyphase coefficient generation and NumPy quantile internals are not reimplemented here; results depend on the installed dependency implementations. No algorithm is added or changed, and a function's existence does not imply that every CLI/Dash combination exposes it. The cache chapter explains deterministic keys, stored content, and cross-fold leakage avoidance; later chapters cover quality scores, denoisers, and features.
