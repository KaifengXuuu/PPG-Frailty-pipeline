"""English UI explanations for preprocessing, quality and artifact parameters.

The text follows signal/preprocess.py, signal/motion_imu.py, signal/sqi.py,
quality/window_selection.py and artifacts/*.py; it never changes configuration.
Reducer-specific names are kept separate because shared names can have different
meanings in different algorithms. Numerical vector items reuse their parent help
only when their physical interpretation is identical (for example delay taps).
"""
from __future__ import annotations

from typing import Any, Mapping


SIGNAL_HELP: dict[str, str] = {
    'signal.gap_repair.max_gap_samples':
        "Maximum internal gap repaired by connecting its endpoints, in original samples. Higher allows longer synthetic gaps; lower rejects more records. Leading/trailing gaps are never extrapolated.",
    'signal.ppg_filter.low_hz':
        "PPG bandpass lower cutoff. Higher suppresses more baseline drift but can weaken slow pulses and morphology; lower retains more slow variation.",
    'signal.ppg_filter.high_hz':
        "PPG bandpass upper cutoff. Lower smooths more but may remove peaks/details; higher preserves details and more high-frequency noise.",
    'signal.ppg_filter.order':
        "Butterworth design order for forward-backward zero-phase filtering. Higher sharpens the transition but increases edge transients and numerical sensitivity; lower broadens it. Does not change sampling.",
    'signal.imu.gravity_method':
        "Gravity route: calibrated_roll_pitch_ekf fuses acceleration/gyro after same-participant B calibration; profile_a_lowpass_0p3hz estimates gravity by lowpass after B calibration; sensor_filter_only_no_gravity_removal calibrates/filters but retains gravity. Historical quaternion_error_state_ekf and low_pass_0p3hz do not use that B calibration. Switching changes dynamic acceleration and downstream inputs, not a simple strength level.",
    'signal.imu.calibration_start_s':
        "Start second in same-participant static B data for bias/initial-orientation estimation. Higher skips startup movement but shortens available static data. The entire specified interval must exist and should be motionless.",
    'signal.imu.calibration_stop_s':
        "Static B calibration end in seconds, later than the start. Higher averages more random variation but movement can bias estimates. Out-of-range intervals are not automatically shortened.",
    'signal.imu.sensor_lowpass_acc_hz':
        "Calibrated-route accelerometer lowpass cutoff. Lower suppresses vibration and possibly fast real movements; higher retains faster changes and noise.",
    'signal.imu.sensor_lowpass_gyro_hz':
        "Calibrated-route gyroscope lowpass cutoff. Lower smooths rotation but may lose fast turns; higher sends faster changes and noise into orientation estimation.",
    'signal.imu.sensor_filter_order':
        "Accelerometer/gyroscope Butterworth lowpass order for both calibration and full-record processing. Higher sharpens transitions with more edge effects; lower gives a gentler transition.",
    'signal.imu.gravity_lowpass_hz':
        "Lowpass gravity cutoff: slower variations are treated as gravity and subtracted. Higher tracks faster but may subtract real motion; lower follows more slowly and may leave posture-related gravity.",
    'signal.imu.gravity_filter_order':
        "Order used only by the selected lowpass gravity branch. Higher sharpens gravity/dynamics separation but increases edge sensitivity; lower broadens it. Does not enable EKF.",
    'signal.imu.gravity_mps2':
        "Gravity constant in m/s² for g conversion, static reference and orientation calculations. Changes physical scale and subtraction, not denoising strength; follow data units and experiment definitions.",
    'signal.imu.dynamic_observation_scale':
        "Five-state EKF multiplies acceleration-observation uncertainty by 1 plus this value times relative excess magnitude above gravity. Higher trusts gyro prediction more in motion; lower trusts acceleration. The current formula is not symmetric below gravity.",
    'signal.imu.initial_covariance_diagonal':
        "Initial five-state EKF covariance diagonal: roll, pitch, X/Y/Z gyro bias. Higher distrusts that initial value and permits stronger early correction; lower trusts initialization more.",
    'signal.imu.process_covariance_diagonal_per_second':
        "Five-state EKF process uncertainty per second: roll, pitch, X/Y/Z gyro bias, divided by sampling rate per step. Higher permits faster changes and usually stronger correction; too low can lag real changes.",
    'signal.imu.observation_covariance_diagonal_rad2':
        "Acceleration-derived roll/pitch observation uncertainty in rad². Higher favors gyro prediction; lower follows acceleration more closely but may confuse linear movement with tilt.",
    'artifact.motion_detector_enabled':
        "Use the selected trained motion model and saved thresholds on native windows for routing. Disabled skips learned detection, not IMU preprocessing. Analyse never trains a new detector.",
    'artifact.motion_detector.evidence_path':
        "Motion evidence JSON locating weights, thresholds and input definitions. Select artifacts appropriate to the intended use and split; Analyse does not refit on this participant.",
    'artifact.motion_detector.batch_size':
        "Inference windows per motion-detector batch. Higher usually improves hardware utilization but uses more memory; lower saves space. Does not change model parameter count.",
    'artifact.motion_detector.device':
        "Motion inference device, e.g. cpu, cuda or cuda:0. Changes speed/memory, not weights; backends may differ slightly numerically.",
    'artifact.motion_detector.reuse_scope':
        "matching_outer_fold_or_all29_final uses roster-matched fold detectors in outer evaluation and the full model for final inference. all29_smoke_or_final_only limits the full model to smoke/final use. all29_frozen_in_sample_auxiliary allows an in-sample auxiliary in outer evaluation, not strictly out-of-fold motion detection.",
    'quality.mode':
        "off skips source SQI; diagnostics_only reports without SQI routing; route uses rate/morphology quality for acceptance, denoising or rejection. Missing/flatline checks remain. Motion, window selection and denoising are not automatically disabled; enabled recovery retains post-denoising checks.",
    'quality.flatline_duration_s':
        "Duration in seconds for exactly constant raw PPG, also used in SQI severity. Lower rejects shorter flat segments; higher tolerates more but may admit sensor stalls.",
    'quality.calibrator':
        "fixed_formula_thresholds_v1 uses formula-normalized components; outer_train_empirical_quantiles_v1 additionally scales using outer-training quantiles. Analyse reuses saved empirical calibration, never refits on the current participant. Score scales differ.",
    'quality.calibrator_quantiles':
        "Lower/upper training quantiles defining component mapping to 0–1. Raising the lower bound usually lowers a fixed score; lowering the upper usually raises it. Analyse uses saved bounds; editing these does not refit.",
    'quality.calibrator_quantiles.0':
        "Training lower quantile mapped to 0. Higher raises the lower reference and generally lowers intermediate scores at fixed upper bound. Affects newly fitted calibration only, not saved artifacts.",
    'quality.calibrator_quantiles.1':
        "Training upper quantile mapped to 1. Lower generally raises intermediate scores at fixed lower bound. Must exceed the lower quantile; Analyse does not refit artifacts.",
    'quality.rate_threshold':
        "Q_rate pass threshold, also requiring original valid coverage. Higher tightens rate-feature availability; lower accepts lower-quality data without changing peak detection.",
    'quality.morph_threshold':
        "Q_morph pass threshold, also requiring original valid coverage. Higher restricts morphology inputs; lower retains more potentially distorted waves. Evaluated separately from rate quality.",
    'quality.minimum_coverage':
        "Minimum original valid-sample fraction. Gap filling does not create real observations. Higher is stricter; lower allows more repaired data. High SQI scores cannot replace this requirement.",
    'quality.cardiac_band_hz':
        "Cardiac-energy SQI band [low, high] in Hz. Wider includes more frequencies and possibly noise; narrower can miss real rates. Does not change PPG bandpass filtering.",
    'quality.cardiac_band_hz.0':
        "SQI cardiac-band lower Hz. Lower includes slower cycles/interference; higher excludes them. A scoring band, not the PPG filter cutoff.",
    'quality.cardiac_band_hz.1':
        "SQI cardiac-band upper Hz. Higher includes faster cycles/noise; lower excludes them. Must exceed the lower bound; does not refilter.",
    'quality.spectral_analysis_band_hz':
        "SQI total-energy/normalized-spectral-entropy band in Hz. Widening changes the cardiac-energy denominator and entropy bins, possibly adding interference; narrowing can hide interference. Score changes do not necessarily mean improved signals.",
    'quality.spectral_analysis_band_hz.0':
        "SQI total-energy/entropy lower boundary. Lower includes slower components; higher excludes them. Does not modify filtered waveforms.",
    'quality.spectral_analysis_band_hz.1':
        "SQI total-energy/entropy upper boundary. Higher includes faster components; lower excludes them. Interpret with the cardiac band; a score change is not denoising.",
    'quality.peak_density_bpm_range':
        "Plausible detected peaks per minute: full component score inside, distance-based decay outside. Wider tolerates more densities; narrower is stricter. Does not move peaks; the lower bound also scales outside-range penalties.",
    'quality.peak_density_bpm_range.0':
        "Lower full-score peak density in peaks/min. Higher excludes slower densities; lower admits them. Also sets the outside-range decay scale, so score changes need not be monotonic.",
    'quality.peak_density_bpm_range.1':
        "Upper full-score peak density in peaks/min. Higher tolerates faster/excess peaks; lower penalizes dense detections sooner. Does not set minimum peak spacing.",
    'quality.ppi_range_s':
        "Seconds used for plausible PPI fractions and periodicity searches. Wider admits more intervals, including missed/extra peaks; narrower is stricter. Does not replace detector interval rules.",
    'quality.ppi_range_s.0':
        "Lower plausible PPI/autocorrelation lag in seconds. Lower admits faster rhythms; higher excludes them. Changes subsequent quality assessment, not the detector.",
    'quality.ppi_range_s.1':
        "Upper plausible PPI/autocorrelation lag in seconds. Higher admits slower rhythms and possibly missed-beat intervals; lower excludes them.",
    'quality.ppi_stability_min_intervals':
        "Minimum valid intervals for variability scoring. Higher requires more evidence and excludes short segments; lower covers shorter segments with less stable estimates.",
    'quality.welch_max_nperseg':
        "Maximum Welch segment samples, limited by input length. Higher improves frequency resolution but usually reduces averaging segments; lower does the reverse. Sampling rate is unchanged.",
    'quality.template_min_peaks':
        "Minimum detected peaks before template similarity is attempted. Higher excludes more short segments; lower broadens coverage. Complete usable beats have a separate threshold.",
    'quality.template_min_beats':
        "Minimum complete valid beats for template similarity; incomplete edge beats do not count. Higher requires more repeated shapes; lower permits fewer without guaranteeing representativeness.",
    'quality.template_resample_points':
        "Uniform interpolation points per beat before median-template comparison. Higher refines the comparison grid at extra cost; lower coarsens it. Creates no measurements and changes no source sampling rate.",
    'quality.component_normalization.template_half_width_s':
        "Seconds on each side of a peak for template comparison. Higher includes more shape but can cross neighboring beats or lose boundary segments; lower compares only near-peak shape.",
    'quality.component_normalization.cardiac_concentration_reference':
        "Reference dividing cardiac-band energy fraction into a score clipped to 0–1. Higher generally lowers scores for fixed data and demands stronger concentration; lower reaches full score more easily.",
    'quality.component_normalization.autocorrelation_reference':
        "Reference dividing autocorrelation strength into a quality score. Higher is stricter; lower reaches full score more easily. Does not change searched lags.",
    'quality.component_normalization.ppi_cv_scale':
        "PPI variability tolerance in exp(-interval coefficient of variation / value). Higher tolerates irregularity; lower penalizes more. Does not smooth or correct intervals.",
    'quality.component_normalization.motion_rms_scale':
        "Motion tolerance in exp(-dynamic acceleration RMS / value). Higher reduces motion penalties; lower increases them. Changes no IMU waveform or detector weights.",
    'quality.component_normalization.nonflat_std_threshold':
        "PPG standard-deviation threshold for a nonflat score. Higher treats weak variation as near-flat; lower is more tolerant. Distinct from exactly constant-run duration checks.",
    'quality.component_normalization.clipping_fraction_reference':
        "Repeated raw minimum/maximum fraction at which the clipping heuristic reaches zero. Higher tolerates repeated extremes; lower is more sensitive. Not proof that ADC voltage limits were reached.",
    'quality.component_normalization.saturation_fraction_reference':
        "Reference for penalizing saturation fraction when explicit evidence exists. Higher is more tolerant; lower penalizes faster. Without a measured saturation fraction the component is unavailable, not inferred.",
    'quality.component_normalization.morph_skewness_scale':
        "Skewness tolerance in exp(-absolute skewness / value). Higher tolerates asymmetry; lower favors symmetry. Does not change the waveform.",
    'quality.component_normalization.morph_kurtosis_center':
        "Kurtosis reference center; deviation lowers the morphology score. Higher favors sharper/heavier-tailed distributions, lower flatter ones. Not a monotonic strictness or quality control.",
    'quality.component_normalization.morph_kurtosis_scale':
        "Tolerance for kurtosis deviation, scored as exp(-deviation / value). Higher weakens penalties; lower strengthens them. A separate parameter sets the center.",
    'quality.component_normalization.component_pass_threshold':
        "Display-only PASS/FAIL threshold for individual SQI components. Higher shows more failures. Continuous weighted scores still determine Q_rate/Q_morph, so this does not directly change totals or their thresholds.",
    'quality.window_selection.policy':
        "none retains candidates; legacy_per_file_top_fraction ranks cardiac/motion quality within each recording. Never ranks across participants or uses class labels. Subsequent settings determine counts and partitions.",
    'quality.window_selection.keep_fraction':
        "Top-quality fraction retained per recording, rounded up with at least one window. Higher adds diversity and potentially poorer windows; lower is stricter with fewer samples. Active only for fractional selection.",
    'quality.window_selection.application_scope':
        "outer_train_only filters training windows; all_partitions also filters test/inference. legacy_train_and_aggregation filters training but saves all test/inference predictions, selecting only for aggregation. Determines compared prediction units.",
    'artifact.denoiser_enabled':
        "Permit non-identity denoiser recovery where routing requires it; acceptance still follows existing quality rules. Processes the full record before selecting windows. Disabled skips recovery without changing filters; identity performs no removal.",
    'artifact.reducer':
        "Denoiser: identity passes through; NLMS cancels using IMU; spectral_mask suppresses motion spectra; PCA/ICA/NMF separate components; SSA/EMD/CEEMD reconstruct decompositions; DWT retains fixed wavelet approximation. These are distinct algorithms. Non-identity outputs serve eligible rate branches only, without claiming morphology preservation.",
}


# The five-state order is fixed by calibrated_roll_pitch_ekf, not by UI labels.
for _index, _state in enumerate(('roll', 'pitch', 'X-axis gyro bias', 'Y-axis gyro bias', 'Z-axis gyro bias')):
    SIGNAL_HELP[f'signal.imu.initial_covariance_diagonal.{_index}'] = (
        f"Five-state EKF initial uncertainty for {_state}. Higher trusts the initial value less and permits stronger early correction; "
        "lower trusts it more. This does not directly add signal noise.")
    SIGNAL_HELP[f'signal.imu.process_covariance_diagonal_per_second.{_index}'] = (
        f"Five-state EKF process uncertainty added per second for {_state}, divided by sampling rate per step. "
        "Higher allows faster changes and generally stronger observation correction; lower assumes stability but can lag real changes.")
for _index, _angle in enumerate(('roll', 'pitch')):
    SIGNAL_HELP[f'signal.imu.observation_covariance_diagonal_rad2.{_index}'] = (
        f"Acceleration-derived {_angle} observation uncertainty in rad². Higher trusts this observation less; "
        "lower follows it more closely but is more susceptible to linear motion.")


_COMPONENT_MEANINGS = {
    'cardiac_concentration': "energy concentration in the cardiac band",
    'autocorrelation_periodicity': "waveform repetition at plausible beat intervals",
    'normalized_spectral_entropy': "spectral concentration rather than dispersion; lower entropy gives a higher component",
    'peak_density_bpm': "detected peaks per minute within the specified range",
    'ppi_physiological_fraction': "fraction of pulse intervals in the plausible range",
    'ppi_stability': "stability of relative pulse-interval variation",
    'red_ir_agreement': "absolute RED/IR waveform correlation",
    'motion_energy_rms': "dynamic-acceleration penalty; less motion gives a higher component",
    'nonflat_scale': "waveform standard deviation above the near-flat threshold",
    'source_coverage': "coverage of original valid measurements",
    'flatline': "duration penalty for exactly constant segments",
    'clipping': "clipping heuristic penalizing repeated minima/maxima",
    'long_gap': "whether the longest original gap exceeds the permitted length",
    'saturation': "saturation-fraction penalty when explicit evidence exists",
    'template_correlation': "complete-beat shape similarity to the median template",
    'skewness': "waveform-distribution skewness penalty",
    'pearson_kurtosis': "kurtosis deviation from the reference center",
}
_WEIGHT_COMPONENTS = {
    'rate': ('cardiac_concentration', 'autocorrelation_periodicity', 'normalized_spectral_entropy',
             'peak_density_bpm', 'ppi_physiological_fraction', 'ppi_stability', 'red_ir_agreement',
             'motion_energy_rms', 'nonflat_scale', 'source_coverage', 'flatline', 'clipping', 'long_gap', 'saturation'),
    'morph': ('cardiac_concentration', 'red_ir_agreement', 'nonflat_scale', 'source_coverage',
              'flatline', 'clipping', 'long_gap', 'saturation', 'template_correlation', 'skewness', 'pearson_kurtosis'),
}
for _endpoint, _components in _WEIGHT_COMPONENTS.items():
    for _component in _components:
        SIGNAL_HELP[f'quality.{_endpoint}_component_weights.{_component}'] = (
            f"In Q_{_endpoint}, this weight represents {_COMPONENT_MEANINGS[_component]}. "
            "Higher increases relative influence, lower decreases it, and 0 excludes it. Totals renormalize over available positive-weight components; "
            "a larger weight need not raise the score or make a missing component available.")


_REFERENCE_HELP = (
    "imu_axes6_reference_v2 uses six dynamic-acceleration/gyro axes; augmentation adds acceleration, angular-speed and jerk magnitudes. "
    "Reference columns are standardized and constants removed. Derived quantities add cues and redundancy without guaranteeing improvement.")
_EMD_HELP = {
    'max_imfs': "Maximum oscillatory components from fast to slow, limited by remaining extrema. Higher separates more slow structure at extra cost; lower leaves residual sooner. EMD reconstruction discards that residual, changing the result.",
    'max_sift': "Maximum envelope-mean subtraction iterations per component. Higher allows more sifting at extra cost; lower may leave unstable components. Convergence can stop earlier.",
    'sd_threshold': "Stop when relative squared change between sifts is below this threshold. Higher stops earlier; lower requires stability and more computation, not necessarily a truer waveform.",
}
ARTIFACT_HELP: dict[str, dict[str, str]] = {
    'identity': {},
    'dwt_a2_legacy': {},  # The historical db4/level-2 reconstruction has no tunable controls.
    'emd_sifting_rate_only': dict(_EMD_HELP),
    'ceemd_lite_nlms_legacy': {
        **_EMD_HELP,
        'max_imfs': "Maximum components per noise-added decomposition, potentially stopped by insufficient extrema. Higher includes more slow structure for motion/cardiac classification at extra cost; lower leaves more residual, handled by separate legacy CEEMD rules.",
        'pairs': "Positive/negative noise pairs decomposed before averaging matching components. Higher can better cancel perturbations but requires two EMD runs per pair; lower is faster with less averaging.",
        'noise_ratio': "Added-noise standard deviation relative to source standard deviation. Higher perturbs decomposition more and may separate mixed oscillations or leave randomness; lower approaches ordinary EMD.",
        'random_seed': "Seed for paired CEEMD noise. Fixed seeds aid same-environment reproducibility; larger numbers do not mean better quality.",
        'protect_bandwidth_hz': "Hz tolerance protecting estimated heart rate and specified harmonics. Wider protects more components and possibly noise; narrower risks removing cardiac components.",
        'protect_harmonics': "Number of integer heart-rate multiples protected; 1 protects only the fundamental. Higher covers more harmonics and nearby motion; lower protects fewer.",
        'low_motion_hz': "Unprotected components at or below this dominant Hz become motion references. Higher labels more slow components as motion; lower fewer. A classification boundary, not preprocessing cutoff.",
        'high_motion_hz': "Unprotected components at or above this dominant Hz become motion references. Lower includes more fast components; higher fewer. Intermediate bands also use cardiac-waveform correlation.",
        'nlms_length': "Historical reference samples for NLMS using PPG-derived motion, not external IMU. Higher fits longer lags with more coefficients/cost; lower is more local.",
        'nlms_mu': "Legacy residual-driven NLMS coefficient step. Higher adapts faster but can oscillate or over-cancel; lower is slower. Does not directly scale output.",
        'nlms_leak': "Each legacy NLMS step multiplies old coefficients by 1 minus this value before updating. Higher forgets faster and limits accumulation; too much weakens persistent cancellation.",
    },
    'nlms_imu_anc': {
        'imu_reference_profile': _REFERENCE_HELP,
        'taps_per_delay': "Consecutive reference samples after each starting delay; duplicate tap positions are merged. Higher covers longer history with more coefficients/cost and invalid startup; lower may miss longer lags.",
        'delay_taps': "Reference starting delays in samples, each expanded by taps_per_delay. Higher looks farther back; extra entries add lag coverage but duplicates merge. Maximum delay lengthens startup without available history.",
        'step_size': "Residual/reference-energy-normalized NLMS step, strictly between 0 and 2. Higher tracks faster but risks instability or cardiac removal; lower is steadier but slower.",
        'epsilon': "Positive constant in the reference-energy denominator, preventing excessive weak-reference updates. Higher suppresses such updates; lower is more sensitive. Not added to output.",
        'leakage': "On permitted updates, multiply NLMS coefficients by 1 minus this value. Higher forgets faster; lower retains longer. No leakage is applied when the reference update gate is not met.",
        'update_gate_reference_rms': "Minimum standardized reference RMS for NLMS coefficient updates. Higher updates less often; lower more often. Below threshold, existing coefficients still predict/subtract interference.",
    },
    'pca_bss': {
        'imu_reference_profile': _REFERENCE_HELP + " In PCA this only affects motion-correlation penalties for candidate PPG components; it does not add PCA input channels.",
    },
    'fastica_bss': {
        'imu_reference_profile': _REFERENCE_HELP + " In ICA this only affects motion-correlation penalties for candidate PPG components; it does not add ICA input channels.",
        'max_iter': "FastICA maximum iterations; convergence can stop earlier. Higher permits more solving at extra cost; lower may not converge. Unconverged results are not reported as successful.",
        'tolerance': "FastICA inter-iteration convergence tolerance. Higher usually stops sooner with coarser solutions; lower is stricter and may hit the cap. Not a tolerated PPG-noise amplitude.",
        'random_state': "FastICA initialization seed. Fixed seeds aid reproduction for identical inputs/environment; numeric size does not mean stronger/weaker separation.",
    },
    'nmf_bss': {
        'nmf_rank': "Nonnegative bases for dual-channel spectrogram magnitude, limited by matrix dimensions. Higher is more flexible/costly and can split a cardiac source. Only one cardiac-concentration-selected basis is reconstructed.",
        'nperseg': "NMF input STFT window samples, limited by record length. Higher improves frequency resolution but coarsens time localization; lower reverses that. Sampling rate determines duration.",
        'overlap_fraction': "NMF time-frame overlap. Higher gives denser frames at greater cost; lower sparser. Adds no measured samples; overlap samples derive from window length.",
        'max_iter': "Maximum NMF optimization iterations with possible early stopping. Higher permits more objective reduction at added cost; lower can stop at a coarser decomposition. Does not add bases.",
        'tolerance': "NMF convergence tolerance. Higher generally stops earlier with looser accuracy; lower is stricter/costlier. Cardiac-band selection rules remain unchanged.",
        'random_state': "NMF random state for stochastic operations; initialization currently uses NNDSVDA. Fixed seeds aid reproduction; larger values do not increase noise strength. Actual effect depends on the solver path.",
    },
    'spectral_mask': {
        'imu_reference_profile': _REFERENCE_HELP,
        'stft_window_s': "STFT window seconds. Higher improves frequency resolution but slows tracking and expands unusable neighborhoods around missing IMU; lower is more local with coarser frequency resolution.",
        'stft_hop_s': "Seconds between STFT window starts, no greater than window duration. Lower adds overlap/computation and denser frames; higher is sparser. Does not resample the source.",
        'imu_mask_quantile': "Frequency-axis quantile normalizing each IMU spectral frame. Higher gives a nondecreasing denominator and weaker normalized motion, usually lighter suppression; lower usually stronger, not stricter at higher quantiles.",
        'mask_strength': "Maximum in-band motion suppression, with gain floor 1 minus this value. Higher suppresses more but can damage overlapping cardiac content. At 0, only in-band suppression stops; out-of-band frequencies remain zeroed.",
        'preserve_band_hz': "Reconstruction frequency bounds in Hz; outside bins are zero. Wider retains signal and interference; narrower can lose cardiac components. In-band content still faces the motion mask.",
        'preserve_band_hz.0': "Lower retained spectral-mask frequency; lower bins are zeroed. Higher removes slower changes; lower retains slower rhythms/interference. Distinct from preprocessing low cutoff.",
        'preserve_band_hz.1': "Upper retained spectral-mask frequency; higher bins are zeroed. Lower removes fast detail; higher retains harmonics/noise. In-band masking still applies.",
    },
    'ssa_decomposition': {
        'embedding_samples': "Samples per SSA delay-matrix column, capped at one third of record length. Higher captures longer cycles but substantially increases SVD memory/cost; lower may not separate slow structure.",
        'max_components': "Maximum leading singular components inspected before cardiac-energy selection. Higher admits weaker detail/noise candidates; lower inspects stronger ones only. Not a guaranteed retained count.",
        'cardiac_low_hz': "SSA cardiac-concentration lower Hz. Lower accepts slower oscillations/baseline; higher excludes them. Changes component selection, not PPG bandpass.",
        'cardiac_high_hz': "SSA cardiac-concentration upper Hz. Higher accepts faster oscillations/noise; lower excludes them. Only qualifying components are reconstructed.",
        'minimum_cardiac_concentration': "Minimum cardiac-band power divided by fixed analysis-band power for an SSA component. Higher retains fewer; lower is more tolerant. If none qualify, no result is returned; there is no automatic best-component fallback.",
    },
}


_SHARED_ARTIFACT_HELP = {
    'imu_reference_profile': _REFERENCE_HELP,
    'max_imfs': "Maximum EMD/CEEMD oscillatory components, potentially stopped by insufficient extrema. Higher includes more slow structure at extra cost; lower leaves more residual, handled according to the selected denoiser.",
    'max_iter': "ICA/NMF iteration cap; solver-specific convergence can stop earlier. Higher allows more solving but can be slower; lower may not converge. Does not add signal length or components.",
    'tolerance': "ICA/NMF convergence tolerance with solver-specific stopping criteria. Higher generally stops earlier; lower is stricter/costlier. Not a permitted signal-noise amplitude.",
    'random_state': "Selected ICA/NMF random state for stochastic initialization/solving. Fixed seeds aid same-environment reproduction; numeric size is not noise strength, and effects depend on the solver.",
}


def signal_help(path: str, config_context: Mapping[str, Any] | None = None) -> str | None:
    """Return help without reading files, resolving config or mutating UI state."""
    if path.startswith('artifact.parameters.'):
        key = path.removeprefix('artifact.parameters.')
        parent, _, index = key.rpartition('.')
        lookup = parent if index.isdigit() else key
        reducer = (config_context or {}).get('artifact', {}).get('reducer')
        if reducer is not None:
            help_by_parameter = ARTIFACT_HELP.get(str(reducer), {})
            return help_by_parameter.get(key) or help_by_parameter.get(lookup)
        # A context-free caller can still explain shared or unique leaves. UI
        # callers supply context so ICA/NMF and IMU-use differences stay exact.
        if lookup in _SHARED_ARTIFACT_HELP:
            return _SHARED_ARTIFACT_HELP[lookup]
        for help_by_parameter in ARTIFACT_HELP.values():
            if key in help_by_parameter:
                return help_by_parameter[key]
            if lookup in help_by_parameter:
                return help_by_parameter[lookup]
        return None
    return SIGNAL_HELP.get(path)
