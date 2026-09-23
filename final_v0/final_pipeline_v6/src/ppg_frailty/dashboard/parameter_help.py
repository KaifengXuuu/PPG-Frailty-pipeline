"""English UI explanations; these descriptions never resolve or change values.

Scientific meanings follow peaks/resolver.py, signal/prv.py,
representations/raw.py, signal/resample.py and data/windows.py. Signal and
model descriptions live beside this file so that each algorithm has one source
of UI documentation. Unknown user-plan metadata is not given invented effects.
"""
from __future__ import annotations

import re
from typing import Any, Mapping


PARAMETER_HELP = {
    'config_id': "Configuration identifier for provenance. Renaming changes neither calculations nor the output directory.",
    'manifest.path': "Recording-manifest CSV defining available recordings, participants, roles and labels. Changing it changes inputs, not filtering.",
    'splits.path': "Existing repeat/fold training/test participant splits. Changing the table changes evaluation splits; Analyse does not repartition a recording.",
    'roles': "Select B/R/S/W families. Newly selected families expand to all recording IDs; partial YAML selections remain until deselected and reselected. Actual IDs appear below and are exported. classifier_role_families separately restricts classification; no records are generated.",
    'features.enabled_groups': "Select pulse intervals, time/frequency/nonlinear variability, morphology, dual-wavelength relationships and engineering statistics. More groups add features. Raw models still consume signals, with features for preview; feature-vector/fusion routes consume the relevant features.",
    'features.rate_prv_min_duration_s': "Minimum seconds for basic pulse-rate/PPI reporting. Higher excludes more short records; lower accepts shorter segments with less evidence.",
    'features.rate_prv_min_peaks': "Minimum peaks for basic pulse-rate/PPI reporting. Higher requires more pulses; lower covers shorter segments without improving detection accuracy.",
    'features.time_prv_min_duration_s': "Minimum seconds for time-domain interval variability. Higher is stricter; lower accepts shorter records without changing feature formulas.",
    'features.time_prv_min_coverage': "Minimum valid-interval coverage for time-domain PRV. Higher rejects more gaps; lower increases availability but may reduce representativeness.",
    'features.time_prv_min_intervals': "Minimum valid intervals for time-domain PRV. Higher limits small-sample estimates; lower accepts fewer intervals without inventing beats.",
    'features.spectral_prv_min_duration_s': "Minimum seconds for interval spectra. Longer records can capture slower changes. Higher excludes short records; lower does not create reliable low-frequency evidence.",
    'features.spectral_prv_min_coverage': "Minimum coverage for spectral PRV. Higher is stricter; lower accepts more discontinuity and may increase interpolation effects.",
    'features.spectral_prv_min_intervals': "Minimum valid intervals for spectral PRV. Higher requires more beats; lower relaxes eligibility but cannot replace sufficient observation time.",
    'features.tachogram_fs_hz': "Interpolation samples per second for irregular pulse intervals before spectral estimation. Higher increases grid density and computation, not beat information or original PPG sampling rate.",
    'features.sample_entropy.m': "Consecutive intervals matched before checking similarity with one additional interval. Higher compares longer patterns, usually yielding fewer matches and less stable short-record estimates.",
    'features.sample_entropy.r_sd_fraction': "Similarity tolerance as a fraction of interval standard deviation. Higher admits more differing patterns and matches; entropy may decrease, not necessarily monotonically.",
    'features.sample_entropy.min_intervals': "Minimum valid intervals for sample entropy. Higher excludes short samples; lower increases coverage but sparse matches can make results unstable or unavailable.",
    'signal.peak_detector.detector_id': "Peak locator: MSPTDfast uses multiple time scales; Aboy v1/v2 use their adaptive implementations; bipolar prominence is a separate ablation. Switching changes peaks, PPI and features, not just presentation.",
    'signal.peak_detector.min_observation_sec': "Minimum valid observation seconds for accepting peaks. Higher rejects short segments; lower relaxes eligibility without extending the signal.",
    'signal.peak_detector.min_peaks': "Minimum valid detected peaks. Higher is stricter; lower accepts shorter segments without forcing extra peaks.",
    'signal.peak_detector.parameters.minimum_heart_rate_bpm': "Minimum MSPTDfast heart rate in bpm. Lower retains longer search scales for slow pulses; higher reduces scales and computation but can miss slow rhythms.",
    'signal.peak_detector.parameters.target_downsample_hz': "MSPTDfast internal coarse-detection target Hz; actual spacing uses an integer downsampling factor. Higher retains more points at greater cost, without changing the 400 Hz production timebase.",
    'signal.peak_detector.parameters.window_s': "MSPTDfast detection-segment seconds. Higher adds context and local computation; lower adds boundaries and may change merged peaks.",
    'signal.peak_detector.parameters.overlap_fraction': "Overlap between MSPTDfast segments. Higher reduces the hop and repeats boundary observations at extra cost; lower reduces repeats. Original peak-merging rules remain.",
    'representation_mode': "Input form: raw signal windows, feature_vector recording vectors, feature_matrix ordered window features, or fusion signal bags plus recording features. Requires a compatible model and fitted transforms.",
    'signal.dl_resampling.enabled': "Anti-aliased resampling of deep-learning inputs only. Disabled retains 400 Hz; enabled uses target_fs_hz. Feature and peak time grids remain unchanged.",
    'signal.dl_resampling.target_fs_hz': "Deep-learning samples per second. Lower reduces length and cost but removes high frequencies and changes the physical duration of fixed kernels; better classification is not guaranteed.",
    'signal.normalization.raw_ppg': "Window scaling: none, mean/standard deviation, or median/IQR. Despite the raw_ppg name, currently applies to every raw-window channel. Changes model scale, not source signals or morphology input.",
    'signal.normalization.raw_imu': "Additional training-fitted IMU scaling; none disables this layer. Other choices reuse fitted centers/scales from the bundle. Analyse never refits on a new participant.",
    'signal.normalization.clip_after_scale': "Post-scaling clipping [lower, upper]; null disables clipping. Narrower bounds suppress extremes and amplitude differences. The raw-window path skips clipping for raw_ppg=none.",
    'signal.normalization.iqr_fallback': "Fallback when IQR is too small/invalid: standard deviation, MAD, or 1. Determines flat/abnormal-window scaling without changing source data or guaranteeing usability.",
    'signal.normalization.robust_iqr_divisor': "Scale is IQR divided by this value. Without fallback/clipping, higher increases normalized amplitude; lower decreases it.",
    'signal.normalization.mad_consistency_divisor': "For MAD fallback, divide median absolute deviation by this value. Higher reduces scale and increases normalized amplitude. Inactive outside MAD fallback.",
    'signal.normalization.scale_epsilon': "Threshold identifying an excessively small scale. Higher triggers more fallback; lower retains tiny scales that can amplify noise. Not added to every denominator.",
    'signal.normalization.standard_ddof': "Standard-deviation denominator correction: 0 uses n, 1 uses n-1. Choosing 1 slightly increases scale and reduces normalized amplitude, especially for short windows.",
}

WINDOW_HELP = {
    'length_s': "Window duration in seconds. Higher adds context, generally fewer complete windows and longer individual inputs; lower is more local. Feature/model requirements still apply.",
    'hop_s': "Seconds between window starts. Lower adds overlap, computation and adjacent-sample similarity; higher reduces window count without changing signal sampling.",
    'end_alignment': "Fixed-hop windows only, or add a complete end-aligned window. The latter improves tail coverage and may add one window; it is not padding.",
    'padding': "Allow zero-padding short records/tails, still subject to min_valid_fraction. At the default valid fraction of 1, enabling padding alone does not retain padded windows.",
    'min_valid_fraction': "Minimum real valid-sample fraction. Higher excludes more incomplete/padded windows; lower accepts more without treating padding as measurement.",
    'cap_per_file': "Maximum windows per recording; null disables this count cap. Excess candidates are sampled uniformly along the record. Lower saves compute and reduces coverage; incompatible with the fraction cap.",
    'cap_fraction_per_file': "Retained candidate-window fraction per recording; null disables this cap. Lower uniformly samples fewer windows; higher retains more. Incompatible with the count cap.",
}

FIELD_HELP = {
    'training-yaml': "Optional initial YAML; otherwise use function defaults. YAML fills controls, but Analyse always uses their current values without silently restoring YAML.",
    'yaml-case': "Study case whose settings populate the panel. Does not start training or execute the full comparison.",
    'record-ids': "Manifest recordings to analyse. More records add computation and aggregation inputs. A custom CSV table containing valid paths replaces this selection.",
    'preview-record': "Recording shown in stage plots. Model Analyse can still infer on all selected records; changing the plot does not modify files.",
    'participant-id': "Identity for custom CSV inputs. Static B calibration and dynamic records from one person must share it. Not a class label.",
    'preview-start': "Plot start in recording seconds. Higher moves the preview later without cropping full filter/model inputs.",
    'preview-duration': "Plot duration in seconds. Longer views may be subsampled for display only, without changing algorithm inputs or sampling.",
    'calibration-path': "Explicit static B CSV from the same participant for IMU bias/reference estimation. Changes calibration, not classifier weights.",
    'motion-bundle': "Existing motion-evidence JSON and associated weights/thresholds. Analyse only reuses them, never trains on current input. Manual path takes priority.",
    'sqi-artifact': "Training-fitted SQI bounds/participant-IDs JSON, used by routes requiring fitted calibration. Changes score calibration without refitting.",
    'model-mode': "Analyse classifies with existing weights; Train shows training settings and Run. Switching alone starts nothing; Stop remains available.",
    'model-export': "model_config export directory for selecting a case and weights. Does not overwrite controls; select its resolved YAML separately to restore model settings.",
    'model-case': "Trained case in the export. Its representation/architecture must match current inputs.",
    'model-bundle': "Existing learned bundle with weights and applicable training transforms. Manual path takes priority; Analyse does not retrain.",
    'train-target': "Run current controls, comparison queue, full study YAML or a specialized tool. Only Current controls applies the current single-case overrides.",
    'run-name': "Run output directory name; empty uses the existing naming rule. Changes location, not algorithms or splits.",
    'repeat-indices': "Repeat indices, e.g. all or 0,1. More repeats run more predefined experiments, not more epochs in one model.",
    'fold-indices': "Fold indices, e.g. all or 0,1. Subsets save compute but are not complete cross-validation and do not regenerate splits.",
    'job-count': "Parallel experiment processes. Higher may be faster but uses more GPU/system memory and can exhaust resources. Does not add epochs.",
    'cache-mode': "off disables disk cache; read_only reads existing cache; read_write reads/writes. Changes runtime/disk use, never substitutes for fitted transforms or deletes cache.",
    'cache-root': "Preprocessing-cache directory. Changes cache lookup, not the data source; existing cache is not moved/deleted.",
    'resume-path': "Existing pipeline output to resume incomplete work. Empty creates a run; do not select an unrelated experiment.",
    'train-flags': "Refit additionally fits the full cohort and exports weights. Without it, fold weights remain exportable. Dry run only checks/plans, without formal training.",
    'comparison-name': "Name of the current combination in the temporary comparison queue. Add snapshots controls; later changes do not modify queued units.",
    'analysis-runs': "Existing completed/analysable pipeline outputs. Multiple selections enable comparison without training.",
    'analysis-mode': "single, comparison, ablation and test organize existing-result analysis differently without changing weights.",
    'analysis-preset': "Preconfigured report modules. Determines generated figures/tables; explicit figure/table selections override corresponding defaults.",
    'analysis-modules': "Report analysis modules. More modules may add statistics and plots but never alter existing predictions.",
    'analysis-figures': "Explicit figures; empty uses the preset. Missing predictions, labels or histories follow existing report rules.",
    'analysis-tables': "Derived statistical tables; empty uses the preset. These differ from raw prediction tables and never overwrite them.",
    'reference-case': "Statistical reference case. Changes comparison direction/reference, not predictions.",
    'factor-paths': "Comma-separated ablation/comparison configuration paths, e.g. model.model_id. Changes grouping, not executed algorithms.",
    'include-cases': "Include only these comma-separated cases; empty disables this allowlist. Changes the report sample collection.",
    'exclude-cases': "Exclude comma-separated cases from the report without deleting their pipeline outputs.",
    'report-name': "Report subdirectory name; empty follows the input run name rule. Changes paths, not formulas.",
    'statistics-alpha': "Significance level. Lower makes significance stricter but does not decrease p-values. Statistical figures follow their own implemented use of this setting.",
    'calibration-bins': "Probability-calibration bin count. Higher gives finer resolution but fewer samples and potentially noisier bins. Evaluates without recalibrating the model.",
    'report-output': "Existing report directory to preview. Reads different artifacts without regenerating them.",
    'report-figure': "PNG/SVG or HTML artifact to view. Changes display only, not data/models.",
    'report-table': "Report table to preview. In-view sorting/filtering does not modify the file.",
    'tool-operation': "Original CLI export, validation, audit or specialized-study operation. Controls map directly to arguments; selecting alone executes nothing.",
    'tool-plan-yaml': "Load and expand a specialized plan, then use current controls. Execution saves a separate snapshot without overwriting source YAML.",
    'tool-plan-path': "Manual plan-YAML path overrides the dropdown. Clear it to restore dropdown selection; the source is not rewritten.",
}


# These names are argparse destinations, not arbitrary similarly named YAML keys.
TOOL_PARAMETER_HELP = {
    'preset': "Built-in initial pipeline configuration, followed by module/set/unset. Changes declared modules/settings, not just the name. Mutually exclusive with config and manual.",
    'config': "Complete pipeline YAML followed by explicit overrides. Changes execution without modifying the source; not a study/comparison plan or weights file.",
    'manual': "Manual module/set/unset configuration plus config-id, without a named preset as user input. Starts from canonical baseline defaults, not an empty configuration.",
    'module': "Repeatable FAMILY=MODULE_ID, e.g. artifact=nlms_imu_anc. Applies module requirements/defaults before set overrides. Does not run the module immediately.",
    'assignments': "Repeatable --set PATH=YAML. Numbers, booleans, lists and null use YAML parsing. Later assignments win and apply after modules; this changes actual settings.",
    'unset': "Repeatable field paths removed after module/set. Parsers may restore optional defaults or reject missing required fields. Removing a field is not equivalent to disabling a module.",
    'mode_config': "config parses/checks configuration; smoke/full also preflight manifests, fixed splits and constraints. Both resolve all splits; reports summarize the first or all 25. Neither trains or performs a full signal run.",
    'mode_report': "single/comparison/ablation read OOF predictions; test requires independent-test evidence. Changes organization/evidence requirements without converting OOF into independent tests or retraining.",
    'plan': "Tool-supported study YAML defining type, inputs, comparison units and settings. Ordinary/specialized studies have distinct formats. Changes the whole task, not just one model parameter.",
    'study_dir': "Existing study-output directory, not CSV/configuration. Reindex reads artifacts; specialized analysis reads predictions; Special CV complete trains incompletely evaluated candidates according to its plan.",
    'hash_predictions': "Calculate/store prediction-file SHA-256 hashes. Adds file reading/indexing time; disabling changes no predictions and deletes none.",
    'pipeline_output': "Existing pipeline_output/<run> supplying predictions, fold indexes and configuration for Excel/model_config export. Changes export source without training; not the destination for new training.",
    'replace': "Allow existing export replacement. Otherwise duplicate names fail. Excel replaces the workbook; model_config replaces its whole export directory, not appends. Source results remain.",
    'input': "Existing pipeline output for reporting/auditing. execution-audit can inspect failed/interrupted execution without predictions; specialized-report reconstructs the specialized report.",
    'source_root': "Relative-input base for oracle, role-scope and motion/peak studies; not an output directory. Hyperparameter training uses the package root. Does not move data or rewrite files.",
    'case_id': "Existing source-case ID overriding a decision-bias oracle plan. Changes analysed predictions, not artifact names. Not used by role-scope.",
    'prediction_file': "Participant-level OOF Parquet overriding the oracle plan input. Resolves multiple completed attempts. Requires compatible classes, repeats and aggregation; not raw CSV.",
    'step': "Oracle grid spacing over three nonnegative class biases summing to 1; must divide 1, e.g. 0.1. Smaller is denser and costlier. Uses the same labels for selection/scoring: optimistic diagnosis, not leakage-safe evaluation.",
    'upstream_study': "Upstream study containing selected_configuration.json. Inherits selected batch size, learning rate and some regularization by study type, then applies candidate overrides. Not pretrained weights or current-task resume.",
    'device': "Specialized training/full-CV device override, e.g. cpu, cuda or cuda:0. Empty inherits settings. Changes speed/memory, not architecture; small floating-point differences are possible. Not for static peak ablation.",
    'no_denoiser': "Skip only the Stage5-pre PTT denoiser benchmark; motion training and other stages remain. Unchecked runs it with added cost. Not the ordinary denoiser switch or other specialized-study types.",
}


def parameter_help(spec: Mapping[str, Any], config_context: Mapping | None = None) -> str:
    """Describe a widget using its canonical path, including nested plan keys."""
    if spec.get('description'):
        return str(spec['description'])
    from .parameter_help_signal import SIGNAL_HELP, signal_help
    from .parameter_help_model import MODEL_HELP, model_help
    path = str(spec['path']).replace('~1', '.').replace('~0', '~')
    # Plans may wrap canonical overrides under cases.0.overrides.*.
    match = re.search(r'(?:^|\.)(signal|quality|artifact|features|windows|model|training|aggregation|evaluation)\.', path)
    canonical = path[match.start():].lstrip('.') if match else path
    parent, _, tail = canonical.rpartition('.')
    indexed = tail.isdigit()
    lookup = parent if indexed else canonical
    exact = MODEL_HELP.get(canonical) or SIGNAL_HELP.get(canonical) or PARAMETER_HELP.get(canonical)
    description = exact or signal_help(canonical, config_context) or signal_help(lookup, config_context) or model_help(lookup, config_context) or PARAMETER_HELP.get(lookup)
    if not description and lookup.startswith('windows.'):
        description = WINDOW_HELP.get(lookup.rsplit('.', 1)[-1])
    if not description and lookup.startswith('features.spectral_bands_hz.'):
        band = lookup.rsplit('.', 1)[-1].upper()
        description = f'{band} pulse-interval spectral bounds [lower, upper] in Hz. Changing them changes band power and ratios. Wider bands include more frequencies, not necessarily better signals or classification.'
    if description:
        return (f'List item {int(tail) + 1} (code index {tail}). ' if indexed and not exact else '') + description
    leaf = lookup.rsplit('.', 1)[-1]
    aliases = {'run_name': 'run-name', 'output_name': 'report-name', 'jobs': 'job-count',
               'repeats': 'repeat-indices', 'folds': 'fold-indices', 'resume': 'resume-path',
               'bootstrap_resamples': 'evaluation.statistics.bootstrap_replicates',
               'permutation_resamples': 'evaluation.statistics.paired_permutation_replicates',
               'statistics_seed': 'evaluation.statistics.seed', 'alpha': 'statistics-alpha',
               'calibration_bins': 'calibration-bins', 'refit': 'train-flags', 'dry_run': 'train-flags'}
    target = aliases.get(leaf, leaf.replace('_', '-'))
    if target in FIELD_HELP or target in MODEL_HELP:
        return FIELD_HELP.get(target) or MODEL_HELP[target]
    if 'arg' in spec:
        if leaf == 'mode':
            choices = set(spec.get('choices') or ())
            if choices == {'config', 'smoke', 'full'}:
                return TOOL_PARAMETER_HELP['mode_config']
            if choices == {'single', 'comparison', 'ablation', 'test'}:
                return TOOL_PARAMETER_HELP['mode_report']
        elif leaf == 'preset' and '--presets' in spec.get('options', ()):
            return FIELD_HELP['analysis-preset']
        elif leaf == 'module' and '--modules' in spec.get('options', ()):
            return FIELD_HELP['analysis-modules']
        elif leaf in TOOL_PARAMETER_HELP:
            return TOOL_PARAMETER_HELP[leaf]
    if spec.get('help'):
        return f"CLI parameter {spec.get('arg') or path}: {spec['help']}. Original command semantics apply; larger is not universally better."
    return f'Plan field {path}, passed unchanged to the script. Its plan defines its meaning; names alone do not determine algorithmic effects or tuning direction.'


def field_help(identity: str | None) -> str | None:
    """Meaning and operational effect of manually arranged UI controls."""
    return FIELD_HELP.get(identity) if isinstance(identity, str) else None
