"""English UI help checked against the model factories and training consumers.

Display metadata only: no defaults, validation, computation or model loading.
List items reuse their parent's explanation; Adam's two betas have distinct help.
"""
from __future__ import annotations

from typing import Mapping


_MODEL = {
    'model_id': "Classifier architecture: convolutions learn from signals, traditional models consume vectors, ShapeFormer discovers discriminative local shapes, and fusion combines signals/file features. Switching changes parameters and required inputs; incompatible old weights cannot be reused.",
    'input_channel_order': "Actual input channels and order. Adding/removing changes information and width; reordering changes which signal each weight receives. Must match training channel semantics, not just display labels.",
    'input_channels': "Input channel count determined by selected signal/feature width. Changes first-layer structure and information sources; do not edit independently of actual channels.",
    'n_classes': "Class-probability columns in project class order. Changing the count changes the head and invalidates old head weights. This project uses three classes.",
    'seed_policy': "Seed source: outer_repeat follows repeats, fixed_explicit uses the explicit training seed, member_roster uses ensemble seeds. Historical aliases retain explicit fixed-seed/five-member restrictions. Changes randomness without monotonic accuracy benefits.",
    'seed': "Explicit model-construction seed. Changes initialization/sampling, not strength. Cross-validation also resolves the actual seed through seed_policy.",
    'member_seeds': "Ordered seeds for independently trained ensemble members whose probabilities are averaged. Changing seeds changes randomness; more members cost more training/inference without guaranteed accuracy gains.",
    'ensemble_size': "Ensemble size derived from member seeds, not an independent switch. More members cost more; editing this metadata alone cannot convert single-model and ensemble outputs.",
    'variant': "Inception small/full base variant. Unspecified widths/depths inherit its defaults; explicit fields are not overwritten by renaming the variant. Current parameters define the actual architecture.",
    'kernel_sizes': "Convolution width in samples: CompactCNN lists successive stages; Inception lists parallel branches. Larger covers longer shapes at extra cost; smaller targets details. Does not resample.",
    'dilations': "CompactCNN sampling gaps within stage kernels. Larger expands receptive span without extra parameters but samples more sparsely; smaller is denser/local. Ordered by the three stages.",
    'dilation': "Inception kernel sampling gap. Larger expands branch time span sparsely; smaller emphasizes neighbors. Does not resample or directly reduce output time points.",
    'pool_sizes': "CompactCNN first-two-stage max-pool widths and strides. Larger shortens time/cost but may lose brief detail; smaller retains more positions with greater downstream cost.",
    'pool_size': "Inception parallel max-pool neighborhood, with stride fixed at 1. Larger uses broader neighborhoods; unlike CompactCNN pool_sizes, does not shorten time by this factor.",
    'stage_channels': "Output channels of three CompactCNN stages. Larger expresses more shapes with more parameters/memory/overfitting risk; smaller is lighter but less expressive.",
    'stage_dropouts': "Training dropout after the first two CompactCNN pools. Higher reduces reliance on individual responses but can impair learning. Disabled for prediction; does not delete raw signal segments.",
    'dropout': "Internal training dropout: mainly CompactCNN/Inception heads, also ShapeFormer attention and fusion layers. Higher regularizes more but can underfit; prediction disables dropout. Editing does not retrain weights.",
    'out_channels': "Channels per Inception branch; concatenated width equals this times convolution-branch count plus one. Higher adds capacity/cost; smaller may miss subtle differences.",
    'bottleneck_channels': "Inception 1-point channel-mixing width before longer convolutions. Smaller compresses cost/information; larger preserves more mixtures at extra cost. Sample count stays unchanged.",
    'depth': "Stacked Inception module count. Higher adds depth/receptive span/cost and may complicate training; lower is lighter. Not a repeat count.",
    'residual_interval': "Inception modules between residual additions. Smaller makes shortcuts more frequent; larger spans more layers. Above depth, no complete interval exists; effects are not monotonic.",
    'feature_hidden_dim': "Fusion file-feature embedding width. Higher allows more combinations and parameters; lower is compact. This file branch enters only once per file.",
    'fusion_hidden_dim': "Width after joining pooled signal and file-feature embeddings. Higher adds joint capacity/cost; lower reduces it. Not a direct branch weight.",
    'pooling': "Fusion window pooling: mean weights valid windows equally; attention learns weights. Attention adds parameters and does not automatically use SQI. Requires compatible learned weights.",
    'logistic_c': "Inverse logistic-regression regularization C. Higher weakens constraints and may overfit; lower constrains coefficients and can underfit. Not a learning rate.",
    'logistic_max_iter': "Logistic-regression solver iteration cap. Higher gives unconverged solving more time; early convergence can make extra allowance irrelevant. Not CV repeats.",
    'logistic_solver': "Numerical logistic-regression solver. lbfgs/newton and sag/saga differ in memory/speed/convergence and version/multiclass compatibility. Equal objectives do not guarantee identical predictions.",
    'svm_c': "RBF SVM training-error penalty C. Higher fits training data more closely with potentially complex boundaries; lower allows errors with stronger regularization. Interacts with gamma; higher is not always better.",
    'svm_gamma': "RBF locality in exp(-gamma times squared distance). Higher is more local; lower broader. scale uses dimensions/variance; auto is inverse dimension; positive numeric values are also accepted.",
    'svm_probability': "The current SVM branch requires probabilities for window/file/participant aggregation. Probability calibration is part of fitting; disabling cannot directly replace the probability workflow.",
    'extra_trees_n_estimators': "Extra Trees count, with probabilities averaged. More trees generally reduce finite-ensemble variability but cost time/model size/inference. Metrics need not improve monotonically.",
    'extra_trees_max_features': "Features considered per tree split: integer count, fractional proportion, sqrt/log2 of width, or null for all. Higher compares more; lower increases tree randomness and changes bias/correlation.",
    'extra_trees_min_samples_leaf': "Minimum leaf training samples: integer count or fraction of total. Higher prevents tiny leaves and smooths predictions but may underfit; lower gives finer groups and can fit noise.",
    'extra_trees_n_jobs': "Tree parallel workers: -1 uses available processors; positive specifies count. More may speed up with more CPU/memory; oversubscription can hurt. Tree count/formulas are unchanged.",
    'input_fs_hz': "Declared actual model sampling rate for converting samples to seconds and checking resampling. Editing alone does not resample; at fixed length, higher means shorter physical duration.",
    'sequence_length_samples': "Expected ShapeFormer input length after resampling. Larger means longer inputs/more processing but must match actual windows. Not an arbitrary padding/cropping switch.",
    'num_pip_ratio': "ShapeFormer key-turning-point fraction: floor(length times ratio), at least 5 and at most window length. Higher generally creates finer shapes/more search; lower is coarser.",
    'shapelets_per_class': "Representative shapes retained per class. Higher adds comparison/discovery/training/inference cost and possible redundancy; lower is compact but can omit useful shapes.",
    'max_discovery_windows': "Training-fold window cap for ShapeFormer discovery. Higher broadens training coverage at extra cost; lower narrows candidates. Not a test-window cap or permission to use outer-test labels.",
    'discovery_balance': "Discovery sampling: participant_file_balanced balances classes/participants/files; class_window_balanced allocates by class; legacy_class_window_balanced serves historical variants. Changes candidates, not outer splits.",
    'position_search_neighbourhood_samples': "Extra sample positions searched near channel-specific shape locations. Higher allows larger timing shifts and more comparisons; lower is stricter. Never searches other participants or test labels.",
    'shapelet_search_window_samples': "Historical ShapeFormer match range around the original location, in samples. Larger finds farther shifts at greater cost; smaller is local. Distinct from the unused historical local-kernel-width marker.",
    'shapelet_length_samples': "Fixed-template length in samples. Larger compares longer shapes at more distance cost; smaller targets brief patterns. Must fit inputs; input_fs_hz determines seconds.",
    'discovery_stride_samples': "Candidate-start stride for fixed-length discovery; fixed-effect variants also use it for sliding discovery distances. Larger is sparser/faster; smaller denser/slower and may change selected shapes.",
    'max_candidates_per_class': "Maximum fixed-effect ShapeFormer candidates per class, seed-sampled above the cap. Higher evaluates more at extra cost; lower may miss discriminative templates. Not the retained-template count.",
    'candidates_per_class_channel': "Historical high-scoring candidates per class/channel before cross-channel ranking. Higher expands the pool; lower filters more. shapelets_per_class determines final count.",
    'local_kernel_width_samples': "ShapeFormer local-convolution width in consecutive samples. Larger changes convolution structure and spans broader morphology; smaller targets details. Distinct from template length/search range.",
    'local_embedding_channels': "ShapeFormer local-convolution embedding width. Higher adds pattern combinations, parameters and attention cost; lower is compact. Must meet multi-head dimension requirements.",
    'shape_embedding_channels': "Shape-matching projection width. Higher represents template/selected-segment differences more richly at extra cost; lower is compact. Does not discover more templates.",
    'hidden_channels': "Patch embedding width in scalar-distance/fixed-effect ShapeFormer. Higher holds more patterns with more parameters/memory; lower is lighter. Must divide evenly by attention_heads.",
    'patch_size_samples': "Nonoverlapping patch samples in scalar-distance/fixed-effect ShapeFormer; convolution kernel and stride both equal this. Larger gives fewer attention tokens and lower cost but coarser time detail; smaller the reverse.",
    'attention_heads': "Parallel attention groups within the representation. At fixed total width, more heads have smaller per-head width, not proportionally more capacity. Requires divisibility and changes information mixing.",
    'attention_layers': "Successive attention-encoder layers in scalar-distance/fixed-effect ShapeFormer. More adds depth/cost/overfitting risk; fewer is lighter. Not head count.",
    'attention_feedforward_channels': "Intermediate per-position feedforward width in attention. Higher adds combinations/parameters/cost; lower reduces capacity. Does not change sampling or window count.",
    'attention_query_chunk_size': "OSD query positions per attention chunk. Larger reduces chunks but needs more temporary memory; smaller lowers peak memory, possibly slower. Every query still compares against the complete key/value sequence.",
    'distance_position_chunk_size': "Candidate positions per shape-distance chunk. Larger adds parallelism/temporary memory; smaller adds batches. Does not remove search positions or templates.",
    'complexity_norm': "Adds 1/value to complexity measures to stabilize flat-segment ratios. Higher reduces smoothing and may emphasize complexity differences; lower smooths more. Does not divide the signal directly.",
    'max_complexity_ratio': "Maximum multiplier from pairwise complexity differences in shape distance. Higher allows stronger roughness penalties; lower limits amplification. At 1, removes this amplification, not base distance.",
}

_TRAINING = {
    'epoch_rule': "fixed_epoch trains a fixed duration; inner_grouped_selection selects epochs using only outer-training participants, then retrains on all outer-training data. Changes cost/workflow without using outer-test labels.",
    'fixed_epochs': "Training-data passes for fixed-epoch deep training. More gives more updates/cost and possible overfitting; fewer can undertrain. Traditional estimators do not use this deep epoch loop.",
    'maximum_inner_epochs': "Inner-selection epoch cap. Higher permits later optima with greater potential cost; lower stops earlier. Fixed-epoch mode uses 0 and starts no inner training.",
    'inner_patience': "Inner participant-balanced-accuracy patience without strict improvement. Higher waits through plateaus at extra cost; lower stops earlier. Unused in fixed-epoch mode.",
    'inner_grouped_folds': "Stratified groups among outer-training participants; one selected by repeat/fold is inner validation, not all trained sequentially. More usually reduces its size; fewer enlarges it. Cannot exceed smallest-class participant count.",
    'batch_size': "Samples per update: usually windows for raw, file bags for fusion. Larger improves throughput/memory use with fewer updates per epoch; smaller updates more noisily/often. Prediction also batches by this without retraining.",
    'learning_rate': "Deep-model weight-update step scale. Higher can learn faster or oscillate/overshoot; lower is steadier but needs more updates. Traditional estimators and loaded-weight inference do not use it for updates.",
    'weight_decay': "Deep-optimizer weight regularization. Higher compresses weights and can underfit; lower relaxes it. AdamW uses decoupled decay; other optimizers follow their implementations. Not equivalent to logistic C.",
    'device': "Deep training/inference device, e.g. cpu, cuda or cuda:0. Changes speed/hardware and possibly floating-point results, not windows/classes. Missing requested GPUs raise errors, not CPU fallback.",
    'num_workers': "Data-loader worker processes; 0 loads in the main process. More can reduce waiting but costs memory/process overhead; cached/small tasks may not speed up. Not ensemble size.",
    'seed': "Seed for initialization, shuffling and sampling; model.seed_policy resolves the outer execution seed. Numeric size has no quality meaning. Editing does not retrain selected weights.",
    'optimizer': "Deep optimizer: SGD uses gradients/momentum; Adam/AdamW adapt from gradient history, with decoupled AdamW decay; RMSprop uses squared-gradient history. Loads algorithm-specific parameters, not comparable speed levels.",
    'class_weighting': "Class-loss weighting: none, inverse_frequency emphasizing rare classes, or effective_number smoothing counts. Counts use current training data only; minority gains may trade off against other classes.",
    'class_count_basis': "Class-count basis: independent participants or training rows (windows/files). Rows depend on recording counts; participant counts avoid treating windows as people. Affects weighting/balanced-softmax correction.",
    'training_balance': "Training file/role balance aligned with final aggregation. equal_files balances files per person; equal_role_families balances families then files. Only balance_line_weighted_v2 realizes these sampling probabilities; other samplers retain their rules.",
    'sampler': "Epoch sampling: exhaustive shuffled once; uniform_replacement resamples equally; balance_line_weighted_v2 uses participant/file/role weights; subject_balanced uses participant quotas; class_subject_balanced also balances classes. Changes repetitions/weights, not outer-fold membership.",
    'samples_per_epoch': "Rows drawn per replacement-sampling epoch; null uses dataset rows. Higher adds draws/updates and repeated rows; lower reduces cost. Participant-quota samplers use participant_window_quota instead.",
    'participant_window_quota': "Rows per participant draw: all, integer count, fraction or percentage. Larger adds rows; above available count samples with replacement. all does not equalize participant contribution.",
    'classifier_role_families': "Classification families: B static baseline, R post-exercise recovery, S/W project movement tasks, selected by role mapping. More families add activity data; removal excludes classification. Same-person B calibration requirements remain independent.",
    'loss': "Deep loss: cross_entropy penalizes wrong probabilities; focal_loss downweights easy samples; balanced_softmax adds log training-class counts to logits. Balanced softmax already corrects counts and cannot combine with class_weighting.",
    'focal_gamma': "Focal exponent in (1-correct-class probability)^gamma. Higher emphasizes poorly learned/noisy samples; lower approaches weighted cross-entropy; 0 removes this downweighting.",
    'class_weight_beta': "effective_number smoothing beta: normalized weights (1-beta)/(1-beta^count). Near 1 approaches inverse frequency; 0 equalizes present classes. Tunable only for this policy.",
    'label_smoothing': "Fraction of target mass redistributed across classes. Higher discourages overconfidence but can weaken separation; 0 uses hard labels. Does not modify data labels or inference weights.",
    'gradient_clip_norm': "Global gradient-norm cap before deep updates; null disables. Lower clips more often and can slow learning; higher triggers less. Does not clip PPG or predicted probabilities.",
    'deterministic_algorithms': "Require deterministic deep-backend algorithms and disable cuDNN benchmarking. Disabling may be faster but less repeatable. Enabled does not ensure bitwise equality across GPUs/dependencies or replace seeds.",
    'optimizer_parameters': "Current optimizer-specific parameters, replaced when selecting another optimizer. Affect deep weight updates only; loaded-weight Analyse performs no optimization.",
    'optimizer_parameters.betas': "Adam/AdamW history coefficients for mean gradient and mean squared gradient. Higher smooths but reacts slower; lower follows current gradients faster. The two coefficients serve distinct roles.",
    'optimizer_parameters.betas.0': "Adam/AdamW beta1 for mean gradient direction. Higher favors long-term direction with slower turns; lower follows current gradients. Not learning rate.",
    'optimizer_parameters.betas.1': "Adam/AdamW beta2 for squared-gradient history controlling coordinate step scales. Higher changes scales slowly; lower reacts faster to recent large gradients. Not a direct direction control.",
    'optimizer_parameters.eps': "Small positive adaptive-optimizer denominator constant. Higher limits amplification of tiny denominators but may weaken adaptation; lower can reduce extreme-case stability.",
    'optimizer_parameters.amsgrad': "Adam/AdamW AMSGrad uses the historical maximum second-moment statistic. Enabled prevents denominator decline with recent statistics and changes the trajectory; neither option guarantees accuracy.",
    'optimizer_parameters.maximize': "Maximize instead of minimize the objective. Classification uses loss, so normal training disables this. Enabled seeks larger loss, not better accuracy.",
    'optimizer_parameters.momentum': "SGD/RMSprop momentum retaining past movement. Higher adds inertia and may traverse flat regions faster or overshoot; lower tracks current gradients, 0 disables momentum.",
    'optimizer_parameters.dampening': "SGD damping multiplies new gradients by 1-dampening in momentum accumulation. Higher admits less new information; lower reacts faster. Must be 0 with Nesterov.",
    'optimizer_parameters.nesterov': "SGD Nesterov look-ahead update. Requires positive momentum and zero dampening; disabled uses ordinary momentum. More complex does not guarantee improvement.",
    'optimizer_parameters.alpha': "RMSprop squared-gradient memory coefficient. Higher smooths and reacts slower; lower follows recent gradients faster with potentially noisier step scales.",
    'optimizer_parameters.centered': "Centered RMSprop also tracks mean gradients and estimates variance as second moment minus squared mean. Changes denominators and state storage; no universally better direction.",
    'epoch_profile': "Descriptive epoch-policy label. Renaming does not add epochs; change epoch_rule and actual epoch fields.",
    'execution_mode': "Execution metadata normalized to formal in ordinary configuration. Not an alternate model equation; short debugging runs are arranged by the entry point.",
    'cache_policy': "Trainer state cache is currently disabled to avoid cross-fold reuse of fitted state. Distinct from deterministic preprocessing cache, which can remain enabled.",
    'outer_labels_visible_to_trainer': "Outer holdout-label visibility, fixed false. Labels cannot fit models or select inner epochs. A data-use contract, not a score-improvement switch.",
    'refit_on_all_outer_training': "After inner epoch selection, reinitialize and train on every participant in this outer-training fold; currently enabled. Different from optional full-cohort deployment --refit.",
    'n_classes': "Training-loss classes, aligned with model outputs and data order. Currently 3; editing alone cannot reuse incompatible labels/head weights.",
}

MODEL_HELP: dict[str, str] = {f'model.{key}': value for key, value in _MODEL.items()}
MODEL_HELP.update({f'training.{key}': value for key, value in _TRAINING.items()})
MODEL_HELP.update({
    'aggregation.balance_line': "Participant aggregation: line_a averages windows into files then weights files equally; line_b first averages files within B/R/S/W families then weights present families equally. Different total R influence when many R files exist. No retraining.",
    'aggregation.quality_weighting': "Enable quality-weighted probabilities; otherwise selected Line A/B uses equal weights. Low-quality windows/files contribute less, requiring real scores matching quality_weight_source.",
    'aggregation.quality_weight_source': "Quality source: none, route_file_q_rate weighting from files upward with ordinary window means, or legacy_window_sqi weighting windows into files before higher aggregation. Changes weighting levels; absent scores cannot be treated as 1.",
    'aggregation.hierarchy': "Derived hierarchy: window -> file -> participant, or window -> file -> role family -> participant. Describes probability aggregation, not training.",
    'aggregation.window_to_file': "Within-file window aggregation, derived as ordinary or historical SQI-weighted mean. Changing the source may change file probabilities without recomputing saved window predictions.",
    'aggregation.file_to_role': "Line B averages same-participant/same-family files; Line A skips this layer. Derived from balance and quality source so numerous files do not automatically increase total family weight.",
    'aggregation.role_to_participant': "Line B combines present family probabilities equally or by real quality scores. Line A averages files directly and skips this layer.",
    'aggregation.missing_role_policy': "Absent families receive no invented predictions; average renormalizes over present families. Missing families change contributors, not zero probabilities.",
    'aggregation.quality_weight_levels': "Weighting levels derived from quality_weight_source and Line A/B. Metadata for the selected algorithm, not an extra repeated-weighting switch.",
    'aggregation.direct_all_window_participant_mean': "Formal pipeline does not directly average all windows per participant, avoiding dominance by longer files. Such comparisons require an explicit report view, not changing this metadata.",
    'evaluation.statistics.bootstrap_replicates': "Confidence-interval bootstrap count, preserving participant-grouped predictions instead of treating windows as independent people. Higher is usually more stable/slower; lower suits previews. No extra model training.",
    'evaluation.statistics.paired_permutation_replicates': "Repeated participant-paired swaps for the null difference distribution. Higher refines Monte Carlo estimates with more time; lower previews faster. Does not increase independent sample size.",
    'evaluation.statistics.seed': "Resampling/permutation seed. Fixed inputs/settings/environment aid statistical reproducibility; larger values do not imply stronger significance.",
    'evaluation.statistics.lcb95_percentile': "Recorded lower-bound percentile. Current reports use a fixed 2.5% lower percentile and do not consume this editable field. Changing it affects neither current bounds nor predictions.",
    'evaluation.statistics.cluster_unit': "Participant with all repeat predictions is the resampling cluster, preserving dependence. Records the method, not an option to count windows as people.",
    'evaluation.statistics.confidence_interval': "Current implementation uses two-sided 95% percentile intervals. This field names the method; editing the string does not implement another algorithm.",
    'evaluation.statistics.lcb95_metrics': "Participant metrics whose lower bounds are recorded. Changes no predictions/classes; actual output is controlled by report modules and implementations.",
    'evaluation.statistics.paired_exchange_unit': "Permutation exchanges whole participant results between methods, keeping repeats together to preserve dependence and avoid inflated independent sample counts.",
    'evaluation.statistics.multiplicity_correction': "Holm adjustment within comparison families reduces multiple-testing false positives. This records the method; renaming does not load an unimplemented correction.",
    'evaluation.statistics.affects_automatic_selection': "Statistics never automatically choose the final model. Currently false: reports provide evidence without replacing the selected model based on p-values or intervals.",
})

# The fusion factory reuses these actual encoder parameters, not another formula.
_ENCODER_EXCLUDED = {'input_channel_order', 'input_channels', 'n_classes', 'seed', 'seed_policy',
                     'member_seeds', 'ensemble_size', 'feature_hidden_dim', 'fusion_hidden_dim', 'pooling'}
MODEL_HELP.update({f'model.signal_encoder.{key}': "Fusion signal encoder: "+value
                   for key, value in _MODEL.items()
                   if key not in _ENCODER_EXCLUDED and not key.startswith(('logistic_', 'svm_', 'extra_trees_'))})
MODEL_HELP['model.signal_encoder.model_id'] = "Select the fusion waveform encoder, pool window embeddings into files, then combine file features. CompactCNN/Inception/ShapeFormer switching changes structure/parameters and requires compatible weights."
MODEL_HELP['model.signal_encoder.dropout'] = "Signal-encoder dropout. CompactCNN/Inception use this only in an unused standalone head; fusion calls forward_features, so it has no effect there. ShapeFormer also uses internal attention dropout, stronger at larger values during training and disabled for inference."
for _field in ('kernel_sizes', 'dilations', 'pool_sizes', 'stage_channels', 'stage_dropouts',
               'out_channels', 'bottleneck_channels', 'depth', 'dilation', 'pool_size', 'residual_interval'):
    MODEL_HELP[f'model.signal_{_field}'] = "Fusion signal encoder: "+_MODEL[_field]
MODEL_HELP['model.signal_dropout'] = "Standalone head dropout of fixed Compact/Inception fusion encoders. Fusion calls forward_features rather than this head, so changing it does not change branch outputs. model.dropout controls the fusion layer."


def model_help(path: str, config_context: Mapping | None = None) -> str | None:
    """Look up a full path, falling back to a numerical list item's parent."""
    del config_context  # Wording explicitly distinguishes shared-field consumers.
    value = MODEL_HELP.get(path)
    if value is None and path.rsplit('.', 1)[-1].isdigit():
        value = MODEL_HELP.get(path.rsplit('.', 1)[0])
    return value
