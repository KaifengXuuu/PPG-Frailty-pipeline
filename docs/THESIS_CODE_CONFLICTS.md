# Thesis and Implementation Conflicts

The read-only thesis source is `Kaifeng_Masterarbeit_draft_v1_0.docx` in the original
repository root. Conflicts are ordered by potential impact on predictions,
statistical conclusions, or real-world inference. V6 inherits the V5 resolution:
continue following the frozen V2 numerical semantics. Changes to mathematics,
architectures, sampling rates, splits, or workflow require an explicit new proposal
and a user decision; documentation must not silently change implementation behavior.

`finalcase` is the user's selected Rank 2 `tuned_all_roles_small_no_gravity`.
At the time of the original V5 conflict audit, a formal 5×5 finalcase comparison
had not been completed. This document describes the execution contract, not proof of
25-fold numerical equivalence with V2. The criterion is identical inputs, splits,
GPU/CUDA/PyTorch/dependencies, exact discrete values, and `atol=1e-6, rtol=0` for
floating-point values.

## 1. The Final Experiment Does Not Have a Unique Identity

**Impact: critical; changes training data, the model, and optimization together.**

The draft's general baseline uses B/R roles, 400 Hz, CompactCNN, and Adam/batch 64.
The final candidate in Appendix E is closer to all roles, 64 Hz, Small Inception,
and AdamW/batch 16. These are different experiments; neither result reproduces the other.

The named `baseline` is retained, while the selected Rank 2 is separately named
`finalcase`. Runs explicitly select a preset, complete YAML, or manual CLI
configuration. The formal study is `configs/studies/finalcase.yaml`. Thesis results,
code outputs, and model files must record a case ID, not just “final model”.

## 2. Rank 2 Uses Different Gravity Handling from the Draft's Leading Candidate

**Impact: critical; directly changes three acceleration channels and the model-input distribution.**

The thesis's top-ranked candidate estimates/removes gravity with Profile A. The
selected Rank 2 uses `sensor_filter_only_no_gravity_removal`. Rank 2 still applies
sensor low-pass filtering, same-participant static calibration, and SI conversion;
it simply does not subtract estimated gravity.

`finalcase` implements Rank 2 unchanged rather than reverting it to Rank 1. Report
them as distinct comparison cases; gravity handling is not a display setting.

## 3. IMU Units, Static Calibration, and Filtering Are Described Inconsistently

**Impact: high; changes amplitude, dynamic acceleration, jerk, and all related features.**

The general methods text can be read as retaining g/deg/s, omitting axis calibration
for some gravity profiles, and using a gravity-filter order different from the
implementation. Relevant V2 paths actually use a same-participant B-role 5–100 s
static calibration/bias-removal interval, then convert to m/s² and rad/s. Profile A
follows the filter contract frozen in the implementation.

This behavior is retained. Changing units, the calibration interval, or filter
order invalidates existing weights, features, and comparisons and requires a
separate ablation.

## 4. Participant Aggregation Targets Different Estimands

**Impact: high; changes probabilities, BA/F1/AUC, confusion matrices, and P values.**

One thesis passage averages all of a participant's windows directly, giving longer
recordings greater weight. V2's final Line B instead follows
`window → file → role family → participant`, using ordinary means at every level
and equal weights for available role families.

`finalcase` uses Line B. Per-fold window/file/role/participant probabilities are
preserved so other aggregations and statistical units can be studied after training.
Those results are new post-hoc estimands and must not overwrite the declared primary
result.

## 5. Deep-Input Sampling and Tail-Window Rules Differ

**Impact: high; changes sequence lengths, window counts, boundary samples, and effective receptive fields.**

The general workflow can suggest 400 Hz throughout and mentions zero-padding short
records. The Appendix E/V2 final candidate actually starts with canonical 400 Hz
preprocessing and uses polyphase anti-alias resampling to 64 Hz for deep input,
5 s windows, a 2.5 s hop, a distinct right-aligned complete tail window, no padding,
and at most 128 windows per file.

Execution follows each resolved configuration. The 64 Hz contract above belongs to
finalcase; it must not be described as a universal default for all modules.

## 6. Outer-CV Weights Can Be Confused with Deployment Refit Weights

**Impact: high; does not change OOF values but changes the training population represented by the model and the conclusions it supports.**

Thesis performance comes from participant-grouped outer OOF predictions, whereas
a live demonstration needs loadable weights. Each fold model has seen only that
fold's outer-training participants. Selecting weights using an OOF metric also uses
held-out performance, so the selection is not an unbiased final deployment model.

All fold bundles are saved. By default, the median fold after sorting by
`(balanced_accuracy, repeat, fold)` is published for research/Dash trials. Runtime
`refit` defaults to false. Only `--refit` triggers the existing all-29 refit after
outer folds for **every case** in the run. Refit-related training fields alone do
not trigger this step.

All-29 weights have no internal self-evaluation; quoted performance must still come
from the preceding outer OOF evaluation. Clearly distinguish median-fold and all-29
bundles in the thesis and demonstrations.

## 7. Real-World Dynamic Inputs May Lack a Static B Calibration Recording

**Impact: high but deployment-specific; the missing-input policy systematically changes the IMU distribution.**

Training dynamic R/S/W recordings relies on a static B recording from the same
participant. Real-world use may provide only dynamic recordings. Silently using
zero bias, another participant, or approximate gravity is not equivalent to the
training preprocessing.

CLI/Dash inference currently requires same-participant B alongside dynamic input;
B-only input can be processed independently. Silent missing-B calibration remains
the inherited **V5 TODO** and must be tested in its own future ablation. It is not
an implicit fallback.

## 8. Historical ShapeFormer Epoch Selection Uses Outer-Held-Out Information

**Impact: high but limited to historical modules; introduces optimistic bias.**

The thesis acknowledges that some historical ShapeFormer early stopping referenced
the outer-held-out fold. This is not a fair comparison with fixed-epoch training or
inner grouped selection restricted to outer-training participants.

Faithful/history and corrected modules are retained for review without rewriting
their mathematical algorithms. Reports must identify leakage status; historical
results must not enter the formal leakage-free leaderboard by default.

## 9. PRV Eligibility Thresholds Differ

**Impact: medium to high; changes feature values, missingness, and feature-model inputs.**

The thesis body abbreviates spectral PRV eligibility to at least 60 s/few intervals,
while a table specifies 5 min and 200 intervals for SampEn. V2 distinguishes rate,
time, spectral, and nonlinear quantities: time-domain features require at least
60 s, 30 intervals, and 0.8 coverage; spectral features require at least 300 s,
200 intervals, and 0.8 coverage; SampEn requires at least 200 intervals.

These tiered thresholds are retained. Because finalcase is a raw model, this conflict
primarily affects parallel feature analyses and diagnostics, not its main input tensor.

## 10. PTT Dataset Version, Sampling Rate, and Purpose Are Described Differently

**Impact: medium; primarily affects external motion/peak branches.**

The thesis states PTT-PPG v1.0.0/500 Hz. V2 uses local 1.1 data, normalizes it to
400 Hz under the project processing contract, and cites the v1.0 page as wavelength/
unit evidence. PTT supports beat/motion development and transfer; it is not an
independent frailty test cohort.

V2 version, unit, and resampling semantics remain unchanged. Reports must not label
PTT transfer as an independent frailty test.

## 11. An Older Default Peak Detector Remains in the Thesis Text

**Impact: medium; changes peaks/PPI, PRV, morphology, and SQI.**

One overall-workflow passage still names Aboy++, while results and Appendix E use
MSPTDfast v2.3. The inherited V2/V5 finalcase uses the MSPTDfast Python peak-only port.
Aboy/project variants remain explicit historical/ablation modules. Failure does not
silently switch detectors.

## 12. Bootstrap/Permutation Counts and Test Applicability Differ

**Impact: medium to low; predictions stay fixed, but CI/P-value precision and supportable conclusions change.**

The thesis overview specifies 10,000 participant-cluster permutations. The declared
V2/V5 configuration uses 10,000 participant bootstrap resamples, 100,000 paired
permutations, and Holm correction within each comparison family. Some ROC-AUC
P values are N/A because their assumptions are not satisfied.

Reports must record resample counts, seed, cluster/exchange unit, and multiplicity
family. Tests using other units from saved file/fold predictions must be recorded
as separate analyses.

## 13. Feature-Matrix Dimensions Have Drifted Historically

**Impact: low to medium; using old dimensions makes model inputs incompatible.**

Older material stated `115×150`; current code/configuration and later thesis
sections use `146×variable-K`, padding only within a batch and excluding padding
with a mask. The execution configuration and registry are authoritative; old
dimension descriptions are not carried into the implementation.

## 14. Motion Transfer Is Confused with “Independent Testing”

**Impact: low numerically, but high risk for conclusions.**

Internal PTT22 OOF and PTT→Frailty transfer are different forms of evidence. Internal
frailty data currently provide participant outer OOF, not a roster-disjoint
independent frailty test. Report `test` mode accepts explicitly independent test
evidence only; renaming outer OOF does not make it a test set.

## 15. V2 Couples Training and Presentation

**Impact: low for scientific values, high for maintenance and compute cost.**

Some V2 entry points produce plots/HTML immediately after training, repeating that
cost for repeated comparisons. The inherited V5 pipeline writes data, pipeline
Excel, and weights only. `analyse_report.py` composes ordinary classification
figures/tables from existing results and produces complete registered suites, HTML,
and report Excel for specialized branches. This structural change should not alter
training or predictions and allows repeated analysis of a run without retraining.

## Thesis Locations

- 3.1 Overall workflow: Word page 42.
- 3.2 Datasets: pages 46–47.
- 3.3 Signal preparation: pages 48–59.
- 3.4 Feature engineering: pages 59–70.
- 3.5 Models: pages 70–77.
- 3.6 Evaluation: pages 77–80.
- 4.1 Dynamic experiments: page 81 onward.
- 4.2 Frailty classification: page 88 onward.
- Appendix E: page 141 onward.
