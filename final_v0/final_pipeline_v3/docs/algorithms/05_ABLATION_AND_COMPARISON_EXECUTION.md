# V2 ablation and comparison execution

Named ablations are implemented and retained but are not run automatically.
Each comparison holds manifest, fold roster, seeds and budget fixed and changes
only its declared factor. Epoch 7/15, fixed-sample DL resampling, 0.5–5 Hz
filtering, LPF gravity separation, artifact reducers, ensemble and balance
profiles remain independent families; they are not multiplied into an
unreviewed Cartesian sweep.

A comparison archive accepts only complete, indexed 5×5 run directories. It
verifies every artifact SHA/byte identity, exact participant/repeat/fold
coverage and matching labels. It then rebuilds participant-macro balanced
accuracy, macro-F1, worst-fold BA, worst-class recall/F1, ECE, variability and
confusion matrices. Bootstrap produces BA and macro-F1 LCB95. Paired
participant permutation tests are Holm-adjusted within comparison-family and
metric.

Operational parameter count and CPU inference cost must be measured to make a
configuration ranking-eligible. Each comparison group may retain at most ten
highest-BA eligible configurations for review, while showing all principal
metrics and both LCB columns. This is a review list, not automatic model
selection.

Comparison archives remain immutable with no selections. A human may later
select one or more purpose-specific finals, including an ablation, while its
original registry role/provenance remains permanent.
