# ADR-007: V2 fixed epochs and outer-fold isolation

- Status: accepted_for_v2
- Source: V2-018

The default deep-training rule is fixed 10 epochs. Fixed 7 and fixed 15 are
named single-factor ablations. They are catalogued independently and are not
run by validation or the safe gate.

Outer-held-out labels are unavailable to model fitting, schedulers, callbacks,
feature selection, calibration and epoch choice. Imputation, scaling,
ShapeFormer discovery, ROCKET kernels and any learned quality component are fit
only on outer-training participants. Frozen split seeds control fold membership
only. Every single-model outer cell and final full-cohort refit uses training
seed 42; five-member ensembles use `[42,10042,20042,30042,40042]`.

No result may select epoch count after inspecting outer-fold performance.
