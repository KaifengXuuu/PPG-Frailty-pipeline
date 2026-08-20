# ADR-007: V2 epoch strategies and outer-fold isolation

- Status: accepted_for_v2
- Source: V2-018

The reference deep-training rule is fixed 10 epochs. Ordinary V2 accepts any
positive fixed epoch count and an outer-train-only grouped inner-selection
strategy; fixed 7 and fixed 15 remain named catalogue comparisons. Validation
and the safe test suite never train a formal study.

Outer-held-out labels are unavailable to model fitting, schedulers, callbacks,
feature selection, calibration and epoch choice. Imputation, scaling,
ShapeFormer discovery, ROCKET kernels and any learned quality component are fit
only on outer-training participants. Inner grouped selection uses only a
deterministic partition of that outer-training roster; the chosen epoch is then
refit on the complete outer-training set. Frozen split seeds control fold
membership only. Ordinary training seeds and ensemble member rosters are
explicit effective-config inputs. The named comparison preset uses member
seeds `[50042,60042,70042,80042,90042]`; final refit inherits the selected
configuration rather than imposing a core seed.

No result may select epoch count after inspecting outer-fold performance.
