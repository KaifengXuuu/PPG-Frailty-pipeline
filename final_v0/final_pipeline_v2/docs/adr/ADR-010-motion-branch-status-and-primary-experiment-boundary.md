# ADR-010: V2 motion branch and primary boundary

- Status: accepted_for_v2_without_execution_gates
- Source: V2-002, V2-009d, V2-010 and V2-030

The external PTT mapping is frozen to the distal channels:
`pleth_1=RED`, `pleth_2=IR`. Provenance records both the source-page
description and the project adoption; the mapping is not inferred at runtime.

SQI defaults to `off`. `diagnostics_only` computes and archives raw
components but cannot alter retention, reducer choice, aggregation or
prediction. `route` is an ordinary selectable module whose calibrator fits only
outer-training data. Motion evidence is a separate optional research path; its
unrun internal/PTT comparisons limit scientific claims but do not authorize
core pipeline execution.

IMU reference preprocessing converts g to m/s² and degrees/s to rad/s, applies
the calibrated roll/pitch EKF, derives dynamic acceleration, angular magnitude
and jerk, and fits robust scaling on outer-training participants only. The
0.3-Hz LPF gravity separator is a separately named ablation, never a silent EKF
fallback.

Formal motion execution uses explicit source-bound commands. Source bytes,
schema, units, roster and split identities are validated as data contracts,
not as private authorization tokens or performance gates. Internal training
uses the frozen participant split; PTT evaluation cannot fit or recalibrate on
PTT labels. The frozen historical Light CNN is backup provenance, not an active
default.
