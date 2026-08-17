# ADR-010: V2 motion branch and primary boundary

- Status: accepted_for_v2_with_execution_gates
- Source: V2-002, V2-009d, V2-010 and V2-030

The external PTT mapping is frozen to the distal channels:
`pleth_1=RED`, `pleth_2=IR`. Provenance records both the source-page
description and the project adoption; the mapping is not inferred at runtime.

SQI defaults to `off`. `diagnostics_only` computes and archives raw
components but cannot alter retention, reducer choice, aggregation or
prediction. `route` is disabled until supervised thresholds/weights are
frozen. Motion override is an optional ablation and remains inactive pending
the required internal supervised evidence and external PTT evaluation.

IMU reference preprocessing converts g to m/s² and degrees/s to rad/s, applies
the calibrated roll/pitch EKF, derives dynamic acceleration, angular magnitude
and jerk, and fits robust scaling on outer-training participants only. The
0.3-Hz LPF gravity separator is a separately named ablation, never a silent EKF
fallback.

Formal motion execution uses explicit source-bound commands and requires a
scientific confirmation flag. Internal training binds the 29-participant
single-seed split. PTT evaluation also requires hash-bound unit-resolution
evidence; absent evidence closes the command before execution. The frozen
historical Light CNN is backup provenance, not an active default.
