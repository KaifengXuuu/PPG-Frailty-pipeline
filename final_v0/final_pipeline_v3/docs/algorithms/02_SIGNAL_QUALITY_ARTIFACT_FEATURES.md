# V2 signal, SQI, artifact and feature routes

The default direct filter is 0.2–8 Hz; 0.5–5 Hz is a named ablation. SQI has
three exact modes:

| Mode | Computation | Effect on classifier |
|---|---|---|
| `off` | no SQI computation | none; V2 default |
| `diagnostics_only` | raw components saved separately | none |
| `route` | may keep/drop/select reducer | disabled pending supervision |

`diagnostics_only` calls the raw diagnostic evaluator and cannot emit routing
weights or normalized threshold decisions. Failures are recorded without
dropping a record.

Identity is the current artifact reducer. Named EMD-sifting-rate-only,
CEEMD-lite-NLMS legacy and DWT-A2 legacy routes remain ablations. No fictional
ANS is registered. A non-identity route is rate-only: post-reducer morphology
and amplitude-dependent fields are not applicable, and reducer failure cannot
silently fall back to the direct route.

IMU reference units are m/s² and rad/s. EKF gravity removal must pass its unit
tests and persist conversions/covariances. Outer-training-participant robust
scaling covers all dynamic acceleration, gyro and derived magnitude/angular/
jerk channels. Per-window amplitude normalization is forbidden. LPF gravity
separation is a separate ablation.

