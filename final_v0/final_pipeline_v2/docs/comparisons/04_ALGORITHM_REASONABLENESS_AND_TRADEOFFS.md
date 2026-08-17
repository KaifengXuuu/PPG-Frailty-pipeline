# V2 registered comparison rationale

- Line A tests an equal-file mixture; Line B tests equal physiological
  role-family influence. Deployment role frequency is unknown, so neither is
  declared universally superior.
- Fixed 7/10/15 epochs test training duration without outer-label selection.
- Fixed-sample DL resampling deliberately changes physical receptive duration;
  it answers a different question from physical-time matching.
- 0.5–5 Hz tests a narrower passband against the 0.2–8 Hz default.
- LPF gravity separation tests a simpler estimator against the calibrated EKF
  without becoming a fallback.
- Single models remain defaults; exact five-member ensembles are comparison
  entries.
- Artifact reducers remain named comparisons until a human reviews complete
  evidence.

All comparisons report BA, macro-F1, both LCB95 columns, worst-fold/class
metrics, ECE, variability, confusion matrices, parameters and inference cost.
No single metric automatically chooses a final.

