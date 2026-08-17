# M0 MAdenoiser confirmed-route private draft

> Private staging note only. Do not copy to _agent or display its drafted content until the user explicitly requests a draft and approves the target text.

## Candidate decision entry

- Decision ID: M0-MAD-001
- Status: confirmed_route_implementation_not_started
- Scope remains M0 extension; M1 is not authorized.
- Writes remain restricted to final_v0; root code, data, output, AGENTS.md, and _agent remain read-only.
- No network use or TROIKA/JOSS source retrieval has been authorized.

## Candidate TODO impact

1. Implement and validate SQI-v2 components: skewness, kurtosis, autocorrelation periodicity, template correlation, normalized spectral entropy, complete IBI plausibility, RED/IR agreement, interpretable flags, and fold-local calibration.
2. Define the 29-subject Motion supervision target, then implement subject-grouped nested threshold/CV and integrate fold-local motion probability into SQI before classifier standardization.
3. Implement four comparable route front ends with one common backend:
   - spectral_track_sqi
   - dual_ppg_bss_sqi
   - nonstationary_sqi
   - adaptive_sqi
4. Evaluate preliminary HR/PPI on pulse-transit-time-ppg using ECG R-peaks as HR/RRI reference; use train-fit ECG→PPG delay for absolute PPG event timing.
5. Produce OOF frailty feature blocks for motion HR/PPI, compare identical subject folds and seeds, and select by nested-CV subject BA.

## Candidate evidence boundaries

- The 29-subject frailty cohort has 261 raw role files and no current window-level optical-artifact truth.
- Existing PTT Motion A/B is a 22-subject activity-state experiment, not a 29-subject artifact CV.
- Current SQI has partial components and leakage/unit risks; route confirmation does not validate it.
- PTT peaks are ECG R-peaks, not PPG pulse peaks.
- Initial frailty comparison space is baseline plus four routes × HR-only/PPI-only/HR+PPI.
- Direct winner selection on the same 5-fold result is development selection, not independent final performance.

## Open decision to request before implementation

Define the 29-subject Motion target as one of:

1. manually annotated window-level optical artifact;
2. B/R/S/W activity/motion proxy, named only as activity/motion state;
3. independently defined peak/HR unavailability.

This choice changes label construction, threshold meaning, validation metrics, and paper claims.
