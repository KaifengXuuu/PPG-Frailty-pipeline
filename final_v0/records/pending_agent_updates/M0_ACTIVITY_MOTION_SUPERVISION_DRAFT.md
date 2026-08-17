# M0 Activity/Motion supervision private draft

> Private staging note only. Do not copy to `_agent` or display drafted content until the user explicitly asks for a draft and approves the target text.

## Candidate decision

- Decision `M0-MOT-001` confirms the 29-subject target as activity/motion state.
- Map B and R1–R4 to static; map S1–S2 stand-and-sit and W1–W2 walking to motion.
- Preserve the full role and acquisition sequence for recovery and frailty-feature exploration.
- Reuse the PTT detector architecture/preprocessing concept, but retrain and recalibrate within the local device domain.

## Candidate evidence correction

- Verified early assets are three-class/multiclass SVM datasets, SVM training code, and 649 SVM model files.
- The saved qualitative result says Rest was recognized well while Walking and Sitting/Standing were mixed.
- No verifiable three-class motion CNN source, checkpoint, numeric result, or 3×3 confusion matrix was found.
- Existing A/B CNN motion results are direct sit-versus-walk/run binary experiments and must not be described as a collapsed three-class CNN.

## Candidate TODO impact

1. Add a frozen StageManifest for 29 subjects and all nine roles.
2. Implement subject-disjoint nested 5-fold detector transfer/from-scratch comparisons and fold-local threshold calibration.
3. Produce strict OOF `p_active`, integrate it softly into SQI-v2, and retain hard rejection only as an ablation.
4. Pair each active bout with the immediately following Rk and build route-specific active/recovery HR/PPI features.
5. Keep all route and frailty selection nested within subject folds and preserve coverage/failure metadata.

## Evidence boundaries

- Activity labels are not window-level optical-artifact truth.
- File modification time supports the active→R sequence and is provisional until a formal acquisition manifest is available.
- Existing label-table HRrecovery/maxHR fields have unknown generation provenance and are not accepted as supervision truth.
- No training or new benchmark was run during this documentation step.
