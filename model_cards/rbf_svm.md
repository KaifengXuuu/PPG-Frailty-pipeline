# RBFSVM

- Machine ID: `rbf_svm`
- Scientific status: `reference_feature_baseline`
- Representation mode: `feature_vector`
- Eligible signal routes: `direct_x_filter`, `identity_direct`, `non_identity_x_ar_rate_only`
- Evaluation unit: participant after window→file, then config-dependent Line A equal-files or Line B equal-role-families aggregation
- Current classifier role scope: SQI off; only B and R are admitted
- Execution status: registered/constructible; scientific benchmark not run
- Independent test: absent; `independent_test=false`

## Identity and deviation

Fold-local imputation/scaling precede an RBF SVM.

This card describes only implemented interfaces and naming boundaries. The existence
of an architecture, passing unit tests, and historical scores are not V2 performance evidence.

## Limitations

- No independent frailty test set is available; formal scores must be named oof_validation_*.
- No V2 performance is claimed until the same frozen 5×5 participant protocol is run.
- The implementation is registered/constructible; the scientific benchmark has not been run.
- Only outer-training data may determine fitted transforms.

## Required provenance

Every formal result must bind participant/file/role, repeat/fold/seed, config hash,
manifest and fold hashes, preprocessing and feature hashes, signal route, aggregation,
model state, environment, and coverage. Every formal result must bind all identities listed above.
