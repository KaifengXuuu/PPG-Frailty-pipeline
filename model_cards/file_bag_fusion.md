# FileBagFusion

- Machine ID: `file_bag_fusion`
- Scientific status: `optional_composable_signal_encoder`
- Representation mode: `fusion`
- Eligible signal routes: `direct_x_filter`, `identity_direct`
- Evaluation unit: participant after window→file, then config-dependent Line A equal-files or Line B equal-role-families aggregation
- Current classifier role scope: SQI off; only B and R are admitted
- Execution status: registered/constructible; scientific benchmark not run
- Independent test: absent; `independent_test=false`

## Identity and deviation

Composes one registered raw forward_features encoder with a file-level feature vector after window pooling. Compact, Inception, faithful channel-specific OSD ShapeFormer, its scalar-distance ablation, the newer effect-size model, and the isolated legacy effect-size port are selectable signal modules.

This card describes only implemented interfaces and naming boundaries. The existence
of an architecture, passing unit tests, and historical scores are not V2 performance evidence.

## Limitations

- No independent frailty test set is available; formal scores must be named oof_validation_*.
- No V2 performance is claimed until the same frozen 5×5 participant protocol is run.
- The implementation is registered/constructible; the scientific benchmark has not been run.
- Shapelet discovery is derived only from verified outer-training file bags.
- File features are never repeated per window and never enter shapelet discovery.
- This optional composer is not an additional default catalogue candidate.

## Required provenance

Every formal result must bind participant/file/role, repeat/fold/seed, config hash,
manifest and fold hashes, preprocessing and feature hashes, signal route, aggregation,
model state, environment, and coverage. Every formal result must bind all identities listed above.
