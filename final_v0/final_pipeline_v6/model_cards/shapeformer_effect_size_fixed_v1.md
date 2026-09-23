# ShapeFormerEffectSizeFixedV1

- Machine ID: `shapeformer_effect_size_fixed_v1`
- Scientific status: `experimental_ineligible_for_parity_claim`
- Representation mode: `raw`
- Eligible signal routes: `direct_x_filter`, `identity_direct`
- Evaluation unit: participant after window→file, then config-dependent Line A equal-files or Line B equal-role-families aggregation
- Current classifier role scope: SQI off; only B and R are admitted
- Execution status: registered/constructible; scientific benchmark not run
- Independent test: absent; `independent_test=false`

## Identity and deviation

Outer-fold fixed-length effect-size discovery defaults to 128 samples and stride 64; both controls are runtime-selectable and provenance-bound. It uses non-overlapping patch embedding before mask-aware generic self-attention and trainable shapelet distances; not PISD/original parity.

This card describes only implemented interfaces and naming boundaries. The existence
of an architecture, passing unit tests, and historical scores are not V2 performance evidence.

## Limitations

- No independent frailty test set is available; formal scores must be named oof_validation_*.
- No V2 performance is claimed until the same frozen 5×5 participant protocol is run.
- The implementation is registered/constructible; the scientific benchmark has not been run.
- Discovery method is effect_size_fixed_v1 and never substitutes for channel_specific_osd.
- Input sampling rate and shapelet length in samples/seconds are mandatory provenance.
- Patch size is at least two samples; raw sample-token attention is structurally rejected.

## Required provenance

Every formal result must bind participant/file/role, repeat/fold/seed, config hash,
manifest and fold hashes, preprocessing and feature hashes, signal route, aggregation,
model state, environment, and coverage. Every formal result must bind all identities listed above.
