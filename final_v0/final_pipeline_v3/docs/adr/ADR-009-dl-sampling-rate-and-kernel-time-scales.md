# ADR-009: V2 DL resampling with fixed sample kernels

- Status: accepted_for_v2
- Source: V2-019/V2-019a

Native signal processing, timing and audit remain at 400 Hz. DL-only views may
be anti-aliased and resampled to 100, 160, 200 or 400 Hz after windowing, with
source/target rates and mapping provenance saved.

Convolution kernels remain the same sample counts across sampling rates. They
are deliberately not converted to preserve physical duration. The registered
12 cases are six named cases for CompactCNN1D and the same six for
InceptionTimeFull: reference, 10 s context, 100/160/200 Hz, and dilation 2.
Each case changes only its declared factor relative to the corresponding
reference; no unreviewed Cartesian product is generated. This is a named
fixed-sample ablation family, not a physical-time-matched design, and it
excludes other models.

No fixed-kernel profile runs during materialization, validation or safe tests.
