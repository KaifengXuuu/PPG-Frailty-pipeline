# V2 representations and model catalogue

Four representation modes share one manifest, fold registry, label mapping,
OOF schema and aggregation contract:

- raw: canonical 8-channel windows plus mask, ordered as RED, IR,
  A_dyn_x/y/z, GX/GY/GZ;
- feature-vector: one ordered file vector;
- feature-matrix: ordered feature rows/columns plus mask;
- fusion: a pooled raw-file embedding concatenated once with its file vector.

The formal catalogue carries explicit architecture parameters and training
fields. It includes classical, ROCKET, Compact/Inception, fusion, two explicit
five-member ensemble comparisons, and separately named ShapeFormer routes.
Role-aware Line B (window -> file -> role -> participant) is the canonical
reference. Equal-files Line A is a separately named ablation. Catalogue
materialization never fits or ranks a model.

The canonical ShapeFormer route is channel-specific OSD/PISD. Each shapelet has
one source channel; discovery and best-fit distance search use that channel.
ShapeFormer, its generic branch, raw CNNs, and fusion all receive the same
canonical 8-channel tensor directly; there is no model-specific projection.
A_dyn magnitude, Omega, and J are excluded from every frailty representation
and predictor. They remain available only in a separately named 11-channel
motion-model/denoiser augmentation ablation; the motion reference is also the
same axes-only 8-channel order. The joint-channel
`multichannel_pip_centered_ig` route, if retained, is a separately named
ablation and cannot be called PISDPort or used as fallback. The literature
reference remains unavailable until its faithful implementation gate passes.
