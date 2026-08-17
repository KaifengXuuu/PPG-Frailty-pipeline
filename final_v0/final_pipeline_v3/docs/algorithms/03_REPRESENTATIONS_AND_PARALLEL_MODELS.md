# V2 representations and model catalogue

Four representation modes share one manifest, fold registry, label mapping,
OOF schema and aggregation contract:

- raw: eight-channel windows plus mask;
- feature-vector: one ordered file vector;
- feature-matrix: ordered feature rows/columns plus mask;
- fusion: a pooled raw-file embedding concatenated once with its file vector.

The formal catalogue carries explicit architecture parameters and training
fields. It includes classical, ROCKET, Compact/Inception, fusion, two explicit
five-member ensemble comparisons, and separately named ShapeFormer routes.
Line A is default and Line B is an aggregation/training comparison. Catalogue
materialization never fits or ranks a model.

The canonical ShapeFormer route is channel-specific OSD/PISD. Each shapelet has
one source channel; discovery and best-fit distance search use that channel.
The generic network branch remains multivariate. The joint-channel
`multichannel_pip_centered_ig` route, if retained, is a separately named
ablation and cannot be called PISDPort or used as fallback. The literature
reference remains unavailable until its faithful implementation gate passes.

