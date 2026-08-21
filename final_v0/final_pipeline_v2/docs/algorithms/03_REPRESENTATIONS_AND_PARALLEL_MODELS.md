# V2 representations and model catalogue

Four representation modes share one manifest, fold registry, label mapping,
OOF schema and aggregation contract:

- raw: canonical 8-channel windows plus mask, ordered as RED, IR,
  A_dyn_x/y/z, GX/GY/GZ; every channel is independently robust-scaled inside
  each DL window, while the physical-unit analysis views remain unchanged;
- feature-vector: one ordered file vector (282 fields with all groups enabled);
- feature-matrix: chronological 10 s/2 s-hop engineering rows plus mask,
  exactly 115×150; feature validity is provenance, not predictor channels;
- fusion: a pooled raw-file embedding concatenated once with its selected file
  vector and paired validity channels.

`features.enabled_groups` selects any non-empty composition of basic PPI/rate,
HRV time-domain, HRV spectral, HRV nonlinear, morphology, dual optical, and
engineering-summary modules. Selection is canonicalized before hashing and the
same content-addressed registry is consumed by vector/fusion extraction, fold
transforms, validators, outer experiments, and final refit. Feature-matrix is a
separate fixed engineering-only 115×150 contract.

The formal catalogue carries explicit architecture parameters and training
fields. It includes classical, Compact/Inception, fusion, one explicit raw
five-member ensemble comparison, and separately named ShapeFormer routes.
Role-aware Line B (window -> file -> role -> participant) is the canonical
reference. Equal-files Line A is a separately named ablation. Catalogue
materialization never fits or ranks a model.

The canonical ShapeFormer route is channel-specific OSD/PISD. Each shapelet has
one source channel; discovery and best-fit distance search use that channel.
The faithful downstream implementation is
`LiteratureShapeFormerChannelSpecificOSD`; it is executable and high-compute,
not gated. ShapeFormer, its generic branch, raw CNNs, and fusion all receive the
same `x_dl_all8_window_norm` tensor. SQI, motion detection, denoising, and
engineering continue to consume `processed_imu_physical`; peaks and optical
features consume amplitude-preserving `x_analysis`/`x_native`.

ROCKET/Ridge and MiniROCKET are retired from the executable registry, catalog,
and studies. The 115×150 feature-matrix representation remains available while
its next model is selected explicitly.

The optional `FileBagFusion` composer selects its `signal_encoder` as a nested
runtime module. CompactCNN, full/small Inception, faithful channel-specific OSD,
the separately named channel-specific scalar-distance ablation, the newer fixed
effect-size ShapeFormer, and `ShapeFormerLegacyEffectSizePort` are supported. The existing `FileBagFusionCompact` and
`FileBagFusionInception` names remain compatibility routes. For either
ShapeFormer discovery route, only verified outer-training file bags are
expanded to raw windows; file features never enter discovery. The selected
`features.enabled_groups` vector is transformed fold-locally, concatenated once
after window pooling, and the same preparation path is used for final refit.

`ShapeFormerLegacyEffectSizePort` is a parallel historical ablation, not an
alias for OSD or the newer scalar-distance/effect-size downstream. It executes
the old channel-wise class-v-rest effect map with configurable fixed length,
stride, shapelets/class, discovery-window cap, and candidates/class/channel,
then the functional two-convolution local branch and source-position
shape-token attention branch. Historical `processes` and `verbose` were
execution/UI controls and are not accepted as model parameters. Historical
`len_w` only populated unused bookkeeping; V2 instead exposes the actually
consumed local convolution width and shapelet search span. Discovery remains
outer-train only for raw and nested FileBagFusion, and final refit repeats it on
the exact verified all-29 scope.

A_dyn magnitude, Omega, and J are excluded from every frailty representation
and predictor. They remain available only in a separately named 11-channel
motion-model/denoiser augmentation ablation; the motion reference is also the
same axes-only 8-channel order. The joint-channel
`multichannel_pip_centered_ig` route, if retained, is a separately named
ablation and cannot be called PISDPort or used as fallback.
