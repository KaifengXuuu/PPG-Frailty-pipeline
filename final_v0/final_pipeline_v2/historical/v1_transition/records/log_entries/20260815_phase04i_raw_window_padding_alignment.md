# Phase 04i — Raw-window padding alignment / Raw 窗口 padding 对齐

- Re-read contract §5.3, the resolved window config, `CompactCNN1D.forward_features`, and feature-matrix padding rules.
- The V1 reference raw route now emits complete 5-second windows only; it does not ask a model that rejects non-trivial masks to consume right padding.
- Feature-matrix remains explicitly right-padded after fold-local transformation with a row mask.
- A future padded-raw route requires a tested mask-propagating convolution/pooling policy and a distinct config ID.
