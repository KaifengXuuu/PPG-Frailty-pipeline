# Phase 04g — Matrix input-dimension contract / Matrix 输入维度合同

- Re-read the resolved configs and model factory before editing.
- Replaced the feature-matrix sentinel `input_channels=-1` with an explicit schema-derived resolution rule; negative dimensions remain invalid.
- Added `input_channels_resolution` to every resolved model section so no factory or runner silently guesses dimensions.
- Configuration human names remain canonical presentation names; the model registry records the corresponding stable machine ID.
