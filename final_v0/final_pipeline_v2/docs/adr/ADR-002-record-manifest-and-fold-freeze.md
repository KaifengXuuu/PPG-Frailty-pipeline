# ADR-002: V2 record manifest and fold freeze

- Status: accepted_for_v2
- Source: M2 frozen manifest/registry plus V2-001/V2-001a decisions

V2 verifies 261 internal recordings from 29 participants and imports corrected
participant memberships. Runtime code must never call a splitter for these
memberships.

- `splits/sgkf5_seed42_v2.csv`: single seed42 table used by the internal
  motion reference, 29 participant rows, SHA-256
  `130b2887eb29a5a534397b4ce4dc7032f9de30ae46533fa0b2c41559ff4a1284`.
- `splits/sgkf5_repeated_grouped_5x5_v2.csv`: frailty formal 5×5 table, 145
  rows/25 cells, split seeds `42,10042,20042,30042,40042`, SHA-256
  `1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702`.

Every file belonging to one participant inherits that participant's outer
membership. All fitted objects record training/OOF participants and the
manifest, fold and source-snapshot hashes. Internal OOF is validation, not an
independent test.

Both CSVs preserve source registry ID
`frailty3_future_corrected_sgkf5_v2`; a derived motion registry ID does not
replace the source artifact identity.

