# Phase 04b — Reference configs and baseline / 参考配置与基线

- Re-scanned the 766-line locked specification before this logical write batch.
- Added a generator for four fully resolved YAML configurations; runtime has no config inheritance or hidden behavior defaults.
- Added a read-only baseline-audit generator that writes only under V1 and labels historical metrics non-strict.
- Added the legacy-to-V1 migration crosswalk and explicit no-shortcut boundary.
- Validation planned immediately after materialization: strict config load, strict JSON parse, source-hash inventory, and tracking synchronization.
