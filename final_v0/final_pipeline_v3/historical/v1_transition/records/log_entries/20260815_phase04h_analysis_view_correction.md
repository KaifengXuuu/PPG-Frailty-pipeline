# Phase 04h — Analysis-view correction / Analysis 视图修正

- Re-read contract §5.3, ADR-003, the resolved YAML generator, diagram, and current signal facade.
- Removed an unused 0.4–8 Hz secondary direct filter from the configuration.
- Frozen semantics are now explicit: direct `x_analysis=x_filter` at zero-phase 0.2–8 Hz; non-identity `x_analysis=x_ar` and is rate-only.
- This prevents configuration/runtime drift and preserves the amplitude-sensitive direct contract.
