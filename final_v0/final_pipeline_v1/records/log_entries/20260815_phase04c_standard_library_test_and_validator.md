# Phase 04c — Test and validation harness / 测试与验证框架

- Re-scanned the V1 tree and local dependency inventory before writing.
- Confirmed pytest, ruff, mypy, and coverage are not installed; no undeclared test dependency was introduced.
- Added a standard-library unittest runner with selectable data/signal/artifact/feature/model/training/integration suites and strict JSON reporting.
- Added a deterministic validator for required paths, specification hash, Python AST, bilingual documentation, forbidden legacy runtime imports, strict JSON, and fully resolved configs.
- The validator is intentionally extensible; later batches must add data, route, OOF, bundle, and comparison invariants before the final pass is authoritative.
