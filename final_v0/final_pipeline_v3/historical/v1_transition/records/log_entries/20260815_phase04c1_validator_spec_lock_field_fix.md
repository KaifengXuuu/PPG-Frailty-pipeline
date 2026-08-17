# Phase 04c.1 — Validator lock-field correction / 验证器锁字段修正

- Re-read `SPEC_LOCK.json` and recomputed the attached specification SHA-256 before editing.
- Corrected the validator from nonexistent `sha256` to the authoritative `source_sha256` field.
- The observed file hash remained `cd7c4907...3c5000`; no specification or lock content changed.
