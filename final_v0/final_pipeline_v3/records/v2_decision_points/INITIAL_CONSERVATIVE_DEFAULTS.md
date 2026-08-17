# Superseded initial V2 defaults / 已替代的 V2 初始默认

状态 / Status: **superseded — do not use as the confirmation authority / 已替代，不再作为确认权威**。

This early seven-item note was created before the complete implementation audit and before
the user's later decisions. It is retained only as provenance showing what the first
conservative build assumptions were. It must not be used to infer user approval.

本文件最初只有七个保守假设，早于完整落地审计及用户后续决定。现仅保留为历史溯源，
不得据此推断用户已经同意任何路线。

The authoritative live registry is
[HUMAN_CONFIRMATION_POINTS.md](HUMAN_CONFIRMATION_POINTS.md), which now records 28
individually status-labelled points, including:

- V2-007: confirmed no-precalibration quaternion ESKF primary plus mandatory LPF comparator;
- V2-006, V2-008, V2-011, and V2-015: partially confirmed with unresolved sub-decisions;
- V2-028: pending 0.2–8 Hz versus 0.4–8 Hz direct PPG-band decision;
- every remaining point: explicit V1 default, decision needed, and impact.

Any later V2 implementation must use the detailed registry by ID, not this archived note.
后续 V2 必须按详细清单 ID 逐项确认，不能回退使用本文件。
