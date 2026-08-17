# ADR-006: V2 file and participant aggregation

- Status: accepted_for_v2
- Source: V2-004 corrections

R1–R4 are four relax files, not four physiological roles. The real role
families are B, R, S and W, and no temporal operation crosses file boundaries.

Two mutually exclusive comparison lines are retained:

- Line B canonical: train with `equal_role_families`; aggregate
  `window -> file -> role -> participant`, equal across available
  B/R/S/W roles and equal within a role.
- Line A named ablation: train with `equal_files`; aggregate
  `window -> file -> participant` using
  `equal_files_no_role_layer`.

Training and aggregation must use the same line; A/B hybrids are invalid.
Current formal inputs are B and R because SQI routing is off. Missing files or
families are reported as coverage and weights are renormalized over available
inputs. Diagnostics-only SQI cannot change weights, retention or probabilities.

Deployment role distribution is unknown. Line A estimates an equal-file
mixture and can give a multi-file R family greater influence. Line B estimates
an equal-role mixture and is the thesis reference. Neither
can be declared deployment-optimal without a frozen deployment input contract.
