# ADR-006: V2 file and participant aggregation

- Status: accepted_for_v2
- Source: V2-004 corrections

R1–R4 are four relax files, not four physiological roles. The real role
families are B, R, S and W, and no temporal operation crosses file boundaries.

Two mutually exclusive comparison lines are frozen:

- Line A default: train with `equal_files`; aggregate
  `window -> file -> participant` using
  `equal_files_no_role_layer`.
- Line B: train with `equal_role_families`; aggregate
  `window -> file -> role_family -> participant`, equal across available
  B/R/S/W families and equal within a family.

Training and aggregation must use the same line; A/B hybrids are invalid.
Current formal inputs are B and R because SQI routing is off. Missing files or
families are reported as coverage and weights are renormalized over available
inputs. Diagnostics-only SQI cannot change weights, retention or probabilities.

Deployment role distribution is unknown. Line A estimates an equal-file
mixture and can give a multi-file R family greater influence. Line B estimates
an equal-family mixture and is a physiological sensitivity analysis. Neither
can be declared deployment-optimal without a frozen deployment input contract.

