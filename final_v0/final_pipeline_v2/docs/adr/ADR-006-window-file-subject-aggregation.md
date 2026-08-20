# ADR-006: V2 file and participant aggregation

- Status: accepted_for_v2
- Source: V2-004 corrections

R1–R4 are four relax files, not four physiological roles. The real role
families are B, R, S and W, and no temporal operation crosses file boundaries.

Two independently selectable aggregation modules are retained:

- Line B reference default: aggregate
  `window -> file -> role -> participant`, equal across available
  B/R/S/W roles and equal within a role.
- Line A alternative: aggregate
  `window -> file -> participant` using
  `equal_files_no_role_layer`.

Training balance (`equal_files` or `equal_role_families`) is a separate input
used by the sampler and train/inner BA; it is not bound to reporting
aggregation, so either training balance can be paired with either reporting
line. The same held-out file OOF can be replayed as window-balanced,
Line A and Line B without retraining. Current reference inputs are B and R.
Missing files or
families are reported as coverage and weights are renormalized over available
inputs. Diagnostics-only SQI cannot change weights, retention or probabilities.

Deployment role distribution is unknown. Line A estimates an equal-file
mixture and can give a multi-file R family greater influence. Line B estimates
an equal-role mixture and is the reference default. Neither
can be declared deployment-optimal without a frozen deployment input contract.
