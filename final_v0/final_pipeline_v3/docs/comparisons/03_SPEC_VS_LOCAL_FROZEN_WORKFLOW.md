# V2 authority precedence

1. Explicit human V2 decisions.
2. Frozen M2 manifests, memberships and source hashes.
3. Active V2 configuration/catalogue and resolved-run provenance.
4. DEV0 specification where it does not conflict with items 1–3.
5. Historical V1 code/results as provenance only.

Runtime conveniences, unit-test fixtures and old “current” artifacts are never
parameter authorities. A PISD failure cannot fall back to effect-size discovery;
an EKF failure cannot fall back to LPF; an unavailable optional PRV backend
cannot impersonate the local implementation.

