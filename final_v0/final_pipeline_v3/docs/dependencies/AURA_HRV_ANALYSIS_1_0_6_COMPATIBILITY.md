# Superseded Aura hrv-analysis 1.0.6 compatibility evidence

This probe is retained as historical dependency evidence. V2-035a supersedes
the active comparison target with hrv-analysis 1.0.2; see
`AURA_HRV_ANALYSIS_1_0_2_COMPATIBILITY.md`.

Status date: 2026-08-16

Scope: isolated, fixed-PPI function comparison only. This profile never enters
classifier training and was not installed into the conda ml pipeline
environment.

## Published metadata

The authoritative [PyPI JSON metadata for hrv-analysis
1.0.6](https://pypi.org/pypi/hrv-analysis/1.0.6/json) declares:

- Python >=3.11,<3.14;
- nolds>=0.6.3;
- wheel SHA-256:
  116e749fcb317785f5e274f227faa79d80fd1e2fb4b2e1d689d08364ae5653d2.

The upstream
[Aura pyproject.toml](https://github.com/Aura-healthcare/hrv-analysis/blob/master/pyproject.toml)
independently records version 1.0.6, Python >=3.11,<3.14 and
nolds>=0.6.3.

The authoritative [PyPI JSON metadata for
nolds](https://pypi.org/pypi/nolds/json) lists 0.6.3 as the newest published
release. Its wheel SHA-256 is
ba3fc9c30ba7a2c6eb8756eeb644dcf0d18e597ebe9ae170371b4b5c760213cc.

Consequently, 0.6.3 is currently the only published nolds version satisfying
the 1.0.6 package constraint. Versions through 0.6.2 do not satisfy
nolds>=0.6.3; selecting one would be a metadata-conflicting downgrade, not a
compatible solution.

## Isolated Python 3.11 smoke

Resolved packages included hrv-analysis==1.0.6 and nolds==0.6.3. Package
resolution and pip check passed. Import then failed before a fixed-PPI
function could run:

    TypeError: 'nolds.datasets' is not a package

The failure is triggered while nolds.datasets asks
importlib.resources.files(...) to treat that module as a package.
The still-open upstream
[nolds issue #80](https://github.com/CSchoel/nolds/issues/80) records the same
Python 3.11 / nolds 0.6.3 traceback and identifies that
resources.files(__name__) receives the nolds.datasets module rather than a
package.

## Conclusion and gate

There is no presently published nolds version that is both:

1. permitted by the exact hrv-analysis 1.0.6 dependency metadata; and
2. demonstrated to import successfully in the requested Python 3.11
   comparison environment.

The Aura comparison profile therefore remains fail-closed and unavailable.
There is no silent downgrade and no local/vendor patch. A future human
decision must choose between waiting for an upstream release or authorizing a
separately reviewed, hash-pinned packaging-only patch of 0.6.3.
