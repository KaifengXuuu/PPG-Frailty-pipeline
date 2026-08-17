# Aura hrv-analysis 1.0.2 compatibility evidence

Status date: 2026-08-17

Scope: isolated, fixed-PPI function comparison only. Nothing was installed into
or changed in the conda `ml` pipeline environment.

## Official metadata

The authoritative [PyPI release metadata for hrv-analysis
1.0.2](https://pypi.org/project/hrv-analysis/1.0.2/) declares Python >=3.5 and
dependencies including `nolds>=0.4.1`, `astropy>=3.0.4`,
`numpy>=1.15.1` and `scipy>=1.1.0`. Its universal wheel SHA-256 is
`4d3ea73673a19b4c9acdc86a3c33008ed3c097f036027a6e9717d5175ee5b3dd`.

The official [nolds release
history](https://pypi.org/project/nolds/) lists 0.6.3 and 0.6.2. The default
resolver selects 0.6.3, whose Python 3.11 import failure is recorded in
[upstream issue 80](https://github.com/CSchoel/nolds/issues/80). The highest
published version tested successfully with hrv-analysis 1.0.2 is therefore
`nolds==0.6.2`; its universal wheel SHA-256 is
`91fa5982432d306f9889129bf2f270080cfa11cba69c528a00bdf2abe0e3819b`.

Python 3.11 wheels begin on the tested Astropy 5.2 line. Astropy 5.2.2 retains
the `astropy.stats.LombScargle` import used by hrv-analysis 1.0.2, while the
current Astropy documentation locates LombScargle under
`astropy.timeseries`. The tested Astropy 5.2.2 Linux CPython 3.11 wheel
SHA-256 is
`e14b5a22f24ae5cf0404f21a4de135e26ca3c9cf55aefc5b0264a9ce24b53b0b`.
Astropy 5.2.2 also requires NumPy before the removals in NumPy 2; the tested
NumPy 1.26.4 Linux CPython 3.11 wheel SHA-256 is
`666dbfb6ec68962c033a450943ded891bed2d54e6755e35e5835d63f4f6931d5`.

## Isolated resolution and smoke

The conda-ml interpreter version, Python 3.11.14, created temporary virtual
environments. The protected environment itself was never mutated.

- Unconstrained `hrv-analysis==1.0.2` resolved `nolds==0.6.3`; `pip check`
  passed but import failed with `TypeError: 'nolds.datasets' is not a package`.
- Pinning `nolds==0.6.2` removed that failure. With current Astropy 8.0.1,
  import then failed because `astropy.stats.LombScargle` is absent.
- `hrv-analysis==1.0.2`, `nolds==0.6.2`, `astropy==5.2.2` and
  `numpy==1.26.4` resolved on Python 3.11.14 and passed `pip check`.
- The project adapter ran all five deterministic, untouched 512-interval PPI
  fixtures successfully. No cleaner or classifier was invoked. Constant or
  mathematically degenerate fixture outputs that upstream returned as NaN were
  represented by the adapter as JSON `null`; upstream runtime warnings are
  retained as diagnostics rather than hidden.
- The pinned optional requirements file records the tested scientific package
  versions. It is intentionally separate from the main pipeline environment.

## Frozen conclusion

The Aura comparison profile is supported through its isolated pinned
environment. It must not be installed as a delta into conda ml because its
Astropy compatibility requires NumPy 1.26.4 while the main V2 environment uses
NumPy 2.3.5. This profile remains a function-only comparison and provides no
classifier or clinical-performance evidence. The ordinary CLI reports the
installed backend/version and fixture outcome; it does not claim a tracked-source
or exact-environment attestation. The reproducible input is
`requirements/requirements-prv-aura-compare.txt`.
