# V1 transition archive

Everything below this directory is inactive historical material copied or moved
from the V1-to-V2 transition snapshot.  It is retained only for provenance and
must not be consumed by V2 validation, configuration loading, acceptance gates,
training, comparison, deployment, or scientific claims.

The archive contains old configurations, manifests, reports, split exports,
acceptance evidence, reduced-run artifacts, phase logs, and V1-only tools/tests.
Names such as `current`, `passed`, or `manual` inside the archive describe
their old snapshot only; they do not describe the status of final_pipeline_v2.

`INVENTORY.json` is generated once, without overwrite, by
`tools/materialize_historical_inventory_v2.py`.  It records the archived
content path, byte count, and SHA-256 for every file.  The hashes identify the
bytes as archived; they do not assert that those bytes are scientifically valid
or byte-identical to an external source that was not independently frozen.

