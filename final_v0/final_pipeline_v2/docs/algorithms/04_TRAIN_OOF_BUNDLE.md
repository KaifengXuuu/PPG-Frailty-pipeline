# V2 training, OOF, operational cost and final refit

Outer CV uses frozen split seeds for membership while the effective training
config supplies the model seed or explicit ensemble member roster. OOF rows
preserve distinct `split_seed` and `training_seed` fields plus the
source-snapshot hash. Ensembles accept one or more unique uint32 member seeds,
emit one row per member and one exact arithmetic-mean row per subject/cell.
The named historical comparison preset remains
`[50042,60042,70042,80042,90042]`.

Each completed cell publishes typed OOF tables, metrics, confusion matrices,
resolved architecture/training provenance and a hash/byte artifact index.
Complete root metrics are rebuilt only from 25 trusted cells and one
participant prediction per participant/repeat. Missing real parameter count or
CPU batch-1 inference measurement makes a config ineligible; missing values are
null, never zero. Operational measurement is explicit opt-in.

Final refit is a separate operation after human selection. It must resolve a
plan from one complete indexed run plus a hash-bound purpose-specific selection
record; caller-authored hashes are untrusted. Full-cohort refit inherits the
selected config's single seed or ensemble roster. It is not invoked by
validation, the benchmark or comparison archive.

ONNX is not a V2 winner requirement. A later deployment project may add an
export/latency protocol without changing the scientific OOF pipeline.
