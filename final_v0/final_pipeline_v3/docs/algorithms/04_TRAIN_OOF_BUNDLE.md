# V2 training, OOF, operational cost and final refit

Outer CV uses frozen split seeds for membership while every single-model cell
uses training seed 42. OOF rows preserve distinct `split_seed` and
`training_seed` fields plus the source-snapshot hash.
Five-member ensembles emit five member rows and one exact arithmetic-mean row
per subject/cell with member seeds
`[42,10042,20042,30042,40042]`.

Each completed cell publishes typed OOF tables, metrics, confusion matrices,
resolved architecture/training provenance and a hash/byte artifact index.
Complete root metrics are rebuilt only from 25 trusted cells and one
participant prediction per participant/repeat. Missing real parameter count or
CPU batch-1 inference measurement makes a config ineligible; missing values are
null, never zero. Operational measurement is explicit opt-in.

Final refit is a separate operation after human selection. It must resolve a
plan from one complete indexed run plus a hash-bound purpose-specific selection
record; caller-authored hashes are untrusted. Full-cohort single-model refit
uses seed42, or the exact five ensemble seeds. It is not invoked by the
benchmark, comparison archive or acceptance gate.

The eventual winner must pass the ONNX gate. Other ablations do not need ONNX
export.
