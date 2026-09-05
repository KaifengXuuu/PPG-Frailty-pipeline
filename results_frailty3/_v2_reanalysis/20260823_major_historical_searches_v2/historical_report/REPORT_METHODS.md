# Historical reporter methods

- Analysis unit: one archived repeat summary; n=5 per historical configuration.
- Display: mean ± sample SD.
- Descriptive interval: mean ± t(0.975, n-1) × sample SD / sqrt(n).
- Ranking: subject BA, then subject macro-F1; ranking remains post hoc.
- The matched three-model table includes an exploratory exact sign-flip test across five aggregate repeat seeds, with Holm adjustment across three model pairs within each metric.
- Formal V2 participant-exchange P values are unavailable because participant OOF rows/probabilities were not archived.
- Every consumed report JSON plus the root CSV/manifests is hashed in `tables/source_evidence.csv`.
- Seeds and exact held-out participant rosters are audited within and across all four source studies in the split-audit CSVs.
- Every CSV table is also one worksheet in `tables/report_tables.xlsx`.
- Every generated figure has a numerical companion in `tables/table_figure_pairs.csv`.
