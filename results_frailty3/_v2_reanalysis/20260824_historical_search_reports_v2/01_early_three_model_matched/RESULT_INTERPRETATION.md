# Result interpretation

- Descriptive leader: InceptionTime with subject BA 72.7 ± 6.0% and macro-F1 72.3 ± 5.9%.
- ShapeFormer-PISD trails the leader by 11.1 BA points and 11.8 macro-F1 points.
- ShapeFormer is below both comparators on BA in all 10 matched pair/repeat rows=True and on macro-F1 in all 10 rows=True; its mean runtime is 12.3× the leader.
- This supports excluding this historical ShapeFormer-PISD implementation from the ordinary mega-study on utility/cost grounds; it does not establish that every ShapeFormer implementation is inferior.
- The exact five-repeat sign-flip P values are exploratory; only 32 sign patterns exist, so the minimum attainable two-sided P is 0.0625.
- The held-out fold supplied the historical best-epoch/early-stopping trajectory and the reported score, creating selection contamination; absolute scores are therefore candidate-generation evidence, not selection-unbiased OOF confirmation.
