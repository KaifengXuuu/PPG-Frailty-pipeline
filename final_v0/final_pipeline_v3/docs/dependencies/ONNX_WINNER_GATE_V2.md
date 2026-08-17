# ONNX winner gate V2

The winner gate is an explicit post-selection operation. It never trains or
selects a model. It verifies a hash-bound manual selection and final-refit
bundle, requires a clean tracked V2 source tree and the live exact dependency
profiles, exports the model-input-to-probability graph, executes an ONNX Runtime
CPU readback, and atomically archives the model, both probability matrices,
certificate, and artifact index.

Supported reviewed conversion routes are PyTorch modules through
torch.onnx.export and scikit-learn LogisticRegression/SVC/ExtraTrees estimators
through skl2onnx. ROCKET/MiniROCKET and any converter failure emit
unsupported_no_certificate; they can never produce a winner-release pass.

An isolated Python 3.11 probe validated:

- onnx==1.20.0
- onnxruntime==1.23.2
- skl2onnx==1.20.0
- onnxscript==0.5.7 and onnx-ir==0.1.13
- inherited protected-stack torch==2.9.1+cu126, scikit-learn==1.8.0, and
  numpy==2.3.5
- fixed pre-fitted LogisticRegression and deterministic untrained torch module
  exports, with ORT probability and argmax parity

The probe did not modify conda ml, run scientific training, run CV, or export a
project winner. Its 41-package dependency closure and the complete 131-record
installed-distribution set (including the protected base exposed to the
isolated environment), Python/platform identity, lexical virtual-environment
executable and resolved protected-base interpreter are frozen in
`locks/onnx_winner_gate_py311_v2.json`. Unknown or missing distributions close
the gate. The
`onnx_winner_gate` profile is therefore a validated exact optional lock. It
must run from a recreated isolated environment whose prefix basename and live
closure match that lock; the ordinary conda ml prompt intentionally fails this
operation because it does not contain the isolated converter delta.

The source-bound producer is implemented, but remains operationally
fail-closed unless all of the following are present together: a clean tracked
V2 source snapshot, an immutable eligible manual selection, a verified
`trusted_final_refit_v2` bundle, its independently hashed final-refit
attestation and manifest, the exact isolated ONNX runtime, and explicit
`--confirm-onnx-execution`. Release preflight re-runs Python and ONNX Runtime
probabilities from the hash-bound golden inputs and rejects artifact, source,
environment, class-order, argmax, absolute-error or relative-error drift.
The conversion policy is fixed at opset 17, absolute tolerance 1e-5 and
relative tolerance 1e-5; callers cannot relax it. Generic research bundles
cannot be presented as final winners.

Official metadata and APIs:

- <https://pypi.org/project/skl2onnx/>
- <https://docs.pytorch.org/docs/stable/onnx_export.html>
- <https://onnxruntime.ai/docs/api/python/api_summary.html>

Machine evidence:

- `locks/onnx_winner_gate_isolated_probe_v2.json`
- `locks/onnx_winner_gate_tiny_smoke_v2.json`
- `locks/onnx_winner_gate_py311_v2.json`
- `locks/onnx_winner_gate_installed_distributions_v2.txt`
- the `onnx_winner_gate` row in `locks/profiles.lock.json`
