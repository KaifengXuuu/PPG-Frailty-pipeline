CPU-only deploy bundle

Files:
- peak_hr_gate_model.onnx : model weights for ONNX Runtime inference
- model_reuse_params.json : windowing, thresholds, I/O specification, and deployment metadata

Inference contract:
- input: float32 normalized PPG window shaped [batch, 1, window_samples]
- outputs: peak_logit, ibi_pred, gate_logit
- postprocess: sigmoid on peak_logit and gate_logit, then apply exported thresholds