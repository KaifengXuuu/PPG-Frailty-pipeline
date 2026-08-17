# M3 profile-locked peak and resampling phase 27 / M3 Profile 锁定峰检测与重采样第 27 阶段

- 时间 / Date：2026-08-15
- 状态 / Status：implemented_pending_reference_test
- 流程 / Process：重扫新 preprocessing profiles、公共 API、峰检测与 external resampling 调用点后，把原先只携带 profile 名的两条路径升级为运行时强制绑定。
- 算法 / Algorithm：峰检测只接受 `purpose=peak_detection_input`、future-active、400 Hz、0.4–8 Hz、三阶 SOS、notch-off 的离线或移动 profile；外部重采样只接受 256/500 Hz，并以一个有理数 sample-coordinate 映射同步处理波形、时间、valid mask 与峰标注。
- 结果 / Result：新增 `ExternalResampleResult` 和唯一 future-active facade `resample_external_ppg_to_400`；低层 `resample_poly_explicit` 不再从包级公共 API 暴露。已补正向同步映射、125 Hz 拒绝、错误 peak-purpose 与错误 fs 负例，下一阶段运行完整测试后冻结结果。
- 边界 / Boundary：MIMIC 125 Hz 当前不属于该 PTT/Sim external profile，必须在未来单独登记，不能通过调用参数绕过；valid mask 的 nearest-source 映射和峰事件的 source→target rounding 均显式写入 provenance。
