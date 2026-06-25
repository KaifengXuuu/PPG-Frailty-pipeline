# TODO

状态：draft  
来源：用户当前要求、`_agent/PROJECT_HANDOFF.md`、代码结构检查。  
最后手动更新时间：2026-06-22  
用途：记录明确可执行任务，并把未完成事项与对应脚本/模块一一对应。

## 高优先级

- [ ] 高：复查 frailty3 主训练与评估脚本的最终 CLI/defaults
  - 日期：2026-06-22
  - 用户需求：继续实验前避免旧默认值误用。
  - 涉及文件/模块：`frailty_3class_classifier.py`, `frailty_3class_holdout_eval.py`, `frailty_3class_overfitting_sweep.py`
  - 当前状态：脚本经历多轮更新；部分默认值可能反映旧实验阶段。
  - 阻塞点：需逐项核对 model/input/window/role/split/early-stopping/output 参数。
  - 下一步：列出当前 defaults，与最新实验设计对照，必要时再提交修改草稿。

- [ ] 高：继续 overfitting sweep stage2，不只用自动 Top2
  - 日期：2026-06-22
  - 用户需求：降低 InceptionTime 过拟合，提高 frailty3 三分类泛化。
  - 涉及文件/模块：`frailty_3class_overfitting_sweep.py`, `results_frailty3/_overfitting_sweep/20260608_1206_overfitting_sweep_stage1_rank2`
  - 当前状态：stage1 已完成 930 runs；Top4 为 `s1_085`, `s1_091`, `s1_105`, `s1_163`。
  - 阻塞点：自动 `--stage2-top-n 2` 会只选 `s1_085` 和 `s1_091`，覆盖面不足。
  - 下一步：使用 `--stage2-top-n 4` 或手动指定 Top4，比较 mean BA、macro F1、worst-class F1、CI low、std 和 Pre-Frail vs Robust confusion。

- [ ] 高：为选定 frailty3 config 做 final training/export 方案
  - 日期：2026-06-22
  - 用户需求：CV/holdout 只用于估计；部署需要最终模型。
  - 涉及文件/模块：`frailty_3class_classifier.py`, `frailty_3class_holdout_eval.py`
  - 当前状态：rank2 InceptionTime raw 5s/50% overlap/patience20 当前较平衡，但仍未定版。
  - 阻塞点：尚未固定 final config、训练策略、保存内容和部署格式。
  - 下一步：设计 final training/export，保存 scaler、label map、window 参数、feature schema、模型权重和评估摘要。

- [ ] 高：诊断 Pre-Frail vs Robust/Non-Frail 混淆
  - 日期：2026-06-22
  - 用户需求：当前主要错误不是 Young，而是两个老年组边界。
  - 涉及文件/模块：`frailty_3class_classifier.py`, `analyze_sweep.py`, `frailty_3class_holdout_eval.py`
  - 当前状态：rank2 aggregated confusion 显示 Pre-Frail 与 Robust/Non-Frail 互相误分明显。
  - 阻塞点：subject 数少，fold/repeat 方差大。
  - 下一步：按 subject、role、静/动态状态、HR/HRV、窗口质量分层分析错误来源。

- [ ] 高：正式推进 dynamic heartbeat / peak / IBI / HRV 模块
  - 日期：2026-06-22
  - 用户需求：动态 PPG 降噪失败后，转向直接提取可靠 peak/IBI/HRV。
  - 涉及文件/模块：`ppg_peak_hr_gating_train.py`
  - 当前状态：代码结构已包含 ECG preflight、beat supervision、IBI loss、LODO、scorecard、motion detector benchmark、ONNX wrapper。
  - 阻塞点：尚未完成正式全量训练和稳定评估。
  - 下一步：跑非 smoke 配置，读取 scorecard，重点检查 per-dataset/per-subject/per-activity、extra-holdout、LODO、delay analysis。

- [ ] 高：将新 detector + peak/IBI 模型接入主 notebook
  - 日期：2026-06-22
  - 用户需求：`ppg_analyse4_calib.ipynb` 应从旧 denoiser 路线切到 motion detector + dynamic heartbeat extractor。
  - 涉及文件/模块：`ppg_analyse4_calib.ipynb`, `ppg_peak_hr_gating_train.py`, `funcs.py`
  - 当前状态：notebook 是当前主分析入口；旧 denoiser 不应作为可靠动态修复模块。
  - 阻塞点：最终 detector/peak-IBI 模型尚未定版。
  - 下一步：先定义 notebook 输入输出接口，再接入已验证模型 bundle。

## 中优先级

- [ ] 中：决定 PPI/HRV/manual features 是否保留在最终 frailty3 候选
  - 日期：2026-06-22
  - 涉及文件/模块：`frailty_3class_classifier.py`, `analyze_sweep.py`
  - 当前状态：支持 `extra_input=0/PPI/HRV`；PPI/HRV 作为表格特征 MLP fusion。
  - 下一步：只按 config-level group summary 判断，不按单 run 判断。

- [ ] 中：检查采样率、时间戳和重采样规范
  - 日期：2026-06-22
  - 涉及文件/模块：`frailty_3class_classifier.py`, `funcs.py`, `ppg_analyse4_calib.ipynb`
  - 当前状态：handoff 明确该点需核查，避免 400 Hz 或有效时间轴处理错误。
  - 下一步：形成数据规范，并同步到后续文档批次。

- [ ] 中：为 `ppg_peak_hr_gating_train.py` 输出 ONNX/CPU-only 部署模块
  - 日期：2026-06-22
  - 涉及文件/模块：`ppg_peak_hr_gating_train.py`, `pttppg_denoiser_hybrid_export_onnx.py`, `pttppg_denoiser_onnx_runtime.py`
  - 当前状态：旧 denoiser 已有 ONNX 经验；新 peak/IBI 模型尚需独立 runtime。
  - 下一步：在模型稳定后导出 deploy bundle，并做 CPU-only smoke test。

- [ ] 中：增强 ECG detector 与 PPG delay 校准
  - 日期：2026-06-22
  - 涉及文件/模块：`ppg_peak_hr_gating_train.py`
  - 当前状态：已有 `analyze_ppg_ecg_delay` 等逻辑。
  - 下一步：明确 ECG R peak 与 PPG pulse peak 的 delay 建模方式，避免监督标签相位偏差。

- [ ] 中：评估 HRV 指标层准确性
  - 日期：2026-06-22
  - 涉及文件/模块：`ppg_peak_hr_gating_train.py`, `funcs.py`, `frailty_3class_classifier.py`
  - 当前状态：peak/IBI 训练目标已有；HRV 层指标尚未系统评估。
  - 下一步：重点评估 SDNN、RMSSD、LF/HF 等对 frailty 分类有意义的指标。

- [ ] 中：清理并归档旧 dynamic denoiser 路线
  - 日期：2026-06-22
  - 涉及文件/模块：`pttppg_denoiser_hybrid_*`, `pttppg_stage2_denoiser.py`, `pttppg_pipeline_v7_4_noleak_viz_ae.py`
  - 当前状态：动态去噪路线已判定失败，但仍有历史脚本和输出。
  - 下一步：文档中标注 deprecated/reference，避免后续 chat 误用为主线。

- [ ] 中：小范围评估 ShapeFormer，不进入大规模 sweep
  - 日期：2026-06-22
  - 涉及文件/模块：`shapeformer_port.py`, `frailty_3class_classifier.py`
  - 当前状态：port 已实现，PISD 运行成本高，提升不明确。
  - 下一步：仅做小 ablation，确认是否有保留价值。

## 低优先级

- [ ] 低：决定是否接入 W&B
  - 日期：2026-06-22
  - 涉及文件/模块：frailty3 sweep scripts
  - 当前状态：当前使用本地 CSV/JSON/PNG/Markdown，已够用。
  - 下一步：只有在 sweep 规模继续扩大时再评估。

- [ ] 低：整理项目级 README/实验日志
  - 日期：2026-06-22
  - 涉及文件/模块：`README.md`, `_agent/*`, result directories
  - 当前状态：信息散在 handoff、scorecard、summary、notebook 和脚本中。
  - 下一步：在 `_agent` 文档整理完成后，再决定是否写项目级 README。
