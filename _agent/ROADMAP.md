# ROADMAP

状态：draft  
来源：用户项目总纲、用户当前进度说明、`_agent/PROJECT_HANDOFF.md`  
最后手动更新时间：2026-06-23

## 当前总路线

本项目从“恢复动态 PPG clean waveform”逐步转向更稳健的路线：静态段做 PPG 波形与 HRV 特征，动态段先识别 motion/static 状态，再直接提取 peak、IBI、HR/HRV，最终将静态 PPG 特征、动态 heartbeat 特征和 IMU 状态融合到 frailty classifier。

## 阶段路线

1. 文档与交接系统整理
   - 状态：进行中
   - 目标：将 `PROJECT_HANDOFF.md` 拆分进 `_agent` 各职责文档，并归档 handoff。
   - 下一步：完成 4 批草稿审核后，等待用户“确认录入”。

2. 静态 PPG 预处理与 Aboy++ peak/HRV
   - 状态：暂定可用
   - 目标：形成 thesis-ready 的透明信号处理模块。
   - 下一步：核查采样率/时间戳，整理最终算法说明。

3. Dynamic PPG denoising 路线归档
   - 状态：失败，保留历史价值
   - 目标：明确 denoiser 不再作为主线。
   - 下一步：保留 gating/motion detection 思路，归档旧 denoiser 脚本。

4. Dynamic heartbeat extraction
   - 状态：待完成
   - 目标：用 `ppg_peak_hr_gating_train.py` 从动态 PPG 提取可靠 peak、IBI、HR/HRV。
   - 下一步：正式训练、scorecard、LODO、extra-holdout、delay analysis、ONNX/CPU-only。

5. Frailty3 classifier
   - 状态：主线进行中
   - 目标：从当前约 63% 提高到 73%+。
   - 下一步：继续 InceptionTime overfitting stage2，诊断 Pre-Frail vs Robust 混淆，尝试静态/动态 HRV 和 recovery features。

6. Final fusion pipeline
   - 状态：未完成
   - 目标：融合静态 waveform、动态 HR/HRV、IMU motion state。
   - 下一步：先稳定 dynamic heartbeat 模块，再接入 `ppg_analyse4_calib.ipynb` 和 frailty3 classifier。
