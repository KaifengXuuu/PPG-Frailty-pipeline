# 非根归档代码逐文件 I/O 与版本关系 / Archived Code I/O and Lineage Inventory

- 状态 / Status：`complete; historical_only`
- 覆盖 / Coverage：`CODE_FILES.jsonl` 中全部23个非根代码/Notebook，逐字节复扫并核对SHA；所有 `.py` 静态编译通过。
- 总判定 / Decision：全部属于历史、归档、探索或当前根脚本的直系前身；不得覆盖当前根实现。

## 1. Frailty、Dash、detector、v7与SVM归档

| 路径 | 职责/算法 | 静态输入 | 静态输出与实际对应 | Lineage与状态 |
|---|---|---|---|---|
| `archiv/frailty_3class_classifier - Copy_8channels_08062026.py` | 8通道Frailty3；传统PPI/HRV、CNN、Inception、ShapeFormer/PISD | StudyData、Youngers、V7标签；Red/IR/ACC/GYRO | `datasets/frailty3_*fs64*`、`models/frailty3_*`、`results_frailty3/*`；对应2026-05-27/28早期实验 | 关键历史基线；测试折早停泄漏、外部ShapeFormer路径；被根分类器替代 |
| `Arc/ppg_analy2.ipynb` | 单大单元Dash；Aboy/HRV/SpO2/IMU/denoise/v8 | `.env`、8通道CSV、v8 NPZ | 交互图与可选HRV CSV | 与NPZ-select版约99.75%相似；保存运行先键错后`UnboundLocalError`；历史失败 |
| `Arc/ppg_with_detector_v8.py` | 最早v8 detector接入旧Dash | `.env`固定路径、CSV、旧schema NPZ | Dash与HRV CSV | 根`ppg.py`前身；旧model键/UI/保存契约脆弱 |
| `Arc/ppg_with_detector_v8_npz_select.py` | 增加目录与NPZ选择器 | `.env`、CSV、用户NPZ | Dash与用户路径HRV CSV | 上一版本直接增量；后被viz/根版本替代 |
| `Arc/ppg_with_detector_v8_npz_select_viz.py` | 增加坐/动色带可视化 | 同上 | Dash、HRV CSV、活动覆盖图 | 与根`ppg.py`约94.2%行相似；历史直系前身 |
| `Arc/pttppg_dash.ipynb` | v7 setup1/2结果Dash浏览器 | 根v7脚本、`results/setup{1,2}` JSON | 内嵌Dash，无新固定文件 | 保存输出记录当时重复`groups`语法错误；历史失败 |
| `Arc/pttppg_detector_v8_scores.py` | 10 PPG+27 IMU、Mahalanobis、lag、logistic | PTT `s*_*.csv` | `results_detector_v8` summary/NPZ/CM/ROC/PR；现存6文件精确对应旧schema | v8历史基线；CV前全局fit、跨记录lag、IMU主导 |
| `Arc/pttppg_detector_v8_scores_audit.py` | 初版完整单模态/融合/walk-run审计 | PTT同上 | bundle/summary/audit图 | scores与fix2之间；序列化/绘图顺序及activity-label问题 |
| `Arc/pttppg_detector_v8_scores_audit_fix2.py` | 加ROC/PR、pooled CM，改`trapezoid` | PTT同上 | `results_v8_audit`早期结构 | 旧NumPy兼容问题；核心泄漏/lag未解 |
| `Arc/pttppg_detector_v8_scores_audit_fix3.py` | 预计算IMU阈值、修CM命名 | PTT同上 | 同上 | fix6直接前身；局部修正，不改方法学缺陷 |
| `Arc/pttppg_detector_v8_scores_audit_fix6.py` | 尝试全面JSON sanitize | PTT同上 | 同上 | 确定性`NameError`：调用未定义`_json_sanitize`；归档失败 |
| `Arc/pttppg_detector_v8_scores_audit_fix8.py` | 加sanitizer与headless Matplotlib | PTT同上 | 同上 | 与根fix9约98.6%相似；最终bundle `dict(dataclass)`会`TypeError` |
| `Arc/pttppg_pipeline_v7_2_noleak_viz.py` | CNN-BiLSTM AE、1D U-Net、setup1 proxy/setup2 ECG-HR、可视化 | PTT pleth4/5/6、ECG/peaks、IMU | 默认`results_v72_noleak_viz`（实际不存在） | 与根`cnnppg_v7.py`约86.7%相似；历史可视化分支 |
| `Arc/pttppg_stage1_detector.py` | 独立PPG/IMU logistic、threshold、lag、OR/AND | PTT、activity文件名标签 | `results_stage1` scaler/clf/JSON/CM；现存17文件精确对应 | v7.3/v7.4检测前身；实质activity detector |
| `Arc/pttppg_pipeline_v7_3_noleak_viz.py` | rule OR/AND、lag与STFT MaskNet | PTT pleth1/2、ECG/peaks、IMU | `results_v7_3` 33文件契约匹配 | 根v7.4直系基线；无denoiser holdout，离散`a`无有效梯度 |
| `Arc/svm_dataset_train.ipynb` | 旧45特征、Scaler/PCA/SVC及Dash | segment/label旧CSV流 | `datasets/motion_dataset_*`、`models/svm_motion_*.pkl` | 对应早期552/791样本；被根`svm2_dataset_train.ipynb/.py`扩展；首cell future-import位置错误，保存大量类别缺失/recall警告 |

## 2. PPG_Testing 历史数据分析脚本

| 路径 | 职责/算法 | 静态输入 | 静态输出与实际对应 | 状态/问题 |
|---|---|---|---|---|
| `PPG_Testing_05_01_2026/Archive/ptt_ppg_dataset_analysis.py` | 单文件中段、Chebyshev、Savitzky–Golay、HR/SDNN/SpO2 | 硬编码`s2_sit.csv` pleth1/2 | 仅打印/显示图 | 批处理前身；列缺失提示与实际检查列名不一致 |
| `PPG_Testing_05_01_2026/Archive/ptt_ppg_dataset_analysisv2.py` | 上述目录批处理版 | 硬编码`PPGdf` CSV | `Archive/PPGdf/plots/*.png`；现存18 PNG匹配 | 已落盘历史基线；无汇总CSV |
| `PPG_Testing_05_01_2026/Archive/ptt_ppg_dataset_analysis_esther.py` | 500Hz、中间1/3、5s窗、HR/SDNN/SpO2 | Esther桌面绝对路径、pleth1/2 | 外部`plots/<stem>_plot.png` | 不可移植；脚本要求pleth1/2而本地镜像只有Ir1/Red1，会全部跳过；未找到对应产物；SpO2未校准 |
| `PPG_Testing_05_01_2026/Archive/ptt_ppg_dataset_analysis_fingertiponly.py` | 400Hz RedFinger/IrFinger、HR/SDNN/SpO2/有效窗 | 硬编码7-8-2025 CSV | `plots/*.png`、`PPG_Processing_Summary.csv` | 与下一路径字节/SHA完全相同；重复副本 |
| `PPG_Testing_05_01_2026/Archive/7-8-2025/ptt_ppg_dataset_analysis_fingertiponly.py` | 同上 | 同上 | 现存9 PNG与9行summary对应 | 作为有实际输出关联的主副本登记；算法内容不重复计数 |
| `PPG_Testing_05_01_2026/Archive/FilteredWalkTest/FilteredWalkTest.ipynb` | base.csv原始/基线扣除/Chebyshev/Butterworth探索 | 相对`base.csv`，Timestamp/Ir2/Red2 | 仅内嵌图 | 历史探索，无固定文件输出；保存125Hz只取首个8ms间隔，数据含大量零间隔，采样率估计不可靠 |
| `PPG_Testing_05_01_2026/ptt_ppg_dataset_analysis_16July2025.py` | 400Hz Butterworth0.5–5、IR peaks、RR拒绝、HR/SDNN/SpO2组合图 | 硬编码25July25 CSV，IR/RED/可选Time | `output_plots/*_ppg_analysis.png`、`summary.csv` | summary 12行但PNG含多轮陈旧产物；SpO2含107.294与78.727等越界/异常值；未放Archive但已被`funcs.py/ppg.py`替代 |

## 3. 归档使用规则

1. `results_detector_v8`、`results_stage1`、`results_v7_3`必须分别引用对应归档生产脚本，不能强行归因给当前根版本。
2. detector fix2/3/6/8是错误修复lineage，不是四个独立科学方法；禁止重复计入方法数量。
3. 两份fingertiponly脚本SHA相同，只算一个算法实现；保留两个路径用于provenance。
4. 历史Notebook的保存输出只证明某次交互状态；出现异常或未执行时不得标记验证完成。
5. 归档文件全部保持只读；未来如复用思路，必须在final_v0新实现并显式引用lineage。

