# M0 风险登记 / Risk Register

- 状态 / Status：`complete`
- 分级 / Severity：`critical` 会使结论无效或运行阻断；`high` 会显著偏置结果/部署；`medium` 限制泛化、可复现性或解释；`low` 为文档/工程质量风险。
- 处置原则 / Rule：本表只登记；M0 不修复。进入 M3/M4 前必须把相应风险转为测试或 acceptance gate。

## 1. Critical risks

| ID | 范围 | 风险与代码事实 | 对结果/论文的影响 | 必须的处置门 |
|---|---|---|---|---|
| R-C01 | v7 setup2 | ECG、ECG peaks 与 p6 直接进入 denoiser 推理输入 | target/reference leakage；不可部署 | 永久标记历史失败，不复用该 setup |
| R-C02 | v7 detector | AE threshold 在被评估数据本身拟合；motion label 几乎退化 | detector F1/BA/AUC 无有效含义 | 新 detector 必须 train-only threshold + label audit |
| R-C03 | v8 MaskNet | 局部变量 `F` 覆盖 `torch.nn.functional as F`，后续 `F.interpolate` | 确定性运行崩溃；0 结果 | 如需复用组件，先最小 forward test；不恢复整路线 |
| R-C04 | v8/Stage-2 | 变长 peak tensor 使用默认 DataLoader collate | batch size>1 高概率阻断 | 固定长度 target 或 custom collate 单元测试 |
| R-C05 | v8/Stage-2 | phase delay 符号相反；部分 phase/a 在 holdout 上拟合 | 对齐错误且 test contamination | delay只能用 train/reference协议拟合并锁定 |
| R-C06 | Stage-2 | external holdout 用于 final epoch选择后又报告 | holdout 不再独立 | final epoch只由 train/inner-val 决定 |
| R-C07 | Heartbeat | ECG R timing 未做 ECG→PPG delay 校正，却作为 PPG peak target | 模型目标不是 PPG pulse peak | M4 target必须按数据集/record对齐并审计 |
| R-C08 | Full waveform | 没有真实运动 clean reference；多数 target 是 noisy-self/sit template | “恢复真实 clean PPG”不可辨识 | 维持 full reconstruction deprecated；只评生理下游 |

## 2. High risks

| ID | 范围 | 风险与证据 | 影响 | 后续控制 |
|---|---|---|---|---|
| R-H01 | `funcs.py` notch | 函数参数 `notch_freq`，调用传 `f0` | 启用分支 TypeError | M3 API contract test |
| R-H02 | Aboy++ | `reject_artifacts` 位置参数错位，把 `fs` 当 BPM下界 | PPI/HRV artifact rejection失真 | keyword-only参数 + reference fixture |
| R-H03 | Aboy++ window | 10 s wrapper只提交末4 s，首尾缺覆盖；`HRi`状态语义错 | coverage缺口、HR自适应错误 | coverage mask与边界测试 |
| R-H04 | IMU filter | 400 Hz下高截止520 Hz超过Nyquist | 滤波设计失败 | fs-aware cutoff validation |
| R-H05 | IMU EKF | 估计的init未传入；`use_ekf=False`仍运行EKF | 模式开关和姿态结果不可信 | parity/branch tests |
| R-H06 | v7.2 | 四组独立holdout SNR均负，HR MAE 33–38 bpm | 已证实不适合作主路线 | 保留为negative baseline |
| R-H07 | v7.4 | 记录/subject拼接后shift，跨边界构造lag样本 | leakage/错位 | 每record独立shift并保留group key |
| R-H08 | v7.4 MaskNet | `round(a*delta0)`切断梯度；peak curvature符号错误；无一对一峰配对 | 生理约束无效 | 独立loss gradient/peak matcher测试 |
| R-H09 | v8/Stage-2 mask | 同一time mask复制到所有频率，freq smooth恒0/无意义 | 宣称的频域正则未实现 | 明确time-mask或真正freq-time mask |
| R-H10 | Hybrid units | PTT样例acc很可能已为m/s²、gyro已为rad/s，代码再次换算 | IMU尺度严重错误 | M2单位registry + magnitude sanity plot |
| R-H11 | Hybrid loss | artifact L1与clean L1代数等价 | 同一proxy重复计权，不能视为双监督 | 删除重复项或定义独立监督 |
| R-H12 | Hybrid split | holdout只记录、不推理评分；whole-record ridge使用整条记录 | 无独立效果，且transductive preprocessing | train-fit causal baseline + holdout scorecard |
| R-H13 | Hybrid OLA | 边界未被window覆盖时输出0；窗口上限取最早片段 | 长记录尾部/边缘结果错误 | full-length coverage与boundary fixture |
| R-H14 | Heartbeat event metric | overlapping windows重复计算同一beat；pooled而非subject-balanced | F1/CI偏向长记录/多窗口subject | 先合并唯一beat，再subject bootstrap |
| R-H15 | Heartbeat IBI | 输出限制0.3–2 s，但reference RR未过滤；VitalDB MAE 18.27 s | aggregate被坏pseudo-label支配 | reference QC、failure status、dataset breakdown |
| R-H16 | Gate external | extra gate F1 `.4690`、AUC `.4088`，rest/motion score方向反转 | 外部不可部署 | 重新定义label/feature并做external calibration |
| R-H17 | legacy v8 | train全体先fit transform再CV；lag跨记录；IMU-only近满分 | CV高估且检测的是activity，不是PPG artifact | 只作历史偏置证据 |
| R-H18 | Dashboard/runtime | 默认bundle文件名不存在；`det_v8=None`仍可能索引；feature contract漂移 | 默认运行崩溃或 silent mismatch | 显式bundle manifest + end-to-end parity |

## 3. Medium and engineering risks

| ID | 范围 | 风险 | 控制建议 |
|---|---|---|---|
| R-M01 | v7/v7.2 | validation同时用于early stopping、threshold和报告 | nested split或锁定epoch后独立评估 |
| R-M02 | v7.4 labels | sit=clean、walk/run=artifact只是activity proxy | 增加窗口级reference或明确改名motion-state |
| R-M03 | CEEMD-lite | 实现不是标准CEEMDAN却容易被同名描述 | 论文和registry保持精确命名 |
| R-M04 | Hybrid preview | raw曲线已滤波；选择非blind；无数字指标 | 只放附录定性图，不能作性能结论 |
| R-M05 | ONNX | `.onnx.data`外部权重依赖未完整写进契约；随机tensor parity无阈值 | bundle完整性校验 + CSV端到端parity |
| R-M06 | ONNX runtime | preprocessing在core/runtime/dashboard多份实现 | 单一公共contract与golden fixture |
| R-M07 | Heartbeat ONNX | 无dynamic axes，per-record z-score非streaming，无正式runtime | M1/M9固化batch/time/normalization contract |
| R-M08 | JSON | 部分scorecard允许NaN | strict JSON writer与schema验证 |
| R-M09 | Binary evidence | M0未反序列化PT/Pickle/NPZ内容 | 如未来需要内部字段，先安全格式转换或受控读取 |
| R-M10 | Naming | 同一代存在多个目录/默认名不一致 | run manifest强制code SHA/config/data/split |

## 4. 论文风险控制 / Claim controls

- `full-waveform clean reconstruction`：禁止性能性表述。
- `motion artifact detector`：只有使用真实artifact定义时才能这样命名；activity proxy必须明示。
- `PPG peak detector`：当前P01必须改称“从PPG估计未校正ECG timing的历史尝试”。
- `holdout`：必须区分最后CV fold、inner validation、subject holdout、external dataset。
- `improvement`：proxy objective下降、视觉更平滑、模型文件存在均不等于生理性能提升。
- `deployment-ready`：必须有完整输入contract、预处理、模型、后处理、failure state和golden parity。

## 5. M3/M4 必须转化的 acceptance tests

1. 固定 synthetic/real fixtures 的滤波、peak、PPI/IBI、coverage与边界测试。
2. train-only threshold/normalizer/lag/delay，并在 subject-disjoint/external set 一次性评价。
3. 唯一beat合并后再算 event F1；同时报告每subject与bootstrap CI。
4. raw/no-denoising、high-quality-only与coarse-clean三基线并列。
5. preprocessing Python ↔ deployment runtime 的逐样本golden parity。
6. 单位、采样率、通道顺序、缺失通道、NaN和失败状态均为显式contract。

