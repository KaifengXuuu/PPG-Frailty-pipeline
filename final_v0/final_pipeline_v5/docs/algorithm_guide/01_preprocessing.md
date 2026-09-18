# 1. 信号预处理

本章说明数据怎样从一条记录变成可供后续分析或模型读取的数组。路径相对 V5 根目录；`文件:起止行` 是本版本的代码定位，不代表需要修改代码。为便于逐行审阅，连续的参数检查与结果封装合并解释，数值运算按执行次序展开。外部库调用只解释本项目实际传入的参数，不把未在仓库实现的库内部步骤写成本项目代码。

## 1.1 阅读路线、数据形状与 finalcase

直觉解释：同一条记录像八条并排画在同一张时间纸上的曲线。处理时不能把其中一条偷偷往前移，也不能把“每一秒多少个点”和“曲线有多高”混为一谈。先把曲线整理好，再切成小段；给模型的小段可以另做大小调整，但用于测量真实高度、间隔和运动大小的曲线仍保留。

代码主线是 `src/ppg_frailty/experiment.py:308-433` 的 `_fit_imu_calibrations`、`_preprocess_records`，接 `src/ppg_frailty/signal/preprocess.py:632-806` 的 `build_signal_views`。raw 主线随后到 `src/ppg_frailty/representations/raw.py:115-179`、`src/ppg_frailty/experiment.py:1703-1757`，最后到 `src/ppg_frailty/experiment.py:1950-1994`。

| 阶段 | 输入与输出 | 实际顺序或含义 |
|---|---|---|
| CSV 读取 | `N×8`，float64 | 列为 RED、IR、AX、AY、AZ、GX、GY、GZ；400 Hz；没有时间戳列 |
| 静态校准 | 同 participant 的 B 记录，ACC/Gyro 各 `N_B×3` | 先取 B 的 `[5,100)` 秒，再滤波，估计固定偏置 |
| PPG 整理 | `N×2` → `N×2` | 检查/补短缺口 → 去直线趋势 → 双向带通 |
| IMU 整理 | ACC/Gyro 各 `N×3` | 单位换算 → 减 B 偏置 → 双向低通 → 所选重力处理 |
| 原始分析视图 | PPG `N×2`；若干 IMU `N×3`/`N` 数组 | 保留400 Hz和物理单位；不逐窗标准化这些上游数组 |
| raw 分窗 | `N×8` → `W×8×2000` | finalcase 每窗5秒、步长2.5秒；每记录最多128窗 |
| raw 逐窗缩放 | 每窗有效行的八个通道 | finalcase 所有八通道分别按本窗中位数与四分位跨度缩放 |
| 可选 fold 后变换 | `W×8×T` → 同形状 | 只对六个 IMU 通道；仅训练 participant 拟合；finalcase 不执行 |
| DL 重采样 | `W×8×2000` → `W×8×320` | finalcase 到模型前才由400 Hz变为64 Hz；不改变特征分析网格 |

`W` 是实际保留窗口数，`N` 是记录样本数，`T` 是一窗的样本数。正常入口的时间由行号决定：`t[n]=n/400` 秒。调用者先为参与本轮的 participant 建立 B 校准对象，再逐条记录构建 PPG/IMU 视图；表中分窗以后的 raw 分支是在质量与运动路由之后才执行，故本章不是要求所有预处理步骤都先于质量步骤一次执行完。

finalcase 的真实选择来自 `configs/presets/finalcase.yaml:63-154`：PPG 0.2–8 Hz、三阶；ACC20 Hz/Gyro40 Hz、三阶；`sensor_filter_only_no_gravity_removal`；B校准5–100秒；逐窗 robust、`raw_imu:none`；64 Hz模型输入。`quality.mode:off` 在该文件155–167行，不会关闭读取时的物理检查、PPG 输入检查或静态校准。

## 1.2 文件读取、同步时间与缺失片段

### 1.2.1 正式 CSV 入口：检查不等于修补

直觉解释：先确认八列来自同一段完整记录，再决定是否交给后面的整理步骤。这一步是检查原纸张有没有缺字，不负责补字。

`src/ppg_frailty/pipeline.py:275-332`，`_load_record`：

1. 278–285行从同一份文件字节读出数值，跳过标题，指定 float64。
2. 286–288行确保为二维、恰好八列。
3. 291–300行对整条记录执行物理检查；即使请求只取前一小段，也是在检查整条记录之后才于301–307行截取。
4. 311–315行把前两列、接下来的三列、最后三列分别交给 PPG、ACC、Gyro；输入单位声明为 `g`、`deg/s`。

重要实现边界：`src/ppg_frailty/data/qc.py:296-313` 的 `physical_recording_qc_thresholds_v2` 把最短记录设为5秒，`maximum_nonfinite_gap_s=0.0`。因此正常 finalcase CSV 入口中的 NaN/Inf 会先被拒绝，不能因为下面有“100点插值”函数就声称生产入口一定会自动修补缺失。下面的修补算法是真实存在的可复用函数能力，注入其他合格 loader 或直接调用时可执行；本说明不改变这个入口行为。

### 1.2.2 显式时间戳检查

直觉解释：相邻两点应该间隔相同；少数轻微不齐可以接受，倒着走或明显忽快忽慢不接受。此模块只检查，不把不齐的点重新摆齐。

`src/ppg_frailty/signal/preprocess.py:88-109`，`validate_timestamp_grid`：

1. 无时间戳直接返回；有则展开为长度 `N` 的有限 float64 向量。
2. `Δt_i=t_{i+1}-t_i`，要求每项大于0。
3. 理想间隔 `d=1/f_s`，计算 `r_i=|Δt_i-d|/d`。
4. `percentile(r,99)>0.05` 时拒绝。不是“每一个间隔都必须小于5%”。

正式 CSV loader 不提供时间戳，因而依赖 manifest 的400 Hz声明。历史分块 IMU 的 `_timestamp_ok` 另检查当前块首点与上块末点，见 `src/ppg_frailty/signal/imu.py:389-411`；跨块边界误差必须不超过5%。

### 1.2.3 PPG 内部短缺口：线性补点

直觉解释：如果缺口两端都看得见，就用直尺连接两端，沿直尺补齐中间缺的点；缺口在纸张边缘或太长时，不猜它原本长什么样。

输入为 `N` 或 `N×C`，输出为修补后的 `N×C`、原始有效布尔矩阵、修补位置矩阵以及检查结果；正式 PPG 调用 `C=2`。`src/ppg_frailty/signal/preprocess.py:130-203`，`inspect_and_repair`：

1. 138–150行转 float64；一维转成单列。`valid=isfinite(source)` 保存原始缺失事实，不能因修补成功改成全有效。
2. 151行得到 `L_max=round(max_gap_sec×400)`；正式调用由 `signal.gap_repair.max_gap_samples` 转换，默认100点即0.25秒。
3. 153–178行逐列找 `[start,stop)` 的连续缺口。`_true_runs` 在111–121行于布尔数组两端补 False，再差分寻找0→1和1→0边界。
4. 180–185行执行 `np.linspace(left,right,L+2)[1:-1]`。令左端为 `a`、右端为 `b`，缺 `L` 点，则第 `j=1…L` 个补点为 `a+j(b-a)/(L+1)`。
5. 186行只在新填的位置标记 `repair_mask=True`；原始 `source_valid_mask` 不变。
6. 首尾缺口、长度超过上限或整列全缺，会使 `status='failed'`。此函数没有总缺失比例1%限制。
7. 191–200行检查补完后的严格相等平台。`_longest_constant_run` 在123–128行计算连续相等样本数；默认达到 `round(1.0×400)=400` 点拒绝。最小值/最大值出现比例只记入指标，并未在这里用阈值拒绝。

配置为 `signal.gap_repair.method:linear_inside_only`、`max_gap_samples:100`、`edge_extrapolation:false`；平台时长来自 `quality.flatline_duration_s:1.0`。`build_signal_views` 的672–676行要求质量长缺口长度与信号修补长度一致。范围：`max_gap_samples` 非负整数；0表示不修任何非空缺口。

### 1.2.4 历史 IMU 内部短缺口：另一个规则

直觉解释同样是两端拉直尺，但这条旧路线还要求整列缺字不能太多；不要把两套规则当成完全相同。

`src/ppg_frailty/signal/imu.py:424-446`，`_repair_chunk`，由 `CausalImuProcessor.process_chunk` 的552行调用：

- 输入为当前块 `N×6`，分列修补；最大缺口固定 `round(0.25×f_s)`。
- 433–434行：整列全缺，或非有限占比严格大于0.01时拒绝。
- 437行：首尾缺口或超过最大长度拒绝；439–443行使用与上一节相同的直线插值公式。
- 校准型三条 IMU 路线不调用此函数；`motion_imu.py:340-341,544-545` 直接要求输入全部有限。修改 PPG 的 `max_gap_samples` 不会修改这个历史 IMU 函数的固定0.25秒。

## 1.3 PPG 整理与两个带通方案

### 1.3.1 去除直线趋势

直觉解释：把整条曲线慢慢向上或向下倾斜的纸面扶正，让每次起伏围绕较平的地面出现；这不把每次起伏压到相同高度。

输入为补好缺口、保留原始计数大小的 `native:N×2`。`src/ppg_frailty/signal/preprocess.py:227-267`，`preprocess_ppg_pair`：253行令 `native=qc.repaired`，262行执行 `signal.detrend(native,axis=0,type='linear')`。

数学上，每列使用一条最佳拟合直线 `a+bn`，输出 `u_n=x_n-(a+bn)`；这里的“最佳”对应去线性趋势调用的最小二乘含义。项目没有手写求解 `a,b` 的循环，而是调用 SciPy。原始计数视图 `x_native` 保留，不会被 `working` 覆盖。两列独立去趋势，不把 RED 与 IR 混合。

### 1.3.2 默认带通：0.2–8 Hz、三阶 Butterworth

直觉解释：留下节奏适中的起伏，削弱过慢的漂移和过快的毛刺。先从左往右处理，再从右往左处理，避免把山峰整体推迟；但曲线边缘仍要借助额外的边缘处理，太短就无法做。

`src/ppg_frailty/signal/preprocess.py:205-225`，`design_ppg_sos`；264行执行实际滤波：

1. `signal.butter(order,[low,high],btype='bandpass',fs=400,output='sos')` 生成分段系数；默认 `order=3,low=0.2,high=8.0`。
2. `signal.sosfiltfilt(sos,working,axis=0)` 对整条记录两列执行双向处理；不先切5秒模型窗。
3. 对单个二阶系数行 `[b0,b1,b2,a0,a1,a2]`，其差分关系可写成 `a0*y[n]=b0*x[n]+b1*x[n−1]+b2*x[n−2]−a1*y[n−1]−a2*y[n−2]`。各段相接；双向步骤由 SciPy 实现。
4. 前后向结合后有效幅频响应为单向响应的幅度平方；不能把配置中的“三阶”理解为“最后只做了一次三阶滤波”。具体边缘扩展长度及初始状态使用库默认值，本项目没有另写数值常量。
5. 双向滤波抛 `ValueError` 时转为 `zero_phase_filter_insufficient_length`；不会自动切换成单向滤波。

配置键为 `signal.ppg_filter.order/low_hz/high_hz`。阶数整数1–20；`0<low<high<200`。`family='butterworth_sos'`、`phase='zero_phase'`、`short_signal_policy='reject'`、`notch_enabled=false` 是当前实现，不是另有可启用但未解释的陷波模块，见543–557行。

### 1.3.3 备选带通：0.5–5 Hz、三阶

直觉解释：把允许通过的起伏范围再收窄一点，更多慢变化和快变化会被削弱；这也可能削掉真实脉搏中的一些细节，所以它是对照方案，不是永远更好的清理。

`src/ppg_frailty/signal/preprocess.py:21-76` 注册 `butterworth_0p5_5hz_ablation`，数值为 `(0.5,5.0,3)`。实际计算仍为上一节的去趋势及双向 Butterworth，把 `signal.ppg_filter.low_hz/high_hz/order` 设为相应值。注册名称本身不会自动多跑一次，也没有按最近频带替用户匹配的行为。finalcase 使用0.2–8 Hz，不使用本分支。

## 1.4 同 participant 的静态 B 校准

### 1.4.1 选择校准记录与时间段

直觉解释：每个人先有一段坐稳的数据，用来看看设备本来就偏高多少、转动读数本来偏多少。之后该人的其他记录统一扣除这份偏移，而不是借另一个人的数据来纠正。

`src/ppg_frailty/experiment.py:308-359`，`_fit_imu_calibrations`：三个校准型重力分支都会进入此步骤。328–336行只找同 participant、role恰为B、QC为pass或pass_with_warnings的记录；按时长降序、record_id升序选择首条。没有合适 B 则该人的预处理失败；不是偷偷改为无校准。

`src/ppg_frailty/signal/motion_imu.py:312-394`，`fit_motion_imu_calibration`：

1. 332–337行先换算物理单位；默认 ACC 的 `g` 乘9.81，Gyro 的 `deg/s` 乘 `π/180`。
2. 342–345行取 `[round(5×400),round(100×400))=[2000,40000)`，共38000点；原记录必须覆盖该区间，且区间至少16点。
3. 346–351行在已经截出的静态段上分别做三阶20 Hz、40 Hz双向低通，而不是先滤整条 B 再截出该区间。边缘条件因此由这个区间决定。

配置 `signal.imu.calibration_start_s:5.0`、`calibration_stop_s:100.0`；范围 `0≤start<stop`，还受实际记录长度约束。通用函数还支持明确声明的 `PTT_SIT_STATIC_CALIBRATION`，但普通 frailty 实验调用者仍只选B。

### 1.4.2 去离群后的平均值

直觉解释：先找到大家的中间位置，再把离中间太远的少量点暂时放到一边，剩下的求平均。三根轴各自决定哪些点太远，不要求三根轴扔掉完全相同的时刻。

`src/ppg_frailty/signal/motion_imu.py:235-246`，`_robust_mean`：每列 `m=median(x)`、`d=median(|x−m|)`。若 `d≤1e−12`，直接用全部点均值；否则保留满足 `|x−m|/d<3.5` 的点并求均值。这里的 `d` 没有乘1.4826，也不是下一节模型标准化里的四分位跨度。阈值1e−12和3.5是函数里的固定数值，不是 YAML 可调项。

### 1.4.3 初始倾斜与固定偏置

直觉解释：用坐稳时三根轴的平均值，看“向下”的方向在设备坐标中朝哪边；由这个方向画出一支长度固定的箭头。平均读数与这支箭头之间的差，作为设备的固定偏移。它不是能独立识别所有设备误差的万能校准。

`src/ppg_frailty/signal/motion_imu.py:248-264` 与352–365行：令三轴稳健均值为 `(ax,ay,az)`，

`φ=atan2(ay,sqrt(ax²+az²))`

`θ=atan2(−ax,sqrt(ay²+az²))`

`g_B=g*[−sin(θ)cos(φ), sin(φ), cos(θ)cos(φ)]`，默认 `g=9.81 m/s²`。

随后 `b_acc=mean_robust(acc)−g_B`，`b_gyro=mean_robust(gyro)`。`φ,θ`、两个三维偏置和来源身份写入校准对象。质量指标只记录重力模长误差、Gyro逐轴均方根和样本数；365行明确 `quality_threshold_applied=False`，不能把这个模块描述成自动验证“这段必定真的静止”。

拟合范围是本人的B，不使用分类标签、不用跨人统计。它允许用留出 participant 自身在部署时也必须提供的静态校准记录，性质不同于拿留出人的数据去拟合跨人标准化器。

## 1.5 校准型 IMU 共用的单位换算、减偏置与滤波

直觉解释：先统一尺子刻度，再把静止时自带的偏移扣掉，然后磨平细小的快速抖动。是否再扣掉“向下”的分量，是下一步另选的操作。

`src/ppg_frailty/signal/motion_imu.py:515-563`，`_prepare_si_inputs`，输入ACC、Gyro均为 `N×3`，至少16行，全有限，同 participant 校准；输出同形状 float64 SI单位数组：

1. 536–541行换单位。`_convert_profile_acceleration` 的266–282行支持 `g→×g`、`mg→×g/1000`、`m/s2`/`m/s^2→×1`；默认 g常数9.81。`signal/imu.py:88-94` 的 `convert_gyro` 支持 `deg/s→×π/180`、`rad/s→×1`。
2. 546–547行：`a_c=a_SI−b_acc`，`ω_c=ω_SI−b_gyro`，对整条运行记录逐行广播减三维常量。
3. 548–549行调用 `_zero_phase_lowpass`（225–233行）：`butter(order,cutoff,lowpass,fs,sos)` 后 `sosfiltfilt(axis=0)`。ACC默认20 Hz，Gyro默认40 Hz，阶数3。
4. 得到 `a_f,ω_f`。短输入导致双向滤波失败时抛错误；没有“直接跳过滤波”或“用单向代替”的暗中分支。

配置 `signal.imu.sensor_lowpass_acc_hz:20.0`、`sensor_lowpass_gyro_hz:40.0`、`sensor_filter_order:3`。截止频率严格在 `(0,200)`；阶数整数1–20。改变过滤前后减偏置次序或把双向改成单向会改变数值，本章仅解释现状。

## 1.6 IMU 重力/方向的五个互斥算法

配置入口 `signal.imu.gravity_method` 在 `src/ppg_frailty/signal/preprocess.py:356-491` 解析。未显式填写时默认 `profile_a_lowpass_0p3hz`；这与 finalcase 显式选择的 `sensor_filter_only_no_gravity_removal` 不同。

### 1.6.1 finalcase：sensor_filter_only_no_gravity_removal

直觉解释：把设备误差和快速毛刺整理好，但保留“地球一直拉着它”的部分。代码里为接口留出的方向栏填零，不是测得设备一直摆正。

`src/ppg_frailty/signal/motion_imu.py:648-740`，`_preprocess_motion_profile(mode='no_gravity')`；外壳入口790–812行：

1. 661–670行照常执行上一节的 B偏置校准和传感器滤波。
2. 706行 `gravity=zeros_like(acc_filtered)`；707行 roll、pitch都是长度N的零占位数组。
3. `_motion_result` 的613行 `dynamic=a_f−gravity`，故这里名叫 `dynamic_acc_mps2` 的数组实际仍包含重力，严格等于 `a_f`。
4. 709–712行元数据明确重力估计/扣除关闭，输出语义是含重力的已校准已滤波加速度。

不读取重力低通或 EKF数值参数；没有重力滤波可在该分支暗中生效。finalcase 正是本分支。`no_gravity` 不等于 `no_calibration`，也不等于完全原始加速度。

### 1.6.2 profile_a_lowpass_0p3hz：缓慢变化作为重力

直觉解释：把加速度曲线看成缓慢移动的底座加上较快的动作。先画出慢慢变化的底座，再从原曲线减掉它。慢动作也可能被底座带走，因此这不是对任意运动都精确的物理分离。

同函数689–703行，外壳766–788行：

1. 对共用的 `a_f` 逐轴再次双向低通：`g_hat=LPF(a_f,0.3 Hz,order=4)`。
2. 696–697行逐点把 `g_hat` 代入1.4.3的两个角度公式，仅为输出方向描述。
3. `_motion_result` 613行 `a_dyn=a_f−g_hat`。此低通的三维长度不强制等于9.81。

配置 `gravity_lowpass_hz:0.3`、`gravity_filter_order:4`。注意这里四阶双向，与1.6.5的历史二阶单向不是同一个算法；不能只看都叫0.3 Hz就互换。无训练参数、无标签、整条记录内确定性运算。

### 1.6.3 calibrated_roll_pitch_ekf：静态初值加五状态跟踪

直觉解释：手里有两种线索：转动读数能告诉你刚才转了多少，但时间久了会偏；加速度能提示“向下”在哪，但人一动它也会被拉歪。代码每走一步，先按转动猜方向，再按两种线索各自有多不可靠，把猜测往加速度方向拉一点，同时修正转动读数的残余偏移。

`src/ppg_frailty/signal/motion_imu.py:399-513`，`_run_roll_pitch_ekf`；671–688行接入输出。输入为共用预处理后的 `a_f,ω_f:N×3`；状态 `x=[φ,θ,bx,by,bz]`，两个角度用弧度，后三项为残余角速度偏置 `rad/s`。

步骤A：初始化（409–426行）。`φ0,θ0` 来自B；残余偏置初值0。`P0=diag(1,1,0.5,0.5,0.5)`；`Q=diag(5,5,0.05,0.05,0.05)/f_s`；`R0=diag(0.5,0.5)`；`dt=1/f_s`；`H=[[1,0,0,0,0],[0,1,0,0,0]]` 只直接观察角度。

步骤B：按转动预测（427–446行）。从 `i=1` 开始，先 `ω=ω_f[i]−b[i−1]`，写为 `(gx,gy,gz)`；使用上一时刻角度：

`dφ/dt=gx+gy*sinφ*tanθ+gz*cosφ*tanθ`

`dθ/dt=gy*cosφ−gz*sinφ`

`φ_pred=φ+dt*dφ/dt`，`θ_pred=θ+dt*dθ/dt`，`b_pred=b`。当 `|cosθ|<1e−6` 时停止，避免该角度表示在近竖直处出现除零。

步骤C：传播不确定度（447–469行）。令 `s=sinφ,c=cosφ,t=tanθ,u=1/cos²θ`，矩阵F的非恒等部分为：

```text
F 第一行 = [1+dt*(gy*c*t-gz*s*t), dt*(gy*s*u+gz*c*u), -dt, -dt*s*t, -dt*c*t]
F 第二行 = [dt*(-gy*s-gz*c), 1, 0, -dt*c, dt*s]
F 第三至第五行 = 对偏置的单位阵对应三行
P_pred = F @ P @ F.T + Q
```

这些元素就是预测公式对状态各量的局部变化率，源码逐项列出，不能把这里的F改成单位阵仍声称算法不变。

步骤D：由加速度得到方向线索并调权（470–482行）。`z=[roll_from_acc,pitch_from_acc]` 使用1.4.3公式。计算

`d=max(0,||a_f[i]||−g)/g`，`scale=1+α*d`，`R=diag(R0_diagonal*scale)`，默认 `α=3`。

这里是单侧 `max(0,模长−g)`，并非 `abs(模长−g)`；模长低于g不会通过这条规则增加R。

步骤E：融合（483–499行）。`wrap(v)=atan2(sin v,cos v)` 把角差绕回 `[-π,π]`，避免+π与−π被当成相差一整圈。

```text
e = wrap(z - H @ x_pred)
S = H @ P_pred @ H.T + R
K = solve(S, H @ P_pred).T
x_new = x_pred + K @ e
P_new = (I-KH) @ P_pred @ (I-KH).T + K @ R @ K.T
P_new = (P_new + P_new.T)/2
```

更新后的两角再次wrap；若状态非有限或P最小特征值低于−1e−9报错。此处采用源码写出的 Joseph形式，不是简单 `(I−KH)P`。输出残余偏置轨迹只用于跟踪内部估计和诊断；最终输出的 Gyro通道仍是共用 `_prepare_si_inputs` 给出的 `ω_f`，不会再把这一残余轨迹逐点扣一遍。

步骤F：方向转回重力（681行）。由 `φ,θ` 使用1.4.3的三维重力公式，模长由g确定；再 `a_dyn=a_f−g_hat`。没有磁力计，不提供绝对朝向/yaw校正。

可调项均在 `signal.imu`：`process_covariance_diagonal_per_second:[5,5,0.05,0.05,0.05]`，`observation_covariance_diagonal_rad2:[0.5,0.5]`，`initial_covariance_diagonal:[1,1,0.5,0.5,0.5]`，`dynamic_observation_scale:3.0`，`gravity_mps2:9.81`。协方差对角各项需正且有限，动态系数非负；向量长度分别5、2、5。默认数值来自57–78行。本算法不训练分类标签，但使用整条记录的双向预滤波，因此完整链路不是实时纯因果处理。

### 1.6.4 quaternion_error_state_ekf：无 B 预校准的历史方向跟踪

直觉解释：没有先坐稳的一段数据时，代码尝试直接从眼前读数猜设备的摆向，再随着每次转动修正。它需要一段时间建立信心；如果线索冲突太大，就暂时只信转动或宣布估计失效，而不是编出一个看起来平滑的方向。即使它觉得自己稳定，也不等于已证明设备所有方向和偏差都能辨认出来。

外部配置名由 `preprocess.py:752-766` 映射为内部 `no_precalibration_ekf`。实现位于 `src/ppg_frailty/signal/imu.py:193-387` 的 `NoPrecalibrationEskf`；整条路先使用本文件75–94行的单位换算及170–191行单向滤波，不使用1.5的B校准或双向滤波。这里 `STANDARD_GRAVITY=9.80665`，与校准型路线9.81不同。

几何小工具（96–168行）：

- `skew(v)` 构造矩阵使 `skew(v)u=v×u`。
- `quat_normalize(q)=q/||q||`；范数非有限或小于1e−15失败。q按 `[w,x,y,z]` 排列。
- `quat_multiply` 在113–121行逐项实现 Hamilton乘积，合成先后转动。若左侧为 `(w₁,x₁,y₁,z₁)`、右侧为 `(w₂,x₂,y₂,z₂)`，则输出四项为 `w=w₁w₂−x₁x₂−y₁y₂−z₁z₂`，`x=w₁x₂+x₁w₂+y₁z₂−z₁y₂`，`y=w₁y₂−x₁z₂+y₁w₂+z₁x₂`，`z=w₁z₂+x₁y₂−y₁x₂+z₁w₂`。左右交换一般会改变结果，代码始终按已给出的顺序相乘。
- `quat_exp(v)`：`α=||v||`；小于1e−12时归一化 `[1,v/2]`，否则 `[cos(α/2),(v/α)sin(α/2)]`。
- `quat_to_rotation` 在135–143行先归一化q，然后展开完整3×3旋转矩阵R。第一行是 `[1−2(y²+z²),2(xy−wz),2(xz+wy)]`；第二行是 `[2(xy+wz),1−2(x²+z²),2(yz−wx)]`；第三行是 `[2(xz−wy),2(yz+wx),1−2(x²+y²)]`。R表示设备到参考坐标的旋转；方向预测从参考坐标回到设备坐标使用R的转置，不可交换。
- `quat_from_two_vectors` 在145–157行将两方向单位化；一般为 `normalize([1+dot,cross])`；几乎相反时找一根垂直轴作半圈转动。
- `tangent_basis` 在159–168行选择与方向最不平行的坐标轴，连续两次叉乘构造垂直于该方向的两根单位轴，得到 `B:2×3`。

逐样本数学：

1. 初始化（212–229、261–272行）。首次 `0.5g≤||a||≤1.5g` 时，把加速度方向转到 `[0,0,1]`；偏置0。不确定度分为倾斜20°、绕重力方向180°、角速度偏置5°。`P角=tilt²(I−ddᵀ)+yaw²ddᵀ`；`P偏=I*bias²`，d是初始加速度单位方向。
2. 预测（274–291行）。初始化刚成功的那个样本不推进q/P（275行 `if not initialized_now`）；后续样本才执行本步。`ω=gyro−bias`；`q←normalize(q⊗exp(ωdt))`。六维误差状态为小角度三维和偏置三维。`F=[[-skew(ω),−I],[0,0]]`；`Φ=I+Fdt+(Fdt)²/2`。设 `v_g=gyro_noise_density²`、`v_b=gyro_bias_random_walk²`：`Qaa=I(v_g dt+v_b dt³/3)`，`Qab=Qbaᵀ=−I v_b dt²/2`，`Qbb=I v_b dt`；`P←ΦPΦᵀ+Q`。
3. 更新节奏（295–319行）。仅当全局样本号可被4整除时观察加速度方向。`d_pred=R(q)ᵀ[0,0,1]`、`d_obs=a/max(||a||,1e−15)`；`ρ=abs(||a||/g−1)`；`η=||a−a_previous_sample||/(g dt)`。
4. 观察噪声系数 `s=clip(1+(ρ/0.05)²+(η/2)²,1,100)`；s≥25只标记为downweighted。`e=B(d_obs−d_pred)`；`H=[B*skew(d_pred),0]`；`R_obs=(5°换弧度)²*s*I₂`；`S=HPHᵀ+R_obs`；`NIS=eᵀ solve(S,e)`。
5. 仅在模长仍在 `[0.5g,1.5g]` 且 `NIS≤13.8155` 时更新（320–333行）。`K=solve(S,HP)ᵀ`，`δ=Ke`；`q←normalize(q⊗exp(δ前三维))`；`bias←bias+δ后三维`。P先用Joseph形式，再用 `J=diag块(I−skew(δ前三维)/2,I)` 做 `P←JPJᵀ`。
6. 数值整理（334–341行）。对称化P；特征值小于−1e−10失败，介于该值与1e−15间则提升到1e−15并重新组合矩阵。
7. 有效性（235–240、343–387行）。倾斜不确定度 `σ=degrees(sqrt(max_eigenvalue(B P角 Bᵀ)))`。至少经过0.5秒、接受20次更新且σ≤10°才能进入跟踪；q/P非有限或偏置任一轴绝对值大于0.35 rad/s时失效。未曾进入跟踪则保持 `initialization_pending`；已进入跟踪后，若连续仅预测超过2秒或σ>20°，也失效。失效进入 `no_estimate` 并保持到显式重置。其他有效情况为跟踪或仅预测。`g_hat=R(q)ᵀ[0,0,9.80665]`。

参数默认值在 `imu.py:19-32`：角速度噪声密度0.002、偏置随机变化0.0002、观察角噪声5°以及上述阈值。它们属于 `EskfConfiguration` 的 Python层设置；普通 `signal.imu` YAML当前只暴露传感器低通参数，不能虚构所有这些数值已有 CLI键。

整条链的状态管理在 `imu.py:467-727`：滤波状态、方向状态、前次动态加速度及时间戳跨chunk保存；处理前复制估计器，整个chunk成功才在710–719行提交。首次/重新初始化的无效样本保持NaN/False；raw窗口若触及这些无效样本会被丢掉，不在输入里补零冒充有效。604–608行额外输出 `gravity_confidence=exp(−σ/10)`，只在方向有效且σ有限时使用，否则0；这不是经过分类标签校准的正确概率。模型路线的“无B静默校准消融”不能仅凭此历史函数存在便宣称与 finalcase 的部署和比较协议已完成。

### 1.6.5 low_pass_0p3hz：历史单向低通重力

直觉解释：只看已经走过的曲线，慢慢跟随它的底座，不回头利用未来的数据。这适合逐块推进，但底座会有跟随迟滞，与双向查看整段曲线的结果不同。

外部名由 `preprocess.py:759` 映射为内部 `lpf_0p3`。`src/ppg_frailty/signal/imu.py:170-191,625-639`：

1. 输入先按9.80665换单位并做 ACC20/Gyro40 Hz三阶单向过滤。
2. `_causal_filter_axes` 用 Butterworth SOS；首块 `zi=sosfilt_zi(sos)*source[0]`，以后使用前块结尾状态；`sosfilt(...,zi=state)` 同时返回输出和下一状态。
3. `g_hat=causal_LPF(a_f,0.3Hz,order=2)`，`a_dyn=a_f−g_hat`；有限重力向量就标记该位置可估计。
4. 没有B校准，无方向融合，无固定模长约束；动态加速度差分仍需要相邻有效样本。

配置 `signal.imu.gravity_lowpass_hz:0.3`、`gravity_filter_order:2`。独立 API `estimate_gravity_lpf` 在790–819行可给缺省的零Gyro占位，因为此重力分支不用Gyro推方向；标准八通道pipeline并不因此省略Gyro输入。

## 1.7 动态加速度、角速度大小与变化速度

直觉解释：三根轴可组成一支空间箭头；箭头长度表示总体大小，与它朝哪边无关。相邻两时刻箭头差得越多，表示动作变化越突然。

校准型输出在 `src/ppg_frailty/signal/motion_imu.py:602-646`，`_motion_result`：

`a_dyn[n]=a_f[n]−g_hat[n]`

`j[0]=0`，`j[n]=(a_dyn[n]−a_dyn[n−1])*f_s`，对应614行 `diff(...,prepend=dynamic[:1])*fs`。

`A_mag=sqrt(ax_dyn²+ay_dyn²+az_dyn²)`；`Omega_mag=sqrt(gx²+gy²+gz²)`；`J_mag=sqrt(jx²+jy²+jz²)`。九通道次序为三轴动态加速度、三轴角速度、A_mag、Omega_mag、J_mag；单位分别m/s²、rad/s、m/s³。正式raw分类输入只取前六个IMU轴，不是自动把三个大小量也塞入八通道模型。

历史因果路线的 `_jerk`（`imu.py:448-465`）不同：当前chunk第一点有上块末点且两者有效才做跨块差分；初始第一点没有前驱，保持NaN。相邻任一点无效也不填差分。因此其 `imu_valid_mask` 是方向有效与差分有效共同成立，不等于只要六轴原始值有限就有效。

`compare_ekf_lpf_gravity`（`imu.py:831-879`）只是描述性比较：共同有限位置上对差值d算 `RMSE=sqrt(mean(d²))`、`MAE=mean(abs(d))`、逐轴RMSE及重力平均模长。它不依据哪个指标更小自动选算法，`selection_performed=False`。

## 1.8 三种 PPG 视图与可选去伪差视图

直觉解释：同一段曲线留几张用途不同的副本。一张留原本的高低位置，一张去掉慢漂和毛刺，一张可以进一步处理来找节奏。最后一张若改动了山形，就不能再拿它量原山的高度和宽度。

`src/ppg_frailty/signal/preprocess.py:794-804` 创建；`src/ppg_frailty/signal/views.py:22-157`，`CanonicalSignalViews`：

- `x_native`：仅通过PPG输入修补后的原计数，未去趋势/带通；用于需要基线的量。
- `x_filter`：去趋势和带通后的计数波形；`x_analysis` 属性41–43行固定返回它。
- `x_analysis_rate`：构建时是 `x_filter.copy()`；45–59行要求仍与x_filter逐值相等。名称像动态分析结果，但本字段本身不会被非恒等reducer覆盖。
- `x_ar`：可选reducer独立输出，`with_artifact_result` 的118–157行绑定。identity必须与x_filter相等；非identity标记 `ARTIFACT_RATE_ONLY`、`rate_only=True`、`q_morph_state='not_applicable'`，并携带逐点有效mask。
- 真正取节奏波形时使用 `analysis_signal`（78–87行），非identity走x_ar，其余走x_filter；raw八通道窗口使用 `x_analysis`，所以仍取x_filter。

reducer失败会报错而不是假装直接曲线是成功结果。以上是数据分流规则，reducer各算法在质量与运动章节逐项说明。不同视图仍在相同400 Hz网格；这里没有额外重采样或额外滤波。

## 1.9 分窗：按时间切片，而非随机抽样

### 1.9.1 秒数变样本数与完整窗

直觉解释：拿固定宽度的框盖在时间纸上，每次平移相同距离。框的宽度和移动距离都必须恰好对应整数个点，不能每走一步悄悄多半个点。

`src/ppg_frailty/data/windows.py:59-67,106-163`，`WindowPlan._sample_count/plan`：`L=round(window_seconds*f_s)`，`H=round(hop_seconds*f_s)`；原乘积与整数之差必须在1e−9内且L/H为正。

默认profile来自 `src/ppg_frailty/module_registry.py:1080-1099`：engineering10秒/2秒；raw5秒/2.5秒。finalcase raw在400 Hz下是L2000、H1000，不是在64 Hz下先切窗。

### 1.9.2 左对齐规则网格

直觉解释：第一个框从纸张最左边开始，只留下整个框都落在纸内的位置。

`windows.py:126-130`：起点 `0,H,2H,…≤N−L`。配置 `windows.<profile>.end_alignment:left_start_regular_grid`，由 `module_registry.py:1192-1195` 映射到内部 `start`。若未启用尾部补零，最后不足一窗的部分不会单独形成窗口。

### 1.9.3 追加最右完整窗

直觉解释：先按左边开始排框，再检查纸张最右端是否已经被最后一个框刚好盖住；没有的话，加一个贴住右端的框。因此最后两框之间的间距可以比平常小。

`windows.py:131-137`：先生成上一节网格；若最后起点不等于 `N−L`，追加该起点。配置 `include_right_aligned_if_distinct`，finalcase选择本项。它不是把全部网格重新右移。

### 1.9.4 右侧对齐：低层 API 能力

直觉解释：从最右边贴一个框，再往左倒着排，最后恢复从左到右的顺序。

`windows.py:142-145` 内部 `end` 实现 `sorted(range(N−L,−1,−H))`。当前正式配置 `normalize_window_config` 只接受左起点和追加右窗两种名称（`module_registry.py:1119-1120`），所以此项是 `WindowPlan` Python API能力，不是已有 YAML选项。

### 1.9.5 短记录与尾部右补零

直觉解释：只有明确允许时才把不够长的框右边补成空白；同时告诉后续模块哪些位置是空白，不能把空白算作真的测量。

`windows.py:114-124,138-141,165-185`：

- 短于一窗时，`reject` 抛 `ShortRecordError`；`pad_right` 产生从0开始的单窗，真实长度N，补到L。
- 正常长度且启用尾部补零时，在规则网格最后起点加H；若仍小于N，只追加这一窗，不循环追加任意多个尾窗。
- 每候选窗须满足 `valid_length/L≥min_valid_fraction` 才保留。
- `padding_mask` 的真实前缀是False、补零后缀True；raw输出 `valid_mask` 正好取反。

YAML `padding` 四项映射见 `module_registry.py:1196-1204`：`none_complete_windows_only`、`right_zero_pad_short_records`、`right_zero_pad_tail`、`right_zero_pad_short_records_and_tail`。尾部补零只支持左起点规则网格；engineering不支持补零。正式配置 `min_valid_fraction` 在 `(0,1]`，默认1.0；若开启padding却仍保留1.0，不完整窗仍会被过滤。低层 `WindowPlan` 默认0.0是兼容API默认，不能误写成生产默认。

`extract_window`（201–221行）复制 `[start,end)` 并用明确的 `pad_value` 右补。raw构建器不直接依赖先补再缩放：它先标准化有效前缀，后缀保留0，避免补值影响中位数。

### 1.9.6 数量上限与比例上限

直觉解释：一条记录太长时，在整段进度上均匀挑几个框，不只是取最前面的，也不根据标签挑“好看”的片段。

`windows.py:154-163,188-198`：若候选数M，上限K，则索引 `round(j*(M−1)/(K−1)),j=0…K−1`；K=1特殊取索引0。比例上限r先变成 `K=max(1,ceil(M*r))`。M≤K不删窗。顺序保持，且检查无重复。

配置 `cap_per_file` 正整数或null；`cap_fraction_per_file` 为 `(0,1]` 或null，二者互斥。raw默认/finalcase128，engineering默认null。这个均匀上限在raw后续“丢无效IMU窗”之前应用，不会在丢窗后回头补满128。

## 1.10 模型窗口的缩放：三个前变换、三个后变换

### 1.10.1 共同位置与默认值

直觉解释：只给模型的副本换一把尺子，让一条高大的曲线和一条矮小的曲线能比较形状；用于量真实高度的原纸不改动。

`src/ppg_frailty/representations/raw.py:138-174`：取x_filter两列、dynamic_acc三列、Gyro三列，得到 `N×8`。每窗口先确认有效IMU行和有限值，再调用 `_normalize_dl_window`。结果转置为 `8×T` 并转float32；补零区不进入拟合。

`src/ppg_frailty/normalization.py:73-89` 默认如下。最容易误读的一点：名为 `raw_ppg` 的选项实际控制全部八个DL通道，不只控制PPG；`raw_imu` 是之后可选的六轴后变换。

| 完整配置键（前缀 `signal.normalization.`） | 默认值 | 可调范围/形式 |
|---|---|---|
| raw_ppg | per_window_robust | per_window_robust / per_window_standard_zscore / none |
| raw_imu | none | outer_train_robust / outer_train_mean_std / none |
| iqr_fallback | standard_deviation_then_finite_one | 同值 / median_absolute_deviation_then_finite_one / finite_one |
| clip_after_scale | [-8,8] | null或两个有限且递增数 |
| robust_iqr_divisor | 1.349 | 正有限数 |
| mad_consistency_divisor | 0.6744897501960817 | 正有限数 |
| scale_epsilon | 1e−8 | 正有限数 |
| standard_ddof | 0 | 非负整数；样本数不足时按退化规则处理 |

### 1.10.2 per_window_robust：每窗中位数与四分位跨度

直觉解释：把该小段中间的高度移到零，再以中间一半点的展开宽度当尺子。少数特别高的尖点不容易左右这把尺子；最后把过分高低的位置截在固定边界内。

`raw.py:90-111`，逐通道有效样本 `x`：

1. `m=median(x)`；`q25,q75=percentile(x,[25,75])`。
2. `s=(q75−q25)/1.349`。这里没有给分母加epsilon。
3. 当s非有限或不大于1e−8，按1.10.5求备用s；备用仍不可用时令s=1。
4. `z=(x−m)/s`；非有限z替0；若clip非null，逐值截到范围内，默认 `[-8,8]`。

每一窗、每一通道独立计算，窗口间不共享中心/尺度。重叠窗口的同一原始点可能因所在窗的中心/尺度不同而得到不同z。这不是泄漏标签，因为只使用该窗自身信号；也是可在确定性缓存中保存的结果。finalcase使用本项。

### 1.10.3 per_window_standard_zscore：每窗均值与标准差

直觉解释：把整段的平均高度挪到零，用点到平均高度的通常散开程度做尺子。特别高的点会较明显地影响这把尺子。

`raw.py:86-89,103-111`：`m=mean(x)`；`s=sqrt(sum((x−m)²)/(n−ddof))`，默认ddof0。当 `n≤ddof` 时s设NaN，之后统一回退为1；其他过小/非有限s也回退1。最后计算 `(x−m)/s`、非有限变0、可选clip。它不调用IQR的备用算法。

### 1.10.4 none：不做每窗缩放

直觉解释：不给这张模型副本换尺子，保持输入的高低数值；但仍会切窗并按模型需要转成float32。

`raw.py:83-85` 直接返回float64副本。该提前返回也跳过clip，不能说none仍会裁剪到±8。物理信号在之前已经滤波/校准，不是回到CSV原值。

### 1.10.5 robust 尺度退化的三个备选算法

直觉解释：如果曲线中间一半挤在同一条线上，原来的尺子会变成零长；这时另取一把尺子，而不是拿零做除数。

`raw.py:59-73` 与95–107行提供以下三个独立选择。

#### 1.10.5.1 standard_deviation_then_finite_one

直觉解释：中间一半没展开，就改看全部点通常散开多远。代码71–73行使用本通道标准差（ddof可配）；样本不足产生NaN，再由103–107行回退1。这是finalcase默认备用方式。

#### 1.10.5.2 median_absolute_deviation_then_finite_one

直觉解释：先看每个点离中间高度有多远，再取这些距离的中间值作为另一把尺子。代码68–70行 `s=median(|x−median(x)|)/0.6744897501960817`；仍非有限或≤epsilon则由103–107行变为1。

#### 1.10.5.3 finite_one

直觉解释：不另量尺子，直接使用固定单位长度。代码66–67行直接返回每通道s=1。

共同回退1并不是把整条输出变成0；输出仍为 `x−center`，只有原本恒定且中心相同的曲线才全0。

### 1.10.6 outer_train_robust：六轴训练折后缩放

直觉解释：如果另外启用这一层，就把训练人的所有可用小段放在一起，为六根运动曲线各准备一把共同尺子。之后拿这六把尺子处理留出的人，不从留出人的曲线重新量尺子。

`src/ppg_frailty/representations/imu_transform.py:151-240`，`fit_fold_imu_channel_transform`：

1. 164–168行要求拟合participant属于outer_train且不属于outer_oof。
2. 174、182–184行先选训练窗口，再只取有效mask中的各轴样本；重叠窗重复出现的样本会重复计入，不额外去重或按人等权。
3. 196–210行每轴中位数中心、IQR/1.349尺度，退化时采用同样三个备用方式；213行最终不可用则尺度1。
4. 保存六维center、scale、有效数量和拟合participant。`apply_fold_imu_channel_transform` 的243–260行用 `(tensor[:,2:,:]−center)/scale`；PPG两通道不动，padding置0，结果float32，不再clip。

调用者 `experiment.py:1734-1747` 传入的是已经做过1.10.2/1.10.3/1.10.4的raw窗口，而非原物理IMU；因此启用本项可能是第二次缩放。它不能放进跨fold共用的确定性预处理cache。

### 1.10.7 outer_train_mean_std：六轴训练折均值/标准差

直觉解释与上一节相同，只是共同尺子改用全部训练点的平均高度和散开程度，不用“中间一半”。

`imu_transform.py:192-195,213,243-260`：在相同训练/有效掩码范围内 `center=mean`、`scale=std(ddof)`，不足或退化则1；应用方法与上一节相同。没有每participant先求均值再平均的额外步骤。

### 1.10.8 raw_imu:none：不加六轴后变换

直觉解释：前面给八条曲线各自处理好后，不再为六根运动曲线多换一次尺子。

`experiment.py:1726-1732` 直接跳过拟合器，记录不适用；不是调用一个仍会偷偷拟合训练集的对象。finalcase为本项，八通道仍已逐窗robust缩放。底层拟合函数若直接传none也有center0、scale1的恒等形式（`imu_transform.py:188-191`），与生产调用者直接不调用应区分。

## 1.11 重采样：不同入口不能混用

### 1.11.1 finalcase 模型输入：每个有效窗口400→64 Hz

直觉解释：把曲线画成更少的点，先削弱会在稀疏点阵上“看错节奏”的快速起伏，再按新间距取点。每个小窗独立处理；它不是先把整条记录变稀再切窗。

`src/ppg_frailty/signal/resample.py:76-151`，`prepare_configured_dl_input`：输入float32 `W×C×T` 和布尔 `W×T`。配置来自 `signal.dl_resampling`：默认方案须由resolved config提供；finalcase明确 `enabled:true,target_fs_hz:64,method:polyphase_anti_alias,preserve_feature_grid_hz:400`。

1. 100行 `_audited_ratio`（15–20行）把 `target/source` 化为分母≤10000的有理数u/d，并要求还原频率误差≤1e−12。400→64得到4/25。
2. 102–107行由mask求每窗有效长度v；mask必须是前v点True、其后False，且v≥2。
3. 109–113行固定输出长度 `T'=round(T*target/source)`，创建全零float32输出与全Falsemask。finalcase T2000对应T'320。
4. 114–126行对每窗的真实前缀调用 `signal.resample_poly(valid,up=u,down=d,axis=-1,window=('kaiser',5.0),padtype='constant')`。概念上是插入更密点阵、用低通整形、再按间隔保留；实际系数由SciPy生成，本代码没有手工设计其全部抽头。
5. `v'=min(T',round(v*target/source))`；复制 `min(v',实际输出长度,T')` 点，设置对应mask True。允许库实际长度和四舍五入目标相差至多1；其余位置保持零/False。
6. u=d时不调用重采样，直接用有效前缀。`enabled:false` 且无case_id时，`experiment.py:1953-1956` 连此函数也不执行，并要求目标保留400 Hz。

形式范围是 `0<target≤source=400`，还必须使输出至少2点并满足有理数精度；不是只允许100/160/200。PPG峰时间、形态和工程特征仍以400 Hz计算。归一化与重采样不能任意交换：重采样会改变各窗分位数，二者一般不交换。

### 1.11.2 命名的 fixed-kernel 对照重采样

直觉解释：改变每秒的点数，但模型量“相邻多少个点”的尺子保持原来的点数。因此同样长的尺子会覆盖不同的真实时间，这正是该对照要研究的变化。

`src/ppg_frailty/models/time_scale.py:60-67,147-224` 的注册与 `prepare_fixed_kernel_dl_input`：reference400 Hz/5秒、context10秒、100/160/200 Hz、dilation2是已命名条件。输入有效前缀、Kaiser5、常数边界、float32、长度取整和1.11.1一致；区别是输出长度与原输入长度受case对象固定，比例用整数频率构造，模型卷积核样本数不随频率缩放。

通过 `signal.dl_resampling.case_id` 路由，`experiment.py:1957-1970` 只允许raw。本节不把这些“注册条件”当作自动完成的训练，也不把finalcase64 Hz误写成这一命名case；finalcase没有case_id。

### 1.11.3 单/双维独立 DL 视图：线性边缘延伸

直觉解释：另复制一条不同点密度的曲线；在两端暂按边缘走向延长，而不是在框外补零。

`src/ppg_frailty/signal/resample.py:163-194`，`resample_dl_view`：输入一维/二维非空有限float64，源频率必须400 Hz；调用 `resample_poly(...,axis=axis,padtype='line')`，默认axis−1。未显式传window，因此使用库默认滤波窗；输出float64。该独立API允许任意正目标频率，不包含正式DL配置的“不高于400”限制。它不是finalcase使用的三维窗口函数，不能因同名“resample”替换二者。

### 1.11.4 外部多通道同步重采样

直觉解释：八条或更多并排曲线一起换一张时间纸，每一列都在同一批位置生成新点，避免各列各自改点数后错位。

`src/ppg_frailty/signal/resample.py:226-275`，`resample_synchronized_channels`：输入 `N×C` float64，列名唯一且有序；源/目标频率正有限，目标默认400 Hz。按同一个u/d对axis0执行 `resample_poly(...,padtype='line')`。预期行数 `ceil(N*u/d)`；时间 `arange(N')/target_fs`，起点重设0。它保持行同步，不会修复原输入已存在的不同通道时间延迟，也不接受NaN。正常finalcase400 Hz CSV主线不需要这个适配器。

## 1.12 历史 bridge 的预处理差异

此节防止把当前主线公式套到保留的历史对照。finalcase不启用bridge。它是单独协议的复现路线，而不是上述参数的别名。

### 1.12.1 历史过滤与先重采样、后分窗

直觉解释：这条旧路线先把整张纸换成稀疏点阵，再在新纸上切框；新路线则先切框再分别改点阵。框边缘附近看到的纸外内容不一样，因此结果不能视为同一处理。

`src/ppg_frailty/legacy_bridge.py:538-634`，`build_legacy_bridge_raw_windows`：

1. 557–562行整条PPG去线性趋势，再三阶0.2–8 Hz双向带通；`_filter_sos` 实现位于421–423行。
2. 563–577行 `legacy_filtered_axes` 对原ACC20 Hz、原Gyro40 Hz双向三阶滤波，但不转换SI、不扣B偏置、不估计重力。另一个分支578–596行改用已生成的canonical校准IMU，保持时间行数。
3. 599行先把拼好的八通道矩阵转float32；603–608行若目标频率不同，对整记录axis0用 `resample_poly` 的默认边缘/滤波设置，再转float32。mask若存在，用最近源位置 `round(target_index*source_fs/target_fs)` 采样（609–614行），不是对mask做低通。
4. 然后615–619行才调用 `_raw_windows_from_matrix`。不能移除这些float32舍入或改成窗口后重采样而仍称历史数值等价。

### 1.12.2 历史每窗缩放、上限与补零

直觉解释：仍是给每个小框换尺子，但旧尺子的零长保护方法不同，而且旧补零记录把空白也标成可用；这是复现旧程序的行为，不是推荐新调用照抄。

`legacy_bridge.py:426-435` 的 `_robust_scale_all_channels`：先转float32，中心中位数，`s=IQR/1.349`；s≤1e−6时用标准差；`z=(x−m)/(s+1e−6)`，非有限置0，clip±8，再float32。这与标准主线“epsilon只作判定，不加到分母；最终回退1”不同。

`_window_starts` 的438–469行按profile秒数/步长在当前采样率切窗，追加最后右对齐窗；上限用 `linspace(...).round()` 均匀选取；可用历史保留比例算ceil。`_raw_windows_from_matrix` 的491–518行先按profile逐窗缩放所有八列，或只缩放PPG两列再等待训练折六轴后变换；短记录允许时才右补零。然而517行返回的mask全部为True，包含历史填零位置；不能套用标准raw的False后缀语义。

`build_v2_window_scaled_bridge_raw_windows` 在637–654行则从canonical视图取PPG、dynamic、Gyro，仍使用上述历史窗口缩放器。不同bridge profile可能只替换其中一阶段，应以该profile解析结果为准，而不是因文件名带v2就断定走标准主线。

## 1.13 本章覆盖清单与适用范围

| 源文件 | 已解释的数值/流程模块 |
|---|---|
| `signal/preprocess.py` | 时间网格、PPG短缺口与平台、去趋势、两个注册带通方案、IMU配置分派、canonical视图构建 |
| `signal/motion_imu.py` | 稳健均值、B静态偏置、单位、双向ACC/Gyro滤波、五状态方向算法、双向0.3 Hz重力、保留重力、模长与jerk |
| `signal/imu.py` | 历史单位、六轴短缺口、单向状态滤波、四元数几何工具、无预校准方向算法、单向0.3 Hz重力、chunk连续性、比较指标 |
| `signal/views.py` | native/filter/rate/artifact各视图的用途、identity与非identity有效mask |
| `data/windows.py` | 秒到点、左/右/追加右对齐、短记录、尾补零、有效占比、数量/比例均匀上限、窗口复制 |
| `normalization.py`、`representations/raw.py` | 三种逐窗缩放、三种退化尺度、mask与float32边界 |
| `representations/imu_transform.py` | 三种六轴后变换、仅训练人拟合、有效点选择和应用 |
| `signal/resample.py`、`models/time_scale.py` | 通用DL窗口、命名fixed-kernel、独立DL视图、外部同步重采样 |
| `pipeline.py`、`data/qc.py`、`experiment.py`、`module_registry.py` | 实际入口、B选择、正式配置默认/可达性、执行次序 |
| `legacy_bridge.py` | 历史滤波/顺序/浮点精度/尺度公式/补零差异 |

本章为源码解释，不是重新跑实验的验证报告。SciPy底层滤波器设计、多相滤波系数生成和NumPy分位数等库内部实现没有在此复写；结果应以项目安装的依赖实现为准。没有通过本章新增或修改算法，也没有把函数存在误当成所有CLI/Dash组合都已开放。确定性预处理cache的键、存储内容和避免跨fold泄漏规则在cache专节说明；质量评分、denoiser和特征计算由后续章节解释。
