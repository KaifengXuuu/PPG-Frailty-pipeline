"""Chinese UI explanations for preprocessing, quality and artifact parameters.

The text follows signal/preprocess.py, signal/motion_imu.py, signal/sqi.py,
quality/window_selection.py and artifacts/*.py; it never changes configuration.
Reducer-specific names are kept separate because shared names can have different
meanings in different algorithms. Numerical vector items reuse their parent help
only when their physical interpretation is identical (for example delay taps).
"""
from __future__ import annotations

from typing import Any, Mapping


SIGNAL_HELP: dict[str, str] = {
    'signal.gap_repair.max_gap_samples':
        '允许用两端连线补齐的内部缺失段最大长度，单位为原始采样点。调大会允许修补更长空缺，但补出的曲线并非实测；调小会拒绝更多记录，首尾缺失始终不能靠外推补齐。',
    'signal.ppg_filter.low_hz':
        'PPG 带通的低频截止点，低于此处的缓慢变化会被压低。调大可去掉更多基线漂移，也可能削弱慢脉搏及其形态；调小保留更多慢变化。',
    'signal.ppg_filter.high_hz':
        'PPG 带通的高频截止点，高于此处的快速变化会被压低。调小使波形更平滑但可能损失尖峰和细节；调大保留更多细节，也保留更多高频噪声。',
    'signal.ppg_filter.order':
        'Butterworth 带通的设计阶数，决定通带与阻带交界的陡峭程度；实现采用正反向零相位滤波。调大使频率分界更陡，也增加边缘瞬态和数值敏感性；调小过渡更宽，并不改变采样率。',
    'signal.imu.gravity_method':
        '选择 IMU 重力处理路径：calibrated_roll_pitch_ekf 用同一人的 B 记录校准后融合加速度与角速度，profile_a_lowpass_0p3hz 用 B 校准后以低通估计重力，sensor_filter_only_no_gravity_removal 校准和滤波但保留重力。quaternion_error_state_ekf 与 low_pass_0p3hz 是不使用上述 B 校准的历史四元数和低通分支；切换会改变动态加速度及后续运动、特征输入，不是简单的强弱档位。',
    'signal.imu.calibration_start_s':
        '在同一 participant 的 B 静态记录中，从这个秒数开始估计传感器偏置及初始姿态。调大可跳过开始时的摆动，但缩短可用静态片段；指定的完整起止区间必须实际存在，且应保持静止。',
    'signal.imu.calibration_stop_s':
        'B 静态校准片段的结束时间，单位为秒；必须晚于开始时间。调大使用更长片段，有助于平均随机波动，但若加入身体运动会改变偏置估计，超过记录长度不会自动截短。',
    'signal.imu.sensor_lowpass_acc_hz':
        '校准式 IMU 路径中加速度传感器低通的截止频率。调小压制更多快速振动，也可能削弱真实快速动作；调大保留快速变化及更多噪声。',
    'signal.imu.sensor_lowpass_gyro_hz':
        '校准式 IMU 路径中角速度传感器低通的截止频率。调小使转动输入更平滑，但可能损失快速转动；调大让姿态计算接收更多快速变化及噪声。',
    'signal.imu.sensor_filter_order':
        '加速度与角速度传感器低通的 Butterworth 阶数，校准片段和整段处理都使用它。调大使截止附近的分界更陡，但增加边缘效应；调小滤波过渡更缓。',
    'signal.imu.gravity_lowpass_hz':
        '低通重力分支把低于此频率的慢变化当作重力方向变化，并从加速度中扣除。调大让重力估计跟随更快，也可能扣掉真实运动；调小跟随更慢，可能残留姿态变化造成的重力分量。',
    'signal.imu.gravity_filter_order':
        '低通重力估计器的滤波阶数，仅用于所选低通重力分支。调大使重力与动态信号之间的频率分界更陡，但边缘波动更敏感；调小过渡更宽，不会启用 EKF。',
    'signal.imu.gravity_mps2':
        '重力加速度的物理常数，单位为 m/s²，用于 g 单位换算、静态参考重力及相关姿态计算。调大或调小会改变物理尺度和重力扣除量，它不是降噪强度，应按数据单位及实验定义设置。',
    'signal.imu.dynamic_observation_scale':
        '五状态 EKF 在加速度模长超过重力时，按 1＋此值×超出比例放大加速度姿态观测的不确定性。调大会在这种运动下更依赖陀螺仪预测，调小更相信加速度方向；当前公式不对模长小于重力的情况对称放大。',
    'signal.imu.initial_covariance_diagonal':
        '五状态 EKF 初始不确定性的对角线，顺序为横滚角、俯仰角、X/Y/Z 角速度偏置。某项调大表示更不相信该初值，初期更容易被观测修正；调小更信任初值，影响主要在初始化阶段。',
    'signal.imu.process_covariance_diagonal_per_second':
        '五状态 EKF 每秒新增的预测不确定性，顺序为横滚角、俯仰角、X/Y/Z 角速度偏置；每个采样步会除以采样率。调大允许对应状态变化更快并通常增强观测修正，调小假定状态更稳定，但过小可能跟不上变化。',
    'signal.imu.observation_covariance_diagonal_rad2':
        '五状态 EKF 对加速度推算的横滚角、俯仰角观测设置的不确定性，单位为 rad²。调大更少相信加速度而偏重陀螺仪预测，调小跟随加速度更紧，也更容易把线性运动误当作姿态变化。',
    'artifact.motion_detector_enabled':
        '开启后使用所选已训练运动模型及其保存阈值，逐原生窗口判断高低运动并提供给后续路由。关闭会跳过此学习式检测器，不会关闭 IMU 预处理；Analyse 不会训练新的运动模型。',
    'artifact.motion_detector.evidence_path':
        '运动模型的 evidence JSON 路径，用来定位已训练权重、冻结阈值和输入定义。切换文件即切换检测器依据，需选择与目标用途及数据划分相符的现有产物；Analyse 不在当前 participant 上重新拟合。',
    'artifact.motion_detector.batch_size':
        '一次送入已训练运动检测器的窗口数量，仅控制推理批次。调大通常提高硬件利用率但占用更多内存或显存；调小更省空间，不会增加或减少模型参数。',
    'artifact.motion_detector.device':
        '运动检测器推理使用的设备，例如 cpu、cuda 或 cuda:0。切换影响运行速度和可用内存，不改变权重；不同设备后端可能产生微小浮点差异。',
    'artifact.motion_detector.reuse_scope':
        'matching_outer_fold_or_all29_final 在外层评估时使用训练人员名单匹配的 fold 检测器，最终推理使用全体模型；all29_smoke_or_final_only 仅允许全体模型用于试运行或最终推理。all29_frozen_in_sample_auxiliary 允许它作为外层评估的样本内辅助器，但不能把这种结果解释为运动检测器也严格折外。',
    'quality.mode':
        'off 跳过原信号 SQI，diagnostics_only 给出诊断但不使用 SQI 评分路由，route 则以心率与形态质量参与接收、降噪或丢弃的判断。切换不会绕过原始缺失/平线检查，也不会自动关闭运动、选窗或降噪模块；若启用降噪恢复，其恢复后质量检查仍保留。',
    'quality.flatline_duration_s':
        '判断原始 PPG 长时间完全不变的持续时间，单位为秒；SQI 也用它衡量平线段的严重程度。调小更容易把较短不变段判为问题并拒绝记录，调大更宽容，但可能放过更长的传感器停滞。',
    'quality.calibrator':
        'fixed_formula_thresholds_v1 直接使用公式归一化的组件分数，outer_train_empirical_quantiles_v1 再用外层训练集拟合的分位数范围缩放。选择经验校准时 Analyse 必须复用已保存校准产物，不会用当前 participant 重新估计范围；两种分数尺度不同。',
    'quality.calibrator_quantiles':
        '训练经验 SQI 校准器时使用的下、上分位点，保存后作为组件分数映射到 0～1 的参考范围。下界调高通常压低同一原始分数、上界调低通常抬高分数；Analyse 复用产物的已有边界，修改这两个数不会重新拟合它。',
    'quality.calibrator_quantiles.0':
        '经验 SQI 校准拟合的下分位点，决定映射到 0 的训练参考分数。调高会提高这个下界，在固定上界时通常压低中间分数；仅影响重新训练校准器，不改已选产物保存的边界。',
    'quality.calibrator_quantiles.1':
        '经验 SQI 校准拟合的上分位点，决定映射到 1 的训练参考分数。调低会降低这个上界，在固定下界时通常抬高中间分数；必须高于下分位点，Analyse 不重新拟合已选产物。',
    'quality.rate_threshold':
        '心率/峰间距质量总分 Q_rate 的通过门槛，还需同时满足原始有效覆盖率。调大使心率特征可用条件更严格，调小接收更多低分数据；不改变峰检测器本身。',
    'quality.morph_threshold':
        '波形形态质量总分 Q_morph 的通过门槛，还需同时满足原始有效覆盖率。调大更严格地限制形态特征输入，调小保留更多可能失真的波形；它与心率质量门槛分别判断。',
    'quality.minimum_coverage':
        '原始真实有效采样点所占比例的最低要求；补齐空缺不会把原始缺失点变成真实观测。调大更严格、可用记录可能减少，调小允许更多补齐数据；总分再高也不能替代这个覆盖要求。',
    'quality.cardiac_band_hz':
        'SQI 计算心搏频带能量占比时的低、高频边界，单位为 Hz。拓宽会把更多频率计入心搏能量，也可能包括噪声；收窄更专注但可能漏掉真实心率，它不改变 PPG 带通滤波。',
    'quality.cardiac_band_hz.0':
        'SQI 心搏能量频带的下界，单位为 Hz。调小纳入更慢的周期及低频干扰，调大排除它们；这只是评分频带，不是 PPG 滤波的低截止点。',
    'quality.cardiac_band_hz.1':
        'SQI 心搏能量频带的上界，单位为 Hz。调大纳入更快的周期及相应噪声，调小排除它们；需高于下界，它不重新滤波。',
    'quality.spectral_analysis_band_hz':
        'SQI 计算频谱总能量和归一化频谱熵的观察频带，单位为 Hz。拓宽会改变心搏能量占比的分母以及熵的频点范围，可能纳入更多干扰；收窄可能漏掉干扰，因此分数变化不能简单解释为信号改善。',
    'quality.spectral_analysis_band_hz.0':
        'SQI 频谱总能量和熵分析范围的低频边界。调小让更慢的成分参与分母与熵，调大排除它们；它不会修改实际滤波后的波形。',
    'quality.spectral_analysis_band_hz.1':
        'SQI 频谱总能量和熵分析范围的高频边界。调大让更多快速成分参与分母与熵，调小排除它们；应结合心搏评分频带理解，不能把分数变化等同于降噪。',
    'quality.peak_density_bpm_range':
        'SQI 认为合理的每分钟检测峰数量范围，范围内该组件给满分，范围外按偏离距离衰减。拓宽会接收更多快慢峰密度，收窄更挑剔；它不直接改变峰的位置，且下界还参与范围外惩罚的尺度。',
    'quality.peak_density_bpm_range.0':
        'SQI 满分峰密度范围的下界，单位为每分钟峰数。调高不再把更慢的密度视为范围内，调低则扩大慢端；这个下界同时参与范围外衰减尺度，整体分数不保证简单单调。',
    'quality.peak_density_bpm_range.1':
        'SQI 满分峰密度范围的上界，单位为每分钟峰数。调高放宽对快速或多检峰的容忍，调低更容易惩罚过密峰；不控制峰检测最小间隔。',
    'quality.ppi_range_s':
        'SQI 统计合理峰间距比例及寻找重复周期时使用的秒数范围。拓宽让更多间隔被视为合理，也可能纳入漏峰或多检峰；收窄更严格，但不替代峰检测器自身的间隔规则。',
    'quality.ppi_range_s.0':
        'SQI 合理峰间距和自相关搜索延迟的下界，单位为秒。调小允许更短间隔即更快节律，调大排除它们；这不是修改峰检测器，而是改变后续质量观察范围。',
    'quality.ppi_range_s.1':
        'SQI 合理峰间距和自相关搜索延迟的上界，单位为秒。调大允许更长间隔即更慢节律，也可能接受漏峰间隔；调小排除它们。',
    'quality.ppi_stability_min_intervals':
        '至少有多少个有效峰间距才计算间距变异程度。调大要求更充分的观测，短片段更可能没有该质量组件；调小让短片段也得到数值，但估计更不稳定。',
    'quality.welch_max_nperseg':
        'Welch 频谱分析每段最多使用的采样点数，实际不超过输入长度。调大提升频率分辨率但可平均的片段通常减少，调小频率分辨更粗、平均次数更多；不改变采样率。',
    'quality.template_min_peaks':
        '开始计算脉搏模板相似性前，整段至少需要的检测峰数量。调大更容易因峰不足而不给出该组件，调小可覆盖更短片段；实际可用完整脉搏数仍由另一门槛控制。',
    'quality.template_min_beats':
        '计算模板相似性所需的完整有效脉搏片段最少数量，边缘不完整片段不会计入。调大要求更多重复波形支持，调小允许少量脉搏就评分，但不能保证代表整段。',
    'quality.template_resample_points':
        '比较每个脉搏与中位模板前，将每段脉搏插值到的统一点数。调大比较网格更细并增加计算，调小更粗；这是对齐网格，不增加真实测量信息或改变原始采样率。',
    'quality.component_normalization.template_half_width_s':
        '模板比较时，以每个峰为中心向前、向后各截取的秒数。调大包含更多脉搏形状，也可能跨入相邻心搏或丢失边缘片段；调小只比较峰附近局部形状。',
    'quality.component_normalization.cardiac_concentration_reference':
        '把心搏频带能量占比映射成质量分数时所除的参考值，结果限制在 0～1。固定信号时调大通常降低该组件分数、要求更集中的心搏能量；调小更容易达到满分。',
    'quality.component_normalization.autocorrelation_reference':
        '把信号重复周期的自相关强度映射成质量分数时所除的参考值。固定信号时调大更严格、组件分数降低，调小更容易达到满分；不改变自相关搜索间隔。',
    'quality.component_normalization.ppi_cv_scale':
        '峰间距变异惩罚的宽容尺度，组件按 exp(−间距变异系数/此值) 评分。调大对节律不均更宽容，调小惩罚更重；它不平滑或修正峰间距。',
    'quality.component_normalization.motion_rms_scale':
        '运动幅度惩罚的宽容尺度，组件按 exp(−动态加速度均方根/此值) 评分。调大降低运动造成的扣分，调小扣分更强；不改变 IMU 波形或运动检测器权重。',
    'quality.component_normalization.nonflat_std_threshold':
        '判断 PPG 是否具有非零变化的标准差门槛，超过它该组件才得分。调大更容易将微弱波动视为近乎平线，调小更宽容；这与连续完全相同采样点的平线时长检查不同。',
    'quality.component_normalization.clipping_fraction_reference':
        '原始 PPG 最小值/最大值重复占比达到此值时，削顶启发式组件降到零。调大更容忍极值重复，调小更敏感；它只是极值占比线索，并不能证明传感器真的触及 ADC 电压边界。',
    'quality.component_normalization.saturation_fraction_reference':
        '有明确饱和证据时，用于将饱和采样比例转成扣分的参考比例。调大更宽容、调小扣分更快；没有提供真实饱和比例时该组件不可用，不会凭空推断饱和。',
    'quality.component_normalization.morph_skewness_scale':
        '对波形分布偏斜程度的宽容尺度，组件按 exp(−绝对偏度/此值) 评分。调大容忍更不对称的分布，调小更偏好接近对称；不直接改变波形。',
    'quality.component_normalization.morph_kurtosis_center':
        '波形分布峰度的参考中心，偏离该中心会降低形态组件分数。调大把偏好移向更尖重尾的分布，调小移向更平缓的分布；它不是越大越严格或越好的单向旋钮。',
    'quality.component_normalization.morph_kurtosis_scale':
        '峰度偏离参考中心时的宽容尺度，组件按 exp(−偏离量/此值) 评分。调大减轻偏离惩罚，调小加重；参考中心由另一个参数决定。',
    'quality.component_normalization.component_pass_threshold':
        '仅用于为单个 SQI 组件的展示结果标记 PASS 或 FAIL。调大会让更多组件显示未通过，调小则相反；总分仍对连续组件分数加权，因此它不直接改变 Q_rate/Q_morph 总分及其通过门槛。',
    'quality.window_selection.policy':
        'none 保留所有候选窗口，legacy_per_file_top_fraction 在每个 recording 内按心搏/运动质量分数选择排名靠前的比例。切换不会跨 participant 排名，也不使用类别标签；被选择的数量和应用分区由后续参数决定。',
    'quality.window_selection.keep_fraction':
        '在每个 recording 内保留质量排名靠前的窗口比例，数量向上取整且至少留一个。调大保留更多数据及多样性，也可能纳入较差窗口；调小更挑剔但减少样本，仅在启用按比例选窗时生效。',
    'quality.window_selection.application_scope':
        'outer_train_only 只筛训练窗口，all_partitions 同时筛训练、测试和推理窗口。legacy_train_and_aggregation 筛训练窗口，测试/推理仍保存所有窗口预测、仅在聚合时使用所选窗口；选择决定比较的是哪些预测单位。',
    'artifact.denoiser_enabled':
        '开启后允许路由使用所选非 identity 降噪器的输出恢复需要处理的窗口，是否接收仍由原有质量路由决定。算法先处理完整记录再取对应窗口；关闭会跳过降噪恢复而不改变滤波器，选择 identity 也不会进行实际伪影消除。',
    'artifact.reducer':
        '选择降噪算法：identity 原样传递，NLMS 用 IMU 参考抵消，spectral_mask 用运动频谱压制，PCA/ICA/NMF 分离成分，SSA/EMD/CEEMD 分解重构，DWT 保留固定小波近似。不同方法并非同一算法的强弱档；非 identity 输出沿既有规则只用于允许的心率分支，不声称保持脉搏形态。',
}


# The five-state order is fixed by calibrated_roll_pitch_ekf, not by UI labels.
for _index, _state in enumerate(('横滚角', '俯仰角', 'X 轴角速度偏置', 'Y 轴角速度偏置', 'Z 轴角速度偏置')):
    SIGNAL_HELP[f'signal.imu.initial_covariance_diagonal.{_index}'] = (
        f'五状态 EKF 对初始{_state}的不确定性。调大更不相信这个初值、允许更强的初期修正；'
        '调小更信任初值，不是直接对信号增加噪声。')
    SIGNAL_HELP[f'signal.imu.process_covariance_diagonal_per_second.{_index}'] = (
        f'五状态 EKF 对{_state}每秒新增的预测不确定性，计算时除以采样率。'
        '调大允许该状态更快变化并通常更受观测修正，调小假定它更稳定，但可能跟不上真实变化。')
for _index, _angle in enumerate(('横滚角', '俯仰角')):
    SIGNAL_HELP[f'signal.imu.observation_covariance_diagonal_rad2.{_index}'] = (
        f'加速度推算的{_angle}观测不确定性，单位为 rad²。调大更少相信这项加速度观测，'
        '调小跟随它更紧，但线性运动时也更容易受干扰。')


_COMPONENT_MEANINGS = {
    'cardiac_concentration': '能量集中在心搏频带内的程度',
    'autocorrelation_periodicity': '波形在合理心搏间隔上重复的程度',
    'normalized_spectral_entropy': '频谱是否集中而非杂乱分散，熵越低该组件越高',
    'peak_density_bpm': '每分钟检测峰数是否处于设定范围',
    'ppi_physiological_fraction': '落在合理秒数范围内的峰间距比例',
    'ppi_stability': '峰间距相对波动的稳定性',
    'red_ir_agreement': 'RED 与 IR 波形的绝对相关程度',
    'motion_energy_rms': '动态加速度幅度的惩罚，运动越小该组件越高',
    'nonflat_scale': '波形标准差是否超过近乎平线门槛',
    'source_coverage': '原始真实有效采样点覆盖率',
    'flatline': '连续完全不变片段的时长惩罚',
    'clipping': '最小或最大数值反复出现的削顶启发式惩罚',
    'long_gap': '原始最长缺失段是否超过允许长度',
    'saturation': '有明确证据时的采样饱和比例惩罚',
    'template_correlation': '完整脉搏与中位脉搏模板的形状相似度',
    'skewness': '波形分布偏斜程度的惩罚',
    'pearson_kurtosis': '波形分布峰度偏离参考中心的惩罚',
}
_WEIGHT_COMPONENTS = {
    'rate': ('cardiac_concentration', 'autocorrelation_periodicity', 'normalized_spectral_entropy',
             'peak_density_bpm', 'ppi_physiological_fraction', 'ppi_stability', 'red_ir_agreement',
             'motion_energy_rms', 'nonflat_scale', 'source_coverage', 'flatline', 'clipping', 'long_gap', 'saturation'),
    'morph': ('cardiac_concentration', 'red_ir_agreement', 'nonflat_scale', 'source_coverage',
              'flatline', 'clipping', 'long_gap', 'saturation', 'template_correlation', 'skewness', 'pearson_kurtosis'),
}
for _endpoint, _components in _WEIGHT_COMPONENTS.items():
    for _component in _components:
        SIGNAL_HELP[f'quality.{_endpoint}_component_weights.{_component}'] = (
            f'在 Q_{_endpoint} 总分中，此权重对应“{_COMPONENT_MEANINGS[_component]}”。'
            '调大增加相对影响、调小减弱，设为 0 不参与；总分只对可用且正权重的组件重新归一化，'
            '因此调大不保证总分升高，缺失组件也不会因权重大而变得可用。')


_REFERENCE_HELP = (
    'imu_axes6_reference_v2 使用动态加速度及角速度的六个轴，augmentation 选项额外加入加速度模长、角速度模长和加加速度模长。'
    '参考列会标准化并移除常数列；增加派生量提供不同运动线索，也增加冗余，效果并非必然提高。')
_EMD_HELP = {
    'max_imfs': '最多提取多少个由快到慢的振荡分量，实际还受剩余极值数量限制。调大允许更充分地分解慢成分并增加计算，调小更早留下残余；EMD 重构会丢弃残余，因此结果也会变化。',
    'max_sift': '提取每个振荡分量时，最多重复多少次上下包络均值扣除。调大给分解更多迭代机会但增加计算，调小更快、分量可能尚未稳定；仍可能先被收敛阈值提前停止。',
    'sd_threshold': '两次筛分结果的相对平方变化小于此值时，停止该分量的迭代。调大更容易提前停止、筛分较少；调小要求更稳定并可能增加计算，不代表波形一定更真实。',
}
ARTIFACT_HELP: dict[str, dict[str, str]] = {
    'identity': {},
    'dwt_a2_legacy': {},  # The historical db4/level-2 reconstruction has no tunable controls.
    'emd_sifting_rate_only': dict(_EMD_HELP),
    'ceemd_lite_nlms_legacy': {
        **_EMD_HELP,
        'max_imfs': '每次加噪分解最多提取的振荡分量数，实际可能因极值不足提前停止。调大让更多慢成分进入分解和后续运动/心搏判定，也增加计算；调小使更多内容留在残余，CEEMD 对残余另有历史处理规则。',
        'pairs': '给信号加入正、负成对噪声并分解的组数，最后平均对应分量。调大增加抵消随机扰动的机会，但每组需要两次 EMD、计算量随之增长；调小更快但平均更少。',
        'noise_ratio': '加入噪声的标准差相对于原信号标准差的比例。调大更强地扰动分解、可能分开混合振荡，也可能留下更多随机影响；调小更接近普通 EMD。',
        'random_seed': 'CEEMD 成对加噪所用随机数种子，决定具体噪声序列。保持相同种子有助于同环境重现，改成更大或更小的数字没有质量高低含义。',
        'protect_bandwidth_hz': '围绕估计心率及其指定谐波保留的频率容差，单位为 Hz；落入此范围的分量先被保护。调大保护更多分量但也可能留下噪声，调小保护更窄、误删心搏的风险增加。',
        'protect_harmonics': '除基频外按整数倍心率逐项检查保护，计数 1 表示只保护基频。调大覆盖更多高次心搏谐波，也可能保护恰好落在附近的运动；调小保护范围更有限。',
        'low_motion_hz': '在未被心搏保护的分量中，主频不高于此值时归入运动参考。调大把更多慢成分归为运动，调小则更少；这是分量分类边界，不是预处理带通截止点。',
        'high_motion_hz': '在未被心搏保护的分量中，主频不低于此值时归入运动参考。调小把更多快速成分归为运动，调大则更少；中间频段还会按与心搏带波形的相关程度分类。',
        'nlms_length': '以 PPG 分解得到的运动参考进行 NLMS 抵消时，保留多少个历史参考采样点。调大可拟合更长的干扰滞后，但增加系数和计算；调小反应更局部，这个参考不是外部 IMU。',
        'nlms_mu': '历史 NLMS 每次根据当前残差修正抵消系数的步长。调大适应更快，也可能震荡或过度扣除；调小更新更慢，该参数不是直接按比例缩放输出。',
        'nlms_leak': '历史 NLMS 每个采样步把旧系数乘以 1−此值，再加新修正。调大遗忘更快、限制系数长期积累，调小记忆更长；过强遗忘也会削弱持续干扰的抵消。',
    },
    'nlms_imu_anc': {
        'imu_reference_profile': _REFERENCE_HELP,
        'taps_per_delay': '每个起始延迟后连续取多少个参考采样点，所有延迟位置会合并去重后构造 NLMS 输入。调大可覆盖更长局部历史，也增加系数、计算及起始无效长度；调小模型更短，可能无法描述较长滞后。',
        'delay_taps': '参考信号的起始延迟列表，单位为采样点，各项再扩展 taps_per_delay 个连续位置。增大某项会观察更早的 IMU，新增项可覆盖不同滞后，但重复位置被合并，最大延迟还会增加开头无历史可用的长度。',
        'step_size': 'NLMS 按残差及参考能量归一化后更新系数的步长，需大于 0 且小于 2。调大追踪干扰更快，也更可能震荡或误删心搏；调小稳定但跟随更慢。',
        'epsilon': '加在参考能量分母上的正数，避免能量很小时更新过大。调大抑制弱参考条件下的系数更新，调小允许更大的更新但更敏感；不直接加到输出信号。',
        'leakage': '允许更新时将已有 NLMS 系数乘以 1−此值，控制对过去抵消关系的遗忘。调大遗忘更快、调小保留更久；未达到参考幅度更新门槛时不施加这次遗忘。',
        'update_gate_reference_rms': '标准化 IMU 参考的均方根至少达到此值才更新 NLMS 系数。调大减少学习干扰关系的时刻，调小更频繁更新；门槛以下仍用已有系数预测并扣除干扰，并非关闭输出抵消。',
    },
    'pca_bss': {
        'imu_reference_profile': _REFERENCE_HELP + '在 PCA 中它仅影响候选 PPG 分量的运动相关惩罚，不是增加 PCA 的输入通道。',
    },
    'fastica_bss': {
        'imu_reference_profile': _REFERENCE_HELP + '在 ICA 中它仅影响候选 PPG 分量的运动相关惩罚，不是增加 ICA 的输入通道。',
        'max_iter': 'FastICA 求解独立分量的最大迭代次数，达到收敛条件可提前停止。调大允许更充分求解但可能耗时更久，调小可能未收敛；当前实现不会把未收敛结果当作成功输出。',
        'tolerance': 'FastICA 判定相邻迭代是否足够稳定的容差。调大通常更早停止但解可能更粗，调小要求更严格并可能达到迭代上限；它不表示允许的 PPG 噪声幅度。',
        'random_state': 'FastICA 初始随机状态的种子，影响分离过程的起点。相同输入与环境下固定种子有助于重现，改大或改小没有更强或更弱的含义。',
    },
    'nmf_bss': {
        'nmf_rank': '用多少个非负基底分解 RED/IR 的时频幅度图，实际数量还受矩阵尺寸限制。调大表达更灵活但增加计算、可能拆散同一心搏来源；最终仍按心搏频带占比选择一个基底重构，而不是全部保留。',
        'nperseg': 'NMF 输入的短时傅里叶变换窗口点数，实际不超过记录长度。调大频率分辨更细、时间变化定位更粗，调小相反；它与原始采样率共同决定窗口秒数。',
        'overlap_fraction': 'NMF 时频分析相邻窗口重叠的比例。调大产生更密集的时间帧并增加计算，调小帧更稀疏；不增加真实采样点，重叠点数按窗口长度换算。',
        'max_iter': '非负矩阵分解优化的最多迭代次数，达到停止条件时可提前结束。调大给目标函数更多下降机会但可能更慢，调小更快但可能停在较粗分解；不增加基底数量。',
        'tolerance': 'NMF 优化器停止迭代的容差，衡量是否还需继续改进分解。调大通常更早停止、精度较松，调小求解更严格但更耗时；不改变心搏频带选择规则。',
        'random_state': '传给 NMF 分解器的随机状态，用于其涉及随机性的步骤；当前采用 NNDSVDA 初始化。保持种子便于重现，增大数值不代表增加随机噪声强度，具体结果是否改变取决于实际求解路径。',
    },
    'spectral_mask': {
        'imu_reference_profile': _REFERENCE_HELP,
        'stft_window_s': '每个短时频谱窗口覆盖的秒数。调大让频率分辨更细但跟踪瞬时变化更慢，并扩大缺失 IMU 周围不能使用的范围；调小更局部但频率分辨更粗。',
        'stft_hop_s': '相邻短时频谱窗口的起点间隔，单位为秒，不能大于窗口时长。调小增加重叠和计算、使时间网格更密，调大更稀疏；它不是对原信号重新采样。',
        'imu_mask_quantile': '每帧 IMU 频谱按这个频率维分位数归一化，作为与 PPG 比较的运动尺度。其他量固定时调大使分母不减、归一化运动减弱，因此抑制通常更轻；调小通常更强，而不是“更高分位数更严格”。',
        'mask_strength': '心搏保留频带内按运动污染比例压低频谱的最大强度，增益下限为 1−此值。调大抑制更强，也可能损伤重叠的心搏；设为 0 仅关闭带内软抑制，保留频带之外仍会置零。',
        'preserve_band_hz': '最终重构保留的频率上下界，单位为 Hz，范围之外的频点直接置零。拓宽保留更多信号也保留更多干扰，收窄可能丢失真实心搏成分；范围内仍可能被运动掩膜压低。',
        'preserve_band_hz.0': '谱掩膜输出保留频带的下界，低于它的频点直接置零。调大删除更多缓慢变化，调小保留更慢节律及低频干扰；它不同于前面的 PPG 预处理低截止点。',
        'preserve_band_hz.1': '谱掩膜输出保留频带的上界，高于它的频点直接置零。调小删除更多快速成分和细节，调大保留更多谐波及噪声；频带内仍受运动掩膜影响。',
    },
    'ssa_decomposition': {
        'embedding_samples': '把信号排成重叠延迟矩阵时每列覆盖的采样点数，实际最多取记录长度的三分之一。调大可描述更长周期，但奇异值分解的内存和计算明显增加；调小更局部，可能分不开慢结构。',
        'max_components': 'SSA 最多检查奇异值从大到小排列的前多少个分量，再按心搏能量占比选择。调大让更多细节及噪声有机会进入候选，调小只看较强分量；不是保证最终保留这么多个。',
        'cardiac_low_hz': '判断 SSA 分量心搏能量占比时的频带下界，单位为 Hz。调小可接受更慢的振荡也可能纳入基线变化，调大排除它们；只影响分量选择，不重新设定 PPG 带通。',
        'cardiac_high_hz': '判断 SSA 分量心搏能量占比时的频带上界，单位为 Hz。调大可接受更快的振荡及相应噪声，调小排除它们；最终只重构达标分量。',
        'minimum_cardiac_concentration': 'SSA 分量在指定心搏频带内的能量占固定分析频带能量的最低比例。调大更挑剔、保留更少分量，调小更宽容；若没有分量达标就没有降噪结果，不会自动改选最高分量。',
    },
}


_SHARED_ARTIFACT_HELP = {
    'imu_reference_profile': _REFERENCE_HELP,
    'max_imfs': 'EMD 或 CEEMD 每次分解最多提取的振荡分量数，实际可能因极值不足提前停止。调大允许更多慢成分进入分解并增加计算，调小让更多内容留在残余；残余如何处理取决于所选降噪算法。',
    'max_iter': 'ICA 或 NMF 优化器最多执行的迭代次数，满足所选算法的停止条件时可提前结束。调大允许更充分求解但可能更慢，调小可能尚未收敛；它不增加信号长度或模型成分数量。',
    'tolerance': 'ICA 或 NMF 优化器的收敛容差，具体停止量由所选分解器定义。调大通常更容易提前停止，调小求解更严格但更耗时；这不是允许的信号噪声幅度。',
    'random_state': '传给所选 ICA 或 NMF 分解器的随机状态，控制算法涉及随机性的初始化或求解步骤。固定种子便于同环境重现，数值大小不表示随机噪声强弱；具体影响取决于分解器。',
}


def signal_help(path: str, config_context: Mapping[str, Any] | None = None) -> str | None:
    """Return help without reading files, resolving config or mutating UI state."""
    if path.startswith('artifact.parameters.'):
        key = path.removeprefix('artifact.parameters.')
        parent, _, index = key.rpartition('.')
        lookup = parent if index.isdigit() else key
        reducer = (config_context or {}).get('artifact', {}).get('reducer')
        if reducer is not None:
            help_by_parameter = ARTIFACT_HELP.get(str(reducer), {})
            return help_by_parameter.get(key) or help_by_parameter.get(lookup)
        # A context-free caller can still explain shared or unique leaves. UI
        # callers supply context so ICA/NMF and IMU-use differences stay exact.
        if lookup in _SHARED_ARTIFACT_HELP:
            return _SHARED_ARTIFACT_HELP[lookup]
        for help_by_parameter in ARTIFACT_HELP.values():
            if key in help_by_parameter:
                return help_by_parameter[key]
            if lookup in help_by_parameter:
                return help_by_parameter[lookup]
        return None
    return SIGNAL_HELP.get(path)
