"""Chinese UI explanations; these descriptions never resolve or change values.

Scientific meanings follow peaks/resolver.py, signal/prv.py,
representations/raw.py, signal/resample.py and data/windows.py. Signal and
model descriptions live beside this file so that each algorithm has one source
of UI documentation. Unknown user-plan metadata is not given invented effects.
"""
from __future__ import annotations

import re
from typing import Any, Mapping


PARAMETER_HELP = {
    'config_id': '当前参数组合的标识，用于区分配置和记录来源；改名不会改变数学计算，也不会自动更改输出目录。',
    'manifest.path': '记录清单 CSV 的路径，决定可选 recording、participant、角色和标签来源；换清单会改变输入数据，不是滤波参数。',
    'splits.path': '已生成的数据划分表，规定每个 repeat/fold 的训练与测试 participant；换表会改变评估划分，Analyse 不用它重新划分单条记录。',
    'roles': 'Dash 按 B/R/S/W 选择记录类别；新勾选类别展开为其全部记录编号，已加载 YAML 的部分编号保持不变，取消后重选才展开。下方显示实际编号，导出 YAML 仍保存它们；最终分类还受 classifier_role_families 独立限制，不会生成新记录。',
    'features.enabled_groups': '选择脉搏间隔、时域/频域/非线性变化、波形形状、双波长关系和工程统计特征。增加组会增加相应特征；raw 模型仍吃信号，特征区用于预览，feature-vector/fusion 等路线才消费对应特征。',
    'features.rate_prv_min_duration_s': '允许报告基本脉率/PPI 的最短观测时间（秒）；增大时短记录更容易缺失这些结果，减小会接受更短但证据更少的片段。',
    'features.rate_prv_min_peaks': '基本脉率/PPI 至少需要的脉搏峰数；增大要求更多完整脉搏，减小可覆盖短片段，但并不提高峰检测准确度。',
    'features.time_prv_min_duration_s': '计算时域脉搏间隔变化的最低观测时长（秒）；增大使时域特征的输出条件更严格，减小接受更短记录，不改变特征公式。',
    'features.time_prv_min_coverage': '时域 PRV 所需有效间隔覆盖比例；调高会拒绝更多缺失或断续片段，调低提高可用率，但剩余片段可能更不代表整段记录。',
    'features.time_prv_min_intervals': '时域 PRV 至少需要的有效间隔数量；调高减少小样本估计，调低允许更少间隔，不会补造缺失脉搏。',
    'features.spectral_prv_min_duration_s': '计算间隔频谱的最低观测时长（秒）；长记录更有机会观察慢变化。调高会使更多短记录无频域结果，调低不等于获得可靠的低频信息。',
    'features.spectral_prv_min_coverage': '频域 PRV 所需有效间隔覆盖比例；调高筛选更严格，调低接受更多断续信号，可能增加插值对频谱的影响。',
    'features.spectral_prv_min_intervals': '频域 PRV 最低有效间隔数；调高需要更多脉搏才输出频谱，调低放宽资格，但不能替代足够的观测时长。',
    'features.tachogram_fs_hz': '将不等间距的脉搏间隔插值到每秒多少点，再估计频谱；调高加密插值网格并增加计算量，不会创造额外心搏信息，也不改变原始 PPG 采样率。',
    'features.sample_entropy.m': '比较间隔序列时，一次匹配连续多少个间隔，再检查多一个间隔是否仍相似；调大比较更长模式、匹配通常更少，短记录更容易得不到稳定结果。',
    'features.sample_entropy.r_sd_fraction': '相似判定容差占间隔标准差的比例；调大允许差别更大的模式算作相似，匹配更多，熵值可能下降，但不保证严格单调。',
    'features.sample_entropy.min_intervals': '计算样本熵所需的最少有效间隔数；调高会减少短样本熵结果，调低提高覆盖率，但稀少匹配可能使结果不稳定或缺失。',
    'signal.peak_detector.detector_id': '选择脉搏峰定位方法：MSPTDfast 用多时间尺度找峰；Aboy v1/v2 使用各自的自适应检测实现；双极性 prominence 是独立消融路线。切换会改变峰、PPI 和后续特征，不会只更换显示样式。',
    'signal.peak_detector.min_observation_sec': '峰检测结果被接受所需的最短有效观测时间（秒）；调高会排除更多短片段，调低放宽资格，不会增加信号长度。',
    'signal.peak_detector.min_peaks': '峰检测结果至少需要的有效峰数；调高更严格，调低更容易接受短片段，但不等于在信号中强行多找峰。',
    'signal.peak_detector.parameters.minimum_heart_rate_bpm': 'MSPTDfast 用于限制多尺度搜索的最低心率（次/分）；调低保留更长时间尺度以容纳慢脉搏，调高减少这些尺度及计算量，可能漏掉慢节律。',
    'signal.peak_detector.parameters.target_downsample_hz': 'MSPTDfast 内部粗检测的目标采样率（Hz），实际网格由整数降采样倍数决定；调高保留更多时间点并增加计算量，不改变生产信号的 400 Hz 时间基准。',
    'signal.peak_detector.parameters.window_s': 'MSPTDfast 每次检测的片段长度（秒）；调大可利用更长上下文但增加局部计算，调小更局部、分段边界更多，可能改变合并后的峰。',
    'signal.peak_detector.parameters.overlap_fraction': 'MSPTDfast 相邻检测片段的重叠比例；调高会减小步长，让边缘位置被重复观察并增加计算；调低减少重复检测，最终仍按原规则合并峰。',
    'representation_mode': '选择模型输入形式：raw 是信号窗口，feature_vector 是记录特征向量，feature_matrix 是有时间顺序的窗口特征，fusion 结合信号袋与记录特征。切换会改变输入结构，需要匹配的模型及已拟合变换。',
    'signal.dl_resampling.enabled': '是否仅对深度学习输入进行抗混叠重采样；关闭时输入保持 400 Hz，打开后使用 target_fs_hz。特征与峰定位的原时间网格不因此改变。',
    'signal.dl_resampling.target_fs_hz': '深度学习输入每秒保留的采样点数；调低可减少序列长度与计算，但去掉更多高频信息。同一卷积核的物理时间跨度也会变化；不能据此保证分类表现提高。',
    'signal.normalization.raw_ppg': '选择窗口内不缩放、均值/标准差缩放或中位数/四分位距缩放。虽然名为 raw_ppg，当前实现对原始窗口中的所有通道应用该策略；切换改变模型输入尺度，不改原始信号或形态特征源。',
    'signal.normalization.raw_imu': '选择额外的训练集拟合 IMU 缩放策略，none 表示不做该层变换；其他策略复用训练得到的中心与尺度。Analyse 需要权重包中已有对应变换，不会对新 participant 重新拟合。',
    'signal.normalization.clip_after_scale': '归一化后的数值裁剪范围 [下界, 上界]，null 表示不裁剪；收窄会压住更多极端值，也丢掉更多幅度差异。raw_ppg=none 时当前窗口路径不执行此裁剪。',
    'signal.normalization.iqr_fallback': '四分位距太小或无效时的备用尺度：标准差、MAD 或直接用 1；切换决定平坦/异常窗口怎样缩放，不会修改原数据或保证异常信号可用。',
    'signal.normalization.robust_iqr_divisor': '稳健尺度按 IQR/此值计算，再用它除去中心后的信号；在未触发备用尺度和裁剪时，调大此值会放大归一化振幅，调小会缩小。',
    'signal.normalization.mad_consistency_divisor': '选择 MAD 备用尺度时，用绝对偏差中位数除以此值；调大使该尺度更小、归一化振幅更大。未使用 MAD 备用路径时不生效。',
    'signal.normalization.scale_epsilon': '判断尺度是否过小的阈值；调高会让更多近乎平坦的通道转入备用尺度，调低更容易保留极小尺度，从而可能放大微小噪声。它不是加在所有分母上的常数。',
    'signal.normalization.standard_ddof': '标准差分母中的自由度修正：0 使用 n，1 使用 n−1；选择 1 时同一窗口的尺度略大、标准化振幅略小，短窗口差别更明显。',
}

WINDOW_HELP = {
    'length_s': '一个窗口覆盖的时间（秒）；调大让每个输入看到更长上下文，通常完整窗口更少、单窗口更长；调小更局部。还需满足所选特征/模型的输入条件。',
    'hop_s': '相邻窗口起点的间隔（秒）；调小增加重叠窗口、计算量和相邻样本相似性，调大减少窗口数量，不改变原信号采样率。',
    'end_alignment': '只按固定步长从起点切窗，或额外补一个与记录末端对齐的完整窗口；后者改善尾部覆盖，可能增加一个窗口，但不是补零。',
    'padding': '是否为短记录或尾段补零；开启能生成不足整窗的候选输入，但仍受 min_valid_fraction 约束。默认有效比例为 1 时，补零窗口不会仅因开启 padding 就被保留。',
    'min_valid_fraction': '窗口内真实有效样本应占的最低比例；调高会排除更多补零/不完整窗口，调低允许更多这类输入，不会把补零当成真实测量。',
    'cap_per_file': '每条 recording 最多保留多少窗口，null 表示不设此数量上限；超限时沿记录进度均匀选取。调低省计算但减少覆盖，与比例上限不能同时设置。',
    'cap_fraction_per_file': '每条 recording 候选窗口保留的比例，null 表示不设比例上限；调低均匀抽取更少窗口，调高保留更多。与数量上限不能同时设置。',
}

FIELD_HELP = {
    'training-yaml': '可选初始配置；不选时用函数默认值。选择 YAML 会填入控件，之后 Analyse 使用当前控件，不会偷偷还原 YAML。',
    'yaml-case': '选择 study 中哪一个 case 的配置填入面板；不会立即训练，也不等于执行整个 comparison。',
    'record-ids': '选择要分析的清单记录；增加记录会增加处理量和模型聚合输入，自定义 CSV 表有有效路径时替代此选择。',
    'preview-record': '决定各阶段画哪条 recording；模型 Analyse 仍可对已选的多条记录推理，换图不会修改源文件。',
    'participant-id': '自定义 CSV 归属的 participant 标识；同一个人的静态 B 校准及多条动态记录须使用同一身份，不是分类标签。',
    'preview-start': '图从记录的第几秒开始显示；增大只向后移动预览，不裁剪实际送入滤波和模型的完整 recording。',
    'preview-duration': '图显示的时长（秒）；增大展示更长区间，长区间为绘图抽点，不改变实际算法输入或采样率。',
    'calibration-path': '可显式选择同 participant 的静态 B CSV，供 IMU 估计偏置和校准基准；换文件会改变校准，不是加载分类权重。',
    'motion-bundle': '选择已有 motion evidence JSON 及其关联权重/阈值；Analyse 只复用该产物，不在当前输入上训练检测器。手动路径优先于下拉选择。',
    'sqi-artifact': '选择已在训练数据上拟合的 SQI bounds/participant IDs JSON；仅需要拟合校准的 SQI 路线使用。换产物会改变分数标定，不会重新拟合。',
    'model-mode': 'Analyse 复用已有权重做分类；Train 显示训练设置与 Run。切换模式本身不启动任务，Stop 始终保留。',
    'model-export': '选择 model_config 导出目录，供下一项选择 case 和权重；此选择不覆盖当前控件，要恢复模型原参数请另选它的 resolved YAML。',
    'model-case': '选择导出目录中哪个 case 的已训练模型；不同 case 可能有不同表征/架构，必须与当前输入结构匹配。',
    'model-bundle': '直接选择已有 learned bundle，包含权重及适用的训练期变换；手动路径优先，Analyse 不对新输入重新训练。',
    'train-target': '选择运行当前控件、comparison 队列、完整 study YAML 或专项工具请求；只有 Current controls 用当前单组参数覆盖训练配置。',
    'run-name': '输出 run 的目录名；留空按原命名规则生成。只影响输出定位，不改变算法或数据划分。',
    'repeat-indices': '要执行的 repeat 编号（如 all 或 0,1）；增加编号会运行更多既定重复实验，不是增加单个模型 epoch。',
    'fold-indices': '要执行的 fold 编号（如 all 或 0,1）；选择子集可节省计算，但不是完整交叉验证，也不重新生成划分。',
    'job-count': '并行实验进程数；调高可能更快，但占用更多显存/内存，过高可能资源不足。不会增加每个模型的训练轮数。',
    'cache-mode': 'off 不读写磁盘预处理缓存；read_only 只读已有；read_write 可读写。切换影响耗时与磁盘占用，不替代训练期拟合，也不会删除已有缓存。',
    'cache-root': '可复用预处理 cache 的目录；换目录会改变缓存命中位置，不改变数据源。已有缓存不会自动迁移或删除。',
    'resume-path': '已有 pipeline output 的路径，用于按原机制继续未完成任务；留空创建新 run，不应指向无关实验。',
    'train-flags': 'Refit 开启时另做全 cohort 拟合并输出权重；关闭仍可导出 fold 权重。Dry run 只检查/规划，不启动正式模型训练。',
    'comparison-name': '当前参数组合加入临时 comparison 队列时的单位名；Add 保存当时的控件快照，之后改控件不会改变已加入的单位。',
    'analysis-runs': '报告读取的已完成或可分析的 pipeline 输出；多选用于对照，不会重新训练模型。',
    'analysis-mode': 'single、comparison、ablation、test 对应不同分析组织方式；只改变已有结果的分析任务，不改训练权重。',
    'analysis-preset': '预先组合好的报告模块集合；改预设会改变生成哪些图表，明确选择 figures/tables 可覆盖相应默认集合。',
    'analysis-modules': '在报告中加入相应分析模块；更多模块可能增加统计计算与图表，不改变已有预测值。',
    'analysis-figures': '明确选择要生成的图；留空使用预设集合。图所需的预测、标签或训练轨迹缺失时，按报告原规则处理。',
    'analysis-tables': '明确选择派生统计表；留空使用预设集合。它们与 pipeline 的原始预测表不同，不会覆盖原始预测。',
    'reference-case': '对照统计使用的参考 case；切换会改变比较方向和参考对象，不改变任一 case 的预测。',
    'factor-paths': '用逗号分隔配置字段路径，以指定消融/对照的因素；例如 model.model_id。改变分组依据，不改变已运行算法。',
    'include-cases': '只纳入这些 case（逗号分隔）；留空不按此白名单筛选。减少对象会改变报告样本集合。',
    'exclude-cases': '从报告中排除这些 case（逗号分隔）；只筛报告输入，不删除对应 pipeline 输出。',
    'report-name': '报告子目录名；留空按原规则使用输入 run 名。只影响输出路径，不改变统计公式。',
    'statistics-alpha': '统计检验的显著性水平；调小意味着判显著更严格，不会令 p 值自动变小。报告中各统计图按自身已实现规则使用该设置。',
    'calibration-bins': '概率校准图分箱数量；调大分辨率更细，但每箱样本更少、波动可能更大。这里只评估概率，不重新校准模型。',
    'report-output': '选择已有报告目录进行预览；切换只读不同产物，不重新生成报告。',
    'report-figure': '选择 PNG/SVG 或 HTML 图形产物；切换只改变显示，不改变数据或模型。',
    'report-table': '选择报告中的数据表；只读预览，可在表内排序筛选，不改变文件内容。',
    'tool-operation': '选择原 CLI 提供的导出、验证、审计或专项研究操作；下方控件直接对应其参数，选择本身不执行任务。',
    'tool-plan-yaml': '加载专项计划并展开其中参数；之后使用当前控件值。执行时保存独立快照，不覆盖源 YAML。',
    'tool-plan-path': '也可手动填写计划 YAML 路径，此路径优先于下拉框；清空恢复下拉选择，不改写原文件。',
}


# These names are argparse destinations, not arbitrary similarly named YAML keys.
TOOL_PARAMETER_HELP = {
    'preset': '选择内置 pipeline 配置作为起点，再应用 module、set 和 unset。切换预设会同时改变其声明的模块与参数，不会仅改变名称；它与 config、manual 是互斥的配置来源。',
    'config': '提供完整 pipeline YAML 作为配置来源，再应用命令中的显式覆盖。更换文件会改变执行配置而不修改源 YAML；它不是 study/comparison 计划，也不是 learned weights 文件。',
    'manual': '不用命名预设作为用户输入，按 module、set、unset 手工指定配置，并提供 config-id。当前 CLI 从 baseline 的规范默认值起步，未覆盖项仍继承该起点，不是创建空配置。',
    'module': '以 FAMILY=MODULE_ID 选择模块，可重复添加，例如 artifact=nlms_imu_anc。切换会应用该模块所需配置及默认参数，之后的 set 可进一步覆盖；不是立即运行这个模块。',
    'assignments': '对应可重复的 --set PATH=YAML，按完整字段路径覆盖参数；数值、true/false、列表和 null 均按 YAML 解析。相同字段后写值覆盖前值，并在模块选择之后应用；这会真实改变配置，不只是显示说明。',
    'unset': '输入要从配置中移除的完整字段路径，可重复添加，并在 module/set 之后执行。移除可选字段后原解析器可能补回默认值，移除必需字段可能报错，因此 unset 不等同于关闭模块。',
    'mode_config': 'config 只解析和检查配置；smoke/full 还预检记录清单、固定划分及模块约束。当前两者都解析全部划分，但预检报告分别汇总首个 split 与全部 25 个 split；均不训练，也不是完整信号试跑。',
    'mode_report': 'single 分析单组，comparison 比较多组，ablation 按指定因素分析消融，前三者读取折外预测；test 读取有独立测试证据的预测。切换改变报告组织和所需证据，不会把折外预测变成独立测试，也不重新训练。',
    'plan': '选择所执行工具支持的 study YAML，其中声明研究类型、输入、比较单位及相关参数。普通 sweep 与专项研究使用各自的计划格式；改路径会换整份任务定义，而非只替换单个模型参数。',
    'study_dir': '指定已有 study 输出目录，不是原始 CSV 或配置文件。重建索引读取其中产物，专项分析读取已有预测，Special CV complete 则据其计划补训尚未完整评估的候选；具体动作由所选工具决定。',
    'hash_predictions': '为预测文件额外计算并保存 SHA-256 内容摘要，便于检查文件是否改变。开启增加整文件读取和索引耗时，关闭不改变预测数值，也不会删除已有预测。',
    'pipeline_output': '指定已有 pipeline_output/<run>，从其中读取预测、fold 模型索引及配置以导出 Excel 或 model_config。换路径改变导出来源，不启动训练，也不是为新训练指定输出目录。',
    'replace': '允许替换已经存在的目标导出产物；不开启时同名目标会被拒绝。Excel 工具替换工作簿，model_config 工具替换整个对应导出目录而不是追加，源 pipeline 结果仍保留。',
    'input': '报告或执行审计读取的已有 pipeline 输出目录。切换目录会改变分析来源；execution-audit 可检查失败或中断任务的执行记录而不读取预测，specialized-report 则重建相应专项报告。',
    'source_root': '为 oracle、role-scope 以及运动/peak 专项提供解析相对输入路径的基准目录。它不是输出目录；当前超参数训练仍使用 V5 根目录，修改此项不会迁移数据或改写源文件。',
    'case_id': '专项决策偏置分析要读取的已运行 case 标识，用来覆盖计划中的来源 case。切换会换被分析的模型预测，而不是为现有产物改名；仅该 oracle 路线使用此覆盖，role-scope 不使用它。',
    'prediction_file': '专项决策偏置分析直接读取的 participant 层 OOF Parquet 文件，覆盖计划中的预测文件选择。显式指定可消除多个已完成 attempt 的歧义；须符合该分析所需的类别、repeat 和聚合格式，不是原始 CSV。',
    'step': '决策偏置 oracle 在三个非负且和为 1 的类别偏置上枚举的网格间距，必须能整除 1，例如 0.1。调小网格更密、计算与内存更多；它用同一批标签选优并评分，只用于乐观上限诊断，不是无泄漏的正式模型评估。',
    'upstream_study': '依赖式超参数研究读取的上游输出目录，其中应有 selected_configuration.json。下游按研究类型继承选中的 batch size、学习率及部分正则化参数，再施加本轮候选覆盖；不是加载预训练权重，也不是当前任务的 resume。',
    'device': '覆盖专项训练或完整 CV 补训使用的设备，例如 cpu、cuda 或 cuda:0；不填写时继承计划或原执行设置。切换影响速度及内存资源，不改变模型架构，跨设备可能有微小浮点差异；不适用于纯静态 peak 消融。',
    'no_denoiser': '勾选后仅跳过 Stage5-pre 的 PTT denoiser benchmark，运动模型训练与其他阶段仍执行。取消勾选会执行该 benchmark，通常增加耗时；这不是普通 pipeline 的 denoiser 开关，也不适用于其他专项类型。',
}


def parameter_help(spec: Mapping[str, Any], config_context: Mapping | None = None) -> str:
    """Describe a widget using its canonical path, including nested plan keys."""
    if spec.get('description'):
        return str(spec['description'])
    from .parameter_help_signal import SIGNAL_HELP, signal_help
    from .parameter_help_model import MODEL_HELP, model_help
    path = str(spec['path']).replace('~1', '.').replace('~0', '~')
    # Plans may wrap canonical overrides under cases.0.overrides.*.
    match = re.search(r'(?:^|\.)(signal|quality|artifact|features|windows|model|training|aggregation|evaluation)\.', path)
    canonical = path[match.start():].lstrip('.') if match else path
    parent, _, tail = canonical.rpartition('.')
    indexed = tail.isdigit()
    lookup = parent if indexed else canonical
    exact = MODEL_HELP.get(canonical) or SIGNAL_HELP.get(canonical) or PARAMETER_HELP.get(canonical)
    description = exact or signal_help(canonical, config_context) or signal_help(lookup, config_context) or model_help(lookup, config_context) or PARAMETER_HELP.get(lookup)
    if not description and lookup.startswith('windows.'):
        description = WINDOW_HELP.get(lookup.rsplit('.', 1)[-1])
    if not description and lookup.startswith('features.spectral_bands_hz.'):
        band = lookup.rsplit('.', 1)[-1].upper()
        description = f'{band} 脉搏间隔频谱的积分区间 [下限, 上限]，单位 Hz；改变边界会改变该频段功率和有关比值。扩大范围纳入更多频率成分，不代表信号或分类必然更好。'
    if description:
        return (f'列表第 {int(tail) + 1} 项（代码索引 {tail}）。' if indexed and not exact else '') + description
    leaf = lookup.rsplit('.', 1)[-1]
    aliases = {'run_name': 'run-name', 'output_name': 'report-name', 'jobs': 'job-count',
               'repeats': 'repeat-indices', 'folds': 'fold-indices', 'resume': 'resume-path',
               'bootstrap_resamples': 'evaluation.statistics.bootstrap_replicates',
               'permutation_resamples': 'evaluation.statistics.paired_permutation_replicates',
               'statistics_seed': 'evaluation.statistics.seed', 'alpha': 'statistics-alpha',
               'calibration_bins': 'calibration-bins', 'refit': 'train-flags', 'dry_run': 'train-flags'}
    target = aliases.get(leaf, leaf.replace('_', '-'))
    if target in FIELD_HELP or target in MODEL_HELP:
        return FIELD_HELP.get(target) or MODEL_HELP[target]
    if 'arg' in spec:
        if leaf == 'mode':
            choices = set(spec.get('choices') or ())
            if choices == {'config', 'smoke', 'full'}:
                return TOOL_PARAMETER_HELP['mode_config']
            if choices == {'single', 'comparison', 'ablation', 'test'}:
                return TOOL_PARAMETER_HELP['mode_report']
        elif leaf == 'preset' and '--presets' in spec.get('options', ()):
            return FIELD_HELP['analysis-preset']
        elif leaf == 'module' and '--modules' in spec.get('options', ()):
            return FIELD_HELP['analysis-modules']
        elif leaf in TOOL_PARAMETER_HELP:
            return TOOL_PARAMETER_HELP[leaf]
    if spec.get('help'):
        return f"CLI 参数 {spec.get('arg') or path}：{spec['help']}。该选项按原命令执行；不是统一的‘越大越好’评分。"
    return f'计划字段 {path}，原样传入所选脚本；其作用由该计划定义，不能仅从字段名推断算法效果或调参方向。'


def field_help(identity: str | None) -> str | None:
    """Meaning and operational effect of manually arranged UI controls."""
    return FIELD_HELP.get(identity) if isinstance(identity, str) else None
