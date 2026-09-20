"""Chinese UI help checked against the model factories and training consumers.

Display metadata only: no defaults, validation, computation or model loading.
List items reuse their parent's explanation; Adam's two betas have distinct help.
"""
from __future__ import annotations

from typing import Mapping


_MODEL = {
    'model_id': '选择分类器结构。卷积网络从波形学习，传统模型读取特征向量，ShapeFormer寻找有区分力的局部形状，fusion合并波形与文件特征；切换会改变参数结构和所需输入，不能把旧权重直接当成新模型权重。',
    'input_channel_order': '实际送入模型的信号通道及排列顺序。增删通道改变可用信息和输入宽度，调换顺序改变每个权重接收的信号；必须与所选权重训练时的通道含义一致，不是只改显示名称。',
    'input_channels': '模型实际接收的通道数，由所选信号或特征宽度确定。增加通道会改变首层结构和信息来源；这是输入结构记录，不应脱离实际通道清单单独修改。',
    'n_classes': '输出类别概率的列数，与项目类别顺序一致。改变数量会改变分类头，不能沿用原分类头权重；当前项目为三分类。',
    'seed_policy': '训练随机种子的来源：outer_repeat随外层repeat，fixed_explicit固定使用显式训练种子，member_roster按集成成员清单；历史长名称是对应别名，其中明确写出固定种子或五成员的名称仍保留该限制。切换改变随机过程，不代表准确率单调提高。',
    'seed': '显式模型随机种子，用于需要固定随机状态的模型构造。改变数值改变随机初始化或抽样序列，数值大小没有强弱含义；交叉验证实际种子还受seed_policy控制。',
    'member_seeds': '集成中各独立成员的随机种子，按列表顺序对应成员。成员分别训练后平均概率；修改数值改变成员随机过程，增加成员会增加训练和推理成本，不保证精度提高。',
    'ensemble_size': '由成员种子清单推导的集成规模，不是独立于清单的另一套开关。更多成员需要更多训练与推理计算；单模型与集成的输出不能仅靠改这个记录互换。',
    'variant': 'Inception的small/full基础结构档位。未显式给出的宽度和层数使用对应档位默认值；若这些字段已明确配置，单改档位名称不会覆盖它们。实际结构应以当前各项参数为准。',
    'kernel_sizes': '每个卷积核覆盖的连续采样位置数：CompactCNN列表对应先后三级卷积，Inception列表对应同层并行分支。增大可覆盖更长局部形状、增加计算；减小偏向短细节，不等于改变输入采样率。',
    'dilations': 'CompactCNN各级卷积取点的间隔倍数。增大可在参数数量不变时看到更宽时间范围，但中间取点更稀疏；减小更密集地观察局部，列表顺序与三级卷积一致。',
    'dilation': 'Inception卷积取点间隔倍数。增大扩大每个分支的时间覆盖范围但取点更稀疏；减小侧重紧邻样本。它不执行重采样，也不直接减少输出时间点。',
    'pool_sizes': 'CompactCNN前两级最大值汇聚的宽度，同时作为步长。增大使时间轴缩短更多、后续计算更少，但可能丢失短暂细节；减小保留更多时间点并增加后续计算。',
    'pool_size': 'Inception并行最大值分支的邻域宽度，代码步长固定为1。增大在更宽邻域取最大值，减小更局部；它与CompactCNN的pool_sizes不同，不会按该数值缩短时间轴。',
    'stage_channels': 'CompactCNN三级卷积分别输出多少组学到的信号。增大相应级可表示更多不同形状，但增加参数、内存和过拟合机会；减小使模型更轻，也可能限制表达能力。',
    'stage_dropouts': 'CompactCNN前两级汇聚后训练时随机置零的比例。增大减少模型依赖某些局部响应，但过大会妨碍学习；减小保留更多响应。预测时关闭这些随机置零，不是在原始信号中删除片段。',
    'dropout': '训练时随机置零部分内部响应的比例：CompactCNN/Inception主要在分类头，ShapeFormer还用于注意力等层，fusion用于融合层。增大加强随机扰动但可能欠拟合；减小减弱扰动。预测模式关闭Dropout，改此值不会重新训练既有权重。',
    'out_channels': 'Inception每个并行分支输出的通道数；拼接后宽度为此数×（卷积分支数+1）。增大增加表达空间、参数和计算；减小使网络更窄，可能遗漏细微差异。',
    'bottleneck_channels': 'Inception进入较长卷积前，用1点卷积将通道混合到的宽度。减小可压缩计算但也限制信息通路，增大保留更多混合分量并增加成本；不改变采样点数量。',
    'depth': 'Inception连续堆叠的多分支模块数量。增大增加处理层次、时间覆盖与计算，训练也可能更困难；减小更轻量。它是网络深度，不是重复训练次数。',
    'residual_interval': '每隔多少个Inception模块加入一次跨层相加通路。减小使这种信息直通更频繁，增大使每条通路跨越更多层；若大于depth则没有对应完整间隔的残差相加，不能视为单调增强。',
    'feature_hidden_dim': 'fusion中把文件级特征向量映射成多少个内部数值。增大允许保留更多特征组合但增加参数，减小形成更紧凑的表示；此文件特征分支只在文件层进入一次。',
    'fusion_hidden_dim': 'fusion把汇聚后的波形表示与文件特征拼接后，再变换出的表示宽度。增大增加两种信息的组合能力和计算，减小降低容量；不是给某个分支直接指定权重。',
    'pooling': 'fusion将一条记录的多个窗口汇成一个表示：mean对有效窗口等权平均；attention通过训练学到窗口权重。attention增加可学习参数，并非自动使用SQI作为权重；切换需要匹配结构的权重。',
    'logistic_c': '逻辑回归中对大系数约束强度的倒数C。增大C减弱约束，更贴合训练集但更易过拟合；减小C加强约束、系数更受限制，过小可能欠拟合。不是学习率。',
    'logistic_max_iter': '逻辑回归求解器允许的最大迭代次数。增大给尚未收敛的求解更多时间，减小可能提前停止；已提前收敛时增加上限通常不改变结果，不是交叉验证repeat数。',
    'logistic_solver': '求解同一逻辑回归目标的数值方法。lbfgs/newton类与sag/saga等在内存、速度和收敛行为上不同；某些方法对多分类或依赖版本有额外限制，不能按名称假定预测完全相同。',
    'svm_c': 'RBF支持向量机对训练误差的惩罚强度。增大更努力拟合训练样本、边界可能更曲折；减小允许更多训练误差并加强约束。与gamma共同作用，不保证越大越好。',
    'svm_gamma': 'RBF相似度exp(-gamma×距离平方)的范围参数。增大使单个样本的影响更局部，边界更易贴合细节；减小影响更宽。scale按输入维数和方差计算，auto为维数倒数；也可输入正数。',
    'svm_probability': '当前SVM分支固定输出概率，以便后续窗口/文件/参与者聚合；不是只输出类别的开关。概率校准属于训练模型的一部分，关闭不能直接代替现有概率工作流。',
    'extra_trees_n_estimators': '极端随机树的树数量，各树概率再平均。增大通常减少有限树数量带来的随机波动，但训练时间、模型体积和预测成本增加；减小更快，不能保证指标随树数单调变化。',
    'extra_trees_max_features': '每次树节点分裂考虑的特征数量：整数为个数，小数为比例，sqrt/log2按总维数计算，null使用全部。增大让每次分裂比较更多特征，减小增加树之间的随机差异；也会改变偏差与相关性。',
    'extra_trees_min_samples_leaf': '一个叶节点至少包含多少训练样本；整数为个数，小数为总样本比例。增大禁止非常小的叶子，预测更平滑但可能欠拟合；减小可形成更细分组，也更易拟合噪声。',
    'extra_trees_n_jobs': '随机树训练/预测使用的并行工作数量，-1表示使用可用处理器，正数指定数量。增加并行可能加速但提高CPU/内存占用；不改变树的数量或分裂公式，过度并行未必更快。',
    'input_fs_hz': '模型所声明的实际输入采样率，用于把采样点长度换算为秒，并与上游重采样结果核对。单改此数不会真正重采样；在点数不变时，数值越大表示覆盖的物理时长越短。',
    'sequence_length_samples': 'ShapeFormer构造时预期的一段输入长度，以重采样后的点数计。增大对应更长输入及更多位置处理，但必须与窗口实际长度一致；不是允许随意截断或补齐的开关。',
    'num_pip_ratio': 'ShapeFormer发现形状时保留的关键转折点占窗口长度的比例。代码按实际长度乘比例向下取整、至少5点并不超过窗口长度；增大通常生成更细的候选形状和更多搜索，减小更粗略。',
    'shapelets_per_class': '为每个类别发现并保留的代表性局部形状数量。增大让分类器比较更多形状，增加发现、训练和推理成本，也可能引入冗余；减小更紧凑但可能漏掉有用形状。',
    'max_discovery_windows': '仅在当前训练折内，参与ShapeFormer形状发现的窗口数上限。增大覆盖更多训练情况但提高搜索成本；减小更省算力、候选来源更少。这不是测试集窗口上限，也不允许使用外折标签。',
    'discovery_balance': '形状候选发现的取样方式：participant_file_balanced尽量兼顾类别、参与者和文件；class_window_balanced按类别分配窗口；legacy_class_window_balanced用于历史分支。切换改变候选来源，不是改变外层数据划分。',
    'position_search_neighbourhood_samples': '通道专属形状匹配时，在原形状位置附近允许搜索的额外点数。增大允许更大的时间偏移并增加比较位置；减小更严格保持原位置。不会把搜索扩展到其他参与者或测试标签。',
    'shapelet_search_window_samples': '历史ShapeFormer在原形状位置周围展开的匹配搜索范围，以采样点计。增大能寻找偏移更远的相似形状但增加比较，减小更局部；它不是旧代码中未被计算使用的局部卷积宽度标记。',
    'shapelet_length_samples': '固定长度形状模板包含的采样点数。增大比较较长形状、每次距离计算更重，减小关注较短局部；长度必须放得进实际输入窗口，并按input_fs_hz换算物理时长。',
    'discovery_stride_samples': '固定长度形状发现时，候选起点之间的步长；固定效应量分支还将此步长用于发现阶段的滑动距离搜索。增大搜索更稀疏更快，减小更密集更慢，可能改变入选形状。',
    'max_candidates_per_class': '固定效应量ShapeFormer每类最多评估的候选数量，超出时按种子随机抽取候选。增大评估更多候选、提高搜索成本；减小更省时但可能漏掉区分力强的模板，不是最终保留的模板数。',
    'candidates_per_class_channel': '历史ShapeFormer在每个类别、每条通道先保留的高分候选数量，再跨通道排序并选模板。增大扩大进入后续筛选的候选池，减小更强地筛除；最终数量由shapelets_per_class决定。',
    'local_kernel_width_samples': 'ShapeFormer局部卷积分支一次观察多少个连续采样点。增大覆盖更宽的局部形态并改变卷积结构，减小聚焦更短细节；与形状模板长度、形状搜索范围是不同参数。',
    'local_embedding_channels': 'ShapeFormer局部卷积分支产生的表示宽度。增大允许更多局部模式组合但提高参数和注意力计算，减小使局部分支更紧凑；需满足该结构的多头维数要求。',
    'shape_embedding_channels': 'ShapeFormer形状匹配分支投影后的表示宽度。增大给选中片段与模板的差异更多表达空间但增加参数，减小形成更紧凑的表示；不是多发现几个模板。',
    'hidden_channels': '标量距离/固定效应量ShapeFormer中，将输入小段映射到的表示宽度。增大每个小段能保存更多模式但耗用更多参数和内存，减小更轻；宽度需能被attention_heads整除。',
    'patch_size_samples': '标量距离/固定效应量ShapeFormer把信号切成互不重叠小段的点数，卷积核与步长都取此值。增大使注意力处理的小段数量减少、计算下降但时间细节变粗；减小保留更细时间位置、注意力成本增加。',
    'attention_heads': '注意力层把内部表示分成多少组并行比较。固定总宽度下，增大组数会减小每组宽度，不等于按倍数增加整体容量；必须符合宽度整除条件，切换会改变信息组合方式。',
    'attention_layers': '标量距离/固定效应量ShapeFormer连续使用的注意力编码层数。增大增加层次和计算，也更可能过拟合；减小更轻量。不是注意力头的数量。',
    'attention_feedforward_channels': '注意力模块中逐位置变换层的中间宽度。增大提供更多内部组合、增加参数与计算，减小压缩容量；不会改变输入的采样率或窗口数量。',
    'attention_query_chunk_size': 'OSD注意力计算每批处理多少个查询位置。增大减少分批次数但占用更多临时内存，减小节省峰值内存但可能更慢；仍与完整键值序列比较，不是截短注意力范围。',
    'distance_position_chunk_size': '计算形状距离时，每批同时比较多少个候选位置。增大提高批内并行度并增加临时内存，减小更省内存但分批更多；不会删除搜索位置或减少发现模板数量。',
    'complexity_norm': '形状距离计算向复杂度量中加入1/此数的小量，避免平坦片段导致不稳定比例。增大使这个小量更小，平坦片段的复杂度差异可能更突出；减小增强平滑作用，不是直接给信号除以此数。',
    'max_complexity_ratio': '形状距离中，两个片段复杂度差异所形成的乘数上限。增大允许粗糙程度差异产生更强距离惩罚，减小限制这种放大；取1会取消该乘数的差异放大，但不取消基础形状距离。',
}

_TRAINING = {
    'epoch_rule': 'fixed_epoch按固定轮数训练；inner_grouped_selection只用外层训练参与者再分组挑选轮数，然后在全部外层训练数据上从头训练。切换影响训练流程和耗时，不使用外层测试标签挑轮数。',
    'fixed_epochs': '固定轮数模式下遍历训练数据的轮数。增大提供更多权重更新但更耗时、可能过拟合；减小更快但可能未学充分。传统逻辑回归/SVM/树不按此深度训练轮数循环。',
    'maximum_inner_epochs': '内层挑选轮数时允许训练的最大轮数。增大允许搜索更晚的最佳点并增加潜在耗时；减小更早截止。固定轮数模式下该项为0，不会另开内层训练。',
    'inner_patience': '内层参与者级平衡准确率连续多少轮没有严格提高后停止搜索。增大愿意等待更久、可能越过平台期但更耗时；减小更早停止。固定轮数模式不使用它。',
    'inner_grouped_folds': '将外层训练参与者按类别分成多少组，再按当前repeat/fold选其中一组作内验证；不是把这些内折全部训练一遍。增大通常使本次内验证人数减少，减小使内验证比例变大；不能超过最小类别参与者数。',
    'batch_size': '一次权重更新使用的样本数：raw通常按窗口，fusion按文件袋。增大往往提高吞吐但增加内存、同一轮更新次数减少；减小更新更频繁且波动更大。模型预测也按此值分批，不会因此重新训练。',
    'learning_rate': '深度模型每次沿训练误差方向移动权重的步幅尺度。增大可能更快但易振荡或越过合适位置；减小更稳但需要更多更新。传统模型与已加载权重的纯预测不使用它更新参数。',
    'weight_decay': '深度优化器对大权重的抑制强度。增大使权重更受压缩，过大可欠拟合；减小约束减弱。AdamW采用独立权重衰减，其他所选优化器按各自实现处理，不能与逻辑回归的C等同。',
    'device': '深度训练/预测使用的设备，如cpu、cuda或cuda:0。切换改变计算硬件、速度及可能的浮点差异，不改变窗口或类别定义；请求不存在的GPU会报错，不会自动换成CPU。',
    'num_workers': '训练数据加载器的并行工作进程数，0在主进程加载。增大可能缓解数据供给等待，但增加内存和进程开销，缓存数据或小任务未必更快；不是模型集成数量。',
    'seed': '训练初始化、洗牌和抽样使用的随机种子；外层执行还会按model.seed_policy解析最终种子。改值改变随机序列，不代表更强训练，大小无优劣；仅改种子也不会重新训练已选权重。',
    'optimizer': '深度权重更新方法。SGD按梯度和可选动量移动；Adam/AdamW根据历史梯度调整步幅，AdamW独立施加衰减；RMSprop按历史梯度平方调整。切换会加载该优化器的参数，不是可直接比较的快慢档位。',
    'class_weighting': '训练中各类别误差的权重：none不额外加权，inverse_frequency让数量少的类别更有分量，effective_number按有效数量平滑调整。仅从当前训练数据计数；切换可能改善少数类也可能牺牲其他类。',
    'class_count_basis': '计算类别数量时，participant数独立参与者，row数训练行（可能是窗口或文件）。row会受每人记录数量影响，participant避免多窗口被当成更多人；改变会影响类别权重或balanced-softmax校正。',
    'training_balance': '训练的文件/角色平衡语义，与最终聚合线同步。equal_files按每人文件平衡，equal_role_families先平衡角色家族再平衡文件；真正按此分配抽样概率的是balance_line_weighted_v2，其他采样器保持各自规则。',
    'sampler': '每轮训练如何取数据：exhaustive随机排序且不重复遍历；uniform_replacement等概率可重复抽取；balance_line_weighted_v2按参与者/文件/角色权重抽取；subject_balanced按人再取配额；class_subject_balanced还平衡各类的人次。各策略改变样本重复与权重，不改变外折归属。',
    'samples_per_epoch': '可重复抽样策略每轮抽取的训练行数，null使用数据集行数。增大带来更多抽取与更新、可能多次抽到同一行；减小降低每轮成本。按人配额的采样器改用participant_window_quota。',
    'participant_window_quota': '按人采样时每次给该参与者取多少行：all取其全部行，整数为固定数量，小数或百分比为其行数比例。增大增加该人的取样量，超过现有行数会重复抽取；all并不保证不同参与者贡献相同行数。',
    'classifier_role_families': '哪些记录家族参加分类任务：B静态基线、R运动后恢复，S/W为项目运动任务家族；具体文件按角色映射筛选。增加家族纳入更多活动数据，移除则不用于分类，但其他记录的同人B校准需求独立于此选择。',
    'loss': '深度分类误差的计算方法：cross_entropy惩罚错误类别概率；focal_loss相对压低容易样本的贡献；balanced_softmax在训练分数中加入训练类别数量的对数。后者自行处理数量校正，不能再叠加class_weighting。',
    'focal_gamma': 'focal_loss中对容易样本降权的指数，乘数为(1-正确类别概率)^gamma。增大更强调尚未学好的样本，也可能更关注噪声；减小接近普通加权交叉熵，0取消这项降权。',
    'class_weight_beta': 'effective_number类别加权的平滑参数，权重按(1-beta)/(1-beta^类别数量)计算后归一化。接近1时趋近反频率校正；降到0时各已出现类别等权。只在effective_number策略下可调。',
    'label_smoothing': '深度训练中把一部分正确类别目标分给所有类别的比例。增大抑制过度自信但可能削弱类别区别，减小更接近硬标签，0为原始标签。不会修改数据标签或纯预测时的权重。',
    'gradient_clip_norm': '深度训练每次更新前允许的梯度整体长度上限；null关闭。减小会更常压缩过大的更新，增大更少触发裁剪；过小可能学得很慢。它不裁剪原始PPG或预测概率。',
    'deterministic_algorithms': '启用时要求深度后端采用确定性计算，并关闭cuDNN自动择优基准。关闭可能更快但重复运行差异更大；开启不保证跨GPU/依赖版本逐位相同，也不替代随机种子。',
    'optimizer_parameters': '当前优化器的专用参数，切换optimizer会换成相应参数组。它们只影响深度训练的权重更新；加载已有权重进行Analyse不会执行优化步骤。',
    'optimizer_parameters.betas': 'Adam/AdamW的两项历史记忆系数，顺序为梯度均值、梯度平方均值。增大让对应统计变化更平滑但对新情况反应更慢；减小更快跟随当前梯度，两项作用不同。',
    'optimizer_parameters.betas.0': 'Adam/AdamW对梯度方向平均的记忆系数beta1。增大更依赖长期方向、变化更平滑但转向更慢；减小更依赖当前梯度。不是学习率。',
    'optimizer_parameters.betas.1': 'Adam/AdamW对梯度平方平均的记忆系数beta2，用于按坐标调整步幅。增大让步幅尺度估计变化更慢，减小更快响应最近的大梯度；不直接控制梯度方向。',
    'optimizer_parameters.eps': '自适应优化器分母中防止除零的小正数。增大可限制分母很小时的步幅放大，但可能削弱自适应效果；减小通常影响很小，极端情况下会降低数值稳定性。',
    'optimizer_parameters.amsgrad': 'Adam/AdamW是否使用截至当前的历史最大二阶统计作为分母依据。开启会限制分母随近期统计下降、改变更新轨迹；关闭使用普通Adam二阶平均，不保证任一选择更准确。',
    'optimizer_parameters.maximize': '是否让优化器朝增大目标函数的方向更新。当前分类目标是损失，正常训练应关闭以减小损失；开启会反向追求更大损失，并非提高准确率的开关。',
    'optimizer_parameters.momentum': 'SGD/RMSprop更新中保留过去移动方向的强度。增大让方向更有惯性、可能更快通过平缓区但也可能过冲；减小更贴近当前梯度，0关闭相应动量。',
    'optimizer_parameters.dampening': 'SGD动量累积时削弱新梯度贡献的程度，通过(1-dampening)作用于新梯度。增大减少新信息进入动量，减小更快吸收当前梯度；启用Nesterov时此值必须为0。',
    'optimizer_parameters.nesterov': 'SGD是否根据动量前瞻方向修正更新。开启会改变动量更新公式，需要正动量且dampening为0；关闭使用普通动量，效果需实验而非越复杂越好。',
    'optimizer_parameters.alpha': 'RMSprop对梯度平方历史平均的记忆系数。增大使估计更平滑、对变化反应慢；减小更快追随近期梯度，可能让步幅波动更明显。',
    'optimizer_parameters.centered': 'RMSprop是否同时估计梯度均值，再用平方均值减均值平方估计方差。开启会改变自适应分母并多保存统计量；关闭只使用梯度平方平均，没有固定的优劣方向。',
    'epoch_profile': '由轮数策略派生的描述标签，用于记录当前训练方案；修改标签本身不是增加训练轮数，应调整epoch_rule及实际轮数字段。',
    'execution_mode': '训练执行记录中的模式字段，普通配置会规范化为formal；它不是切换另一套模型方程的开关，短跑调试由实际运行入口安排。',
    'cache_policy': '训练器内部状态缓存策略当前固定disabled，避免把折内拟合状态跨折复用。它不同于确定性信号预处理cache；关闭训练缓存不代表预处理cache也关闭。',
    'outer_labels_visible_to_trainer': '记录外层留出标签是否向训练器开放，当前固定false。留出标签不能用于拟合或内层选轮数；这是数据使用约定，不是供调参提高成绩的开关。',
    'refit_on_all_outer_training': '内层选好轮数后，在该外折的全部训练参与者上重新初始化并训练，当前固定启用。这里是外折内部流程，不是另一个面向全队列部署的可选--refit步骤。',
    'n_classes': '训练损失对应的类别数，与模型输出和数据类别顺序一致；当前为3，不能只改该数而沿用原有标签和分类头。',
}

MODEL_HELP: dict[str, str] = {f'model.{key}': value for key, value in _MODEL.items()}
MODEL_HELP.update({f'training.{key}': value for key, value in _TRAINING.items()})
MODEL_HELP.update({
    'aggregation.balance_line': '参与者预测的汇总方式：line_a先把窗口变成文件概率，再让各文件等权；line_b先在B/R/S/W家族内平均文件，再让现有家族等权。因此R记录较多时，两种方式给予R的总分量不同；切换不重新训练。',
    'aggregation.quality_weighting': '是否让质量分数参与概率加权。关闭按所选Line A/B等权平均；开启后低质量记录或窗口分量变小，可能改变类别概率，前提是存在与quality_weight_source匹配的实际分数。',
    'aggregation.quality_weight_source': '加权所用的质量来源：none不加权；route_file_q_rate从文件层起使用路线质量，窗口到文件仍普通平均；legacy_window_sqi先按历史窗口SQI汇成文件，再向上汇总。切换改变加权层级，不能把不存在的质量分数当作1。',
    'aggregation.hierarchy': '所选聚合线派生的层级顺序：窗口→文件→参与者，或窗口→文件→角色家族→参与者。它描述概率汇总对象，不是新一轮模型训练。',
    'aggregation.window_to_file': '一个文件内窗口概率的汇总方式，由质量来源派生为普通平均或历史窗口质量加权；改变聚合来源可能改变文件概率，但不会重新计算已保存的窗口模型输出。',
    'aggregation.file_to_role': 'Line B中，同一参与者同一活动家族的多个文件如何平均；Line A不执行此层。由聚合线和质量来源派生，避免文件较多的家族直接获得更多家族总权重。',
    'aggregation.role_to_participant': 'Line B中，将该参与者各现有活动家族的概率汇成最终概率；普通情况各家族等权，启用质量权重时按实际分数加权。Line A直接汇总文件，不执行此层。',
    'aggregation.missing_role_policy': '没有记录的活动家族不凭空补预测，按现有家族重新归一化平均；缺少家族会改变参与平均的对象，但不会把其概率当作0。',
    'aggregation.quality_weight_levels': '由quality_weight_source和Line A/B自动推导在哪些层加权。它是已选算法的层级记录，不是独立重复加权的开关。',
    'aggregation.direct_all_window_participant_mean': '当前正式pipeline不直接把全部窗口混成参与者平均，避免窗口多的文件天然占更多分量；此类对照只能作为明确的报告视图，不能用这个记录替换已选层级。',
    'evaluation.statistics.bootstrap_replicates': '统计置信区间的重抽样次数，按参与者保留其成组预测，而非把同一人的窗口当成独立人。增大使抽样估计通常更稳定但报告更慢，减小适合快速预览；不会训练更多模型。',
    'evaluation.statistics.paired_permutation_replicates': '配对比较中反复交换同一参与者两方案结果的次数，用于估计差异在零假设下的分布。增大提高随机估计精细度但耗时更长，减小加快预览；不是增加独立参与者样本量。',
    'evaluation.statistics.seed': '统计重抽样和置换所用的随机种子。相同输入、参数与环境下固定它便于重复统计结果；改变只改变抽样序列，数值大小不代表显著性强弱。',
    'evaluation.statistics.lcb95_percentile': '配置中记录置信下界采用的百分位，当前report计算固定使用2.5%下界，未读取这个可编辑字段；因此此处调大或调小不会改变现有报告的置信下界，也不会改变预测。',
    'evaluation.statistics.cluster_unit': '统计重抽样的整体单位是参与者及其各repeat预测，保留同一人的重复结果关联；这是当前统计方法的记录，不是把窗口数量改成人数的选项。',
    'evaluation.statistics.confidence_interval': '当前统计实现的区间类型为双侧95%百分位区间。此字段记录方法名称，不通过改字符串生成另一种区间算法。',
    'evaluation.statistics.lcb95_metrics': '记录哪些参与者级指标需要展示置信下界，不改变模型的预测概率或类别定义；现有具体输出仍由report模块选择和实现决定。',
    'evaluation.statistics.paired_exchange_unit': '配对置换按参与者整体交换两方案结果，同一人的各repeat不独立打散；保持实际依赖结构，避免夸大独立样本量。',
    'evaluation.statistics.multiplicity_correction': '同一比较家族采用Holm方法调整多次检验的P值，减少多次比较误报；字段是当前方法记录，改名称不会加载未实现的校正算法。',
    'evaluation.statistics.affects_automatic_selection': '统计结果不自动替人选定最终模型。当前为false，报告提供证据，不因某次P值或置信区间结果自动替换所选模型。',
})

# The fusion factory reuses these actual encoder parameters, not another formula.
_ENCODER_EXCLUDED = {'input_channel_order', 'input_channels', 'n_classes', 'seed', 'seed_policy',
                     'member_seeds', 'ensemble_size', 'feature_hidden_dim', 'fusion_hidden_dim', 'pooling'}
MODEL_HELP.update({f'model.signal_encoder.{key}': '融合中的波形编码分支：'+value
                   for key, value in _MODEL.items()
                   if key not in _ENCODER_EXCLUDED and not key.startswith(('logistic_', 'svm_', 'extra_trees_'))})
MODEL_HELP['model.signal_encoder.model_id'] = '选择fusion内部的波形编码器，再将其每窗表示汇成文件表示，与文件特征合并。切换CompactCNN/Inception/ShapeFormer改变波形分支结构和参数，不能沿用不匹配的分支权重。'
MODEL_HELP['model.signal_encoder.dropout'] = '融合波形分支的Dropout参数。CompactCNN/Inception的此项只在未调用的独立分类头中，fusion使用forward_features，因此对当前融合计算不起作用；ShapeFormer还在内部注意力层使用，训练时增大随机丢弃更强、预测时关闭。'
for _field in ('kernel_sizes', 'dilations', 'pool_sizes', 'stage_channels', 'stage_dropouts',
               'out_channels', 'bottleneck_channels', 'depth', 'dilation', 'pool_size', 'residual_interval'):
    MODEL_HELP[f'model.signal_{_field}'] = '融合中的波形编码分支：'+_MODEL[_field]
MODEL_HELP['model.signal_dropout'] = '固定Compact/Inception融合模型的信号编码器独立分类头Dropout。当前fusion调用forward_features而非该分类头，所以调此项不会改变融合分支输出；融合层本身的丢弃率由model.dropout控制。'


def model_help(path: str, config_context: Mapping | None = None) -> str | None:
    """Look up a full path, falling back to a numerical list item's parent."""
    del config_context  # Wording explicitly distinguishes shared-field consumers.
    value = MODEL_HELP.get(path)
    if value is None and path.rsplit('.', 1)[-1].isdigit():
        value = MODEL_HELP.get(path.rsplit('.', 1)[0])
    return value
