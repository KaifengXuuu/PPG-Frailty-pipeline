# 2. 确定性预处理 cache

## 2.1 作用与可复用边界

**直觉说明。** 同一张长纸条，按同一种规则擦掉细小抖动、剪成同样长的片段，做第二遍应该得到同样的纸片。cache 就是把第一遍的纸片放进写有原料和做法的抽屉。抽屉存在便取出来；没有就重新做。它既不猜分类答案，也不把另一个人的答案抄给当前人。

**输入与输出。** 输入为源记录身份、该记录或同 participant B 的无标签信号、当前计算参数及上游计算身份；输出是计算函数原样返回的数值数组和必要说明。它不改变滤波、分窗、标准化等数学公式。

源码 `src/ppg_frailty/data/preprocessing_cache.py:1–7,47–54` 定义四个缓存层。`experiment.py:308–351,413–423,953–969,1533–1558` 是校准、信号视图、运动窗口和 raw 窗口的实际调用点（此处及下文省略共同前缀时，均为 `src/ppg_frailty/` 下的文件）。

| 层名 | 存放什么 | 明确不存放什么 |
|---|---|---|
| `imu_calibration` | 同一 participant 的 B 静态偏置、初始方向及校准配置 | 衰弱标签、从分类训练学出的参数 |
| `canonical_signal_views` | 原始/滤后/分析 PPG、有效和修补标记、处理后 IMU | 伪影削减后的波形、质量路由选择 |
| `raw_windows` | 未加路由筛选的 raw 窗口、有效掩码、起始点 | 标签、质量权重、最终聚合保留掩码、fold 内变换 |
| `motion_windows` | 运动模型所需的初始 8 秒/2 秒窗口 | fold 拟合的 IMU 缩放、运动概率、阈值与路由结果 |

`preprocessing_cache.py:799–819` 明确记录不共享的拟合状态。静态校准虽然名字含 `fit`，读取的是同一 participant 的 B 信号，不使用其衰弱标签；这不同于把所有人的训练统计量用于测试者。使用该校准仍意味着部署时需要同 participant 的 B 记录。

## 2.2 三种读写模式

**直觉说明。** `off` 是每次重新做；`read_only` 是允许取抽屉里的现成纸片，但不往抽屉增加东西；`read_write` 是既取现成纸片，也把新做的存起来。

**代码逐步对照。** `PreprocessingCacheSession._operate` 位于 `src/ppg_frailty/data/preprocessing_cache.py:290–334`：

1. 295 行开始计时，296 行检查当前层是否被选中。
2. 297–301 行：总开关关闭或该层未选中时，直接调用 `builder()`，返回其数组和说明，不读写缓存条目。
3. 302–311 行：`read_only` 先 `load(identity)`；只有 `CacheMissError` 才重新计算，不发布条目。损坏、源文件变化等错误不是普通未命中，不能被这一分支当作成功读取。
4. 312–316 行：`read_write` 调用 `get_or_compute`；命中取出，否则计算并存储。
5. 317–334 行：记录耗时、键、数组占用字节、命中状态和返回值。这里的耗时是该 cache 操作范围内的耗时，不是整轮训练耗时。

把某阶段写成确定性函数 $y=f_\theta(x)$，三种模式都应输出同一个 $y$；差别只在是否重复求值、是否保存。这里 $\theta$ 是该阶段的已选参数，不代表训练得到的分类模型权重。若源数据或 $\theta$ 改变，不能沿用旧抽屉。

## 2.3 由什么确定缓存键

**直觉说明。** 抽屉标签不能只写“某个人的记录”，还要写清“哪份记录、用的哪把剪刀、先剪还是先擦、每片多长”。顺序、参数或原料变了，就属于另一份结果。

**代码与数学。** `src/ppg_frailty/data/recording_cache.py:136–208` 定义身份：

1. `NamedSourceDependency.to_payload`（142–147）保存依赖名称、内容摘要和记录属性。
2. `OrderedModuleSpec.to_payload`（157–169）保存模块在链中的位置、名称、版本、实现摘要、开关和参数。顺序也是身份的一部分。
3. `RecordingCacheIdentity.to_payload`（182–204）将源依赖按名称排序，但保留模块计算顺序，并加入输出结构和扩展信息。
4. `key`（207–208）对上述规范化表示计算 SHA-256。
5. `src/ppg_frailty/provenance.py:25–35` 的 `stable_payload_sha256` 使用键排序、紧凑分隔、UTF-8 和禁止 NaN 的 JSON，再计算字节摘要。

可写作：

$$K=H\bigl(\operatorname{JSON}_{\mathrm{canonical}}(I)\bigr).$$

其中 $I$ 是包含源依赖、顺序模块、参数、实现摘要、输出结构和扩展信息的身份对象。

$H$ 是 SHA-256，不是信号变换。`preprocessing_cache.py:63–127,131–158` 读取源文件/实现文件摘要，同一进程中会按设备、inode、文件大小和修改时间等稳定文件身份复用已经算出的摘要。绝对 checkout 路径不作为实现摘要内容，因此搬动相同代码不应仅因路径改变而被视为新算法。

`preprocessing_cache.py:247–288` 将实际源摘要与 manifest 中 `source_hash` 比较。`canonical_views` 还把上游 B 校准身份纳入依赖（450–465），所以不能更换 B 校准后继续用旧动态记录视图。NumPy/SciPy 版本在有关层的 `extra` 中纳入键。缓存自动失效是结果复用规则，不是限制用户升级依赖的运行环境锁。

## 2.4 校准层的存取

**直觉说明。** 静止时的尺子有固定偏差，先记住偏差，再测运动。下次测同一原始记录不必重新计算同一偏差，但不能借用另一人的静止记录。

`src/ppg_frailty/data/preprocessing_cache.py:336–438`：

1. 342–383 行构造以 B 源记录、IMU 配置、计算实现和单位为依据的身份。
2. 385–406 行调用真正的校准函数；保存三维加速度偏置 `acceleration_bias_mps2` 与三维角速度偏置 `gyroscope_bias_rads`，以及初始 roll/pitch、校准起止点、质量说明和参数。
3. 408 行通过统一模式分支读写。
4. 410–427 行恢复元组参数并构建 `MotionImuCalibration`，不再次估计偏置。
5. 433–437 行保存校准对象与上游键的对应关系，供后续信号视图层使用。

校准的数学计算详见信号预处理章；此处只有无损存取，不额外做平均或降精度。`experiment.py:327–336` 先在同 participant 合格 B 记录中按持续时长降序、record ID 升序选择来源，不能把任意静态文件当作同等可替换输入。

## 2.5 完整信号视图层

**直觉说明。** 同一张曲线有“原纸”“去掉慢漂和细抖后的纸”“供下一步看节奏的纸”，再加一张标记缺口的透明纸。cache 同时保存这些版本，防止后续模块混淆。

`src/ppg_frailty/data/preprocessing_cache.py:440–590`：

1. 450–537 行把目标记录、校准、滤波/缺失参数、物理检查、IMU 方法、400 Hz 和软件实现加入身份。
2. 539–543 行 `builder()` 返回视图，要求它仍是 pristine `direct`，没有 `x_ar`。这是缓存边界，不是说算法只能使用 direct。
3. 544–552 行保存 `x_native`、`x_filter`、`x_analysis_rate`、`source_valid_mask`、`repair_mask` 和每个 `imu_processed` 数组。
4. 553–562 行保存路由、物理检查和元信息。
5. 564–578 行读取并重建 `CanonicalSignalViews`；`x_ar=None`，因为此层没有缓存下游伪影削减结果。

这些数组可能数值相等，例如未削减的分析视图与滤后视图；当前实现仍按不同名字分别写入数组文件。相同内容不会自动合并为一份。

## 2.6 raw 窗口层

**直觉说明。** 在长曲线上放一个有固定宽度的纸框，沿时间轴挪动，把每次框内的八条曲线抄下来。cache 保存的是这些纸片，而不是看过答案后挑出的纸片。

`src/ppg_frailty/data/preprocessing_cache.py:592–678`：

1. 601–643 行：键包含上游信号视图键、`WindowPlan` 全参数和 normalization 配置；输出轴声明为 `N_8_T`。
2. 645–649 行：先执行 `build_raw_windows`；若已经有 `window_quality_scores` 或 `window_aggregation_mask`，不能作为此层原始缓存。
3. 650–661 行：保存 `values`、`valid_mask`、`start_samples`；另存候选数、无效丢弃数和处理来源。
4. 663–678 行：按原结构重建 `RawWindows`。
5. 827–842 行：窗口数组必须是 float32 的 $W\times8\times T$，掩码为 $W\times T$，起点为 $W$ 个递增整数；检查这些条件不改变数值。

`experiment.py:1548–1558` 在这里读缓存，1559 行之后才按 `routing_timeline` 筛选；`experiment.py:1950–1972` 所在的 `_prepare_dl_input_dataset` 才处理深度模型输入采样率。因此 finalcase 的这一缓存是 400 Hz 窗口，不是已经变成 64 Hz 的模型张量。

## 2.7 运动窗口层

**直觉说明。** 判断是否在动，需要另一种长度的纸框。它与最终分类用的纸框不同，分别存放；存进去前还没有根据训练人群改变尺子的刻度。

`src/ppg_frailty/data/preprocessing_cache.py:680–791`：

1. 698–749 行：根据运动输入 profile、上游视图、实现、通道结构和采样率构造身份，明确 `fold_imu_scaler_applied=False`。
2. 751–766 行：保存窗口 `values`、`start_samples`、profile 和通道顺序。
3. 768–784 行：恢复张量；record、participant、role/activity、dataset 身份来自本次 manifest，而非缓存中的监督标签。
4. `experiment.py:953–969` 把预先构造的窗口交给 `infer_reused_motion_windows`；后续模型变换和预测不在此缓存内。

finalcase 的运动检测关闭，默认选择的缓存层也不含 `motion_windows`；把该层名称加进配置不会自行开启运动模型。

## 2.8 无损写入、读取及磁盘占用

**直觉说明。** 为省下重复劳动，这里选择把每张纸片原样装进文件夹，而不是把它缩成模糊小图。这样读取方便，但占空间。重复读取同一抽屉不会生成 25 份同样内容。

`src/ppg_frailty/data/recording_cache.py` 的逐步存储过程：

1. `_prepare_array`（236–252）转换为连续 C 顺序；保留 dtype 和 shape。内容摘要包括 dtype、shape 和数组字节，而不仅是打印出的数字。
2. `_load_verified`（322–426）检查元数据、完整提交标记、各文件长度/摘要、shape/dtype；397 行 `np.load(..., allow_pickle=False, mmap_mode="r")` 用只读映射加载，415 行设置数组不可写。
3. `get_or_compute`（446–464）先查找；不存在时在进程互斥区再次查找，然后调用 builder 并发布。互斥只防止两个进程同时写同一条目，不冻结环境或算法选择。
4. `_publish`（484–533）在临时位置逐数组 `np.save`，写 metadata 与完成标记，最后重命名为正式条目。没有有损压缩，也没有把 float64 自动改成 float32。

数组理想占用量为：

$$B=\left(\prod_i d_i\right)b,$$

其中 $d_i$ 是每个轴的长度，$b$ 是每个元素的字节数。加上文件头、掩码、元数据和文件系统分配后，磁盘占用会更高。

例如 18,013 个 finalcase 样式 raw 窗口，每个为 5 秒 × 400 Hz × 8 通道 × float32，仅 `values` 即占

$$18{,}013\times2{,}000\times8\times4=1{,}152{,}832{,}000\text{ bytes}.$$

完整记录的多个 float64 PPG/IMU 视图还会另占数 GB。此例用于说明空间来源，不是任何输入都固定生成相同大小。改变上游参数/源数据/实现后会产生新键，旧条目不会自动清除；namespace 关闭也不会删除历史文件。

## 2.9 配置入口、默认值与核查

| 入口 | 开关与默认 |
|---|---|
| `pipeline.py run` | `--preprocessing-cache-mode off/read_only/read_write`；未指定时通用执行默认 `off` |
| cache 目录 | `--preprocessing-cache-root`；通用执行默认 `artifacts/studies/cache`，路径在 V5 内 |
| 层选择 | `--preprocessing-cache-namespaces` 接逗号分隔名称；通用默认包含四层，但只在相应算法被调用时产生缓存 |
| `sweep.py` | study YAML 的 `execution.preprocessing_cache`；不是 pipeline 叶参数 `--set` |
| finalcase study | `read_write`，`cache/preprocessing`，校准/信号视图/raw 窗口三层 |
| `manual-cli` 输出 | 显式生成 `read_write` 与 `cache/preprocessing`，可直接修改生成命令 |
| Dash | 普通 run 用模式下拉框；sweep 遵循所选 study YAML |

默认来源：`src/ppg_frailty/study/schema.py:473–508`；CLI 参数和优先级：`src/ppg_frailty/v5/cli.py:128–130,208–232,388–390`；finalcase study：`configs/studies/finalcase.yaml:76–83`。

每 fold 的 `preprocessing_cache.json` 保存命中/新写/跳过、各层数组大小和耗时。`preprocessing_cache.py:793–824` 的 `logical_array_bytes` 是本次操作事件中访问数组的字节量之和，不可把 25 个 fold 的该字段相加后说成磁盘唯一占用；多个事件可能读同一条目。

关闭 cache 不改变已保存的预测、权重或 report。报告读取 pipeline 输出；导出模型的 raw 推理使用自身固定参数，不依赖这些预处理缓存文件。清理缓存应在没有任务使用它时另行进行，本说明不提供自动删除动作。
