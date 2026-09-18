---
title: "PPG Frailty Pipeline V5：信号、质量与表征算法逐步详解"
subtitle: "直觉说明 · 数学步骤 · 源代码对照"
lang: zh-CN
toc-title: "目录"
---

# 阅读说明

本文面向希望人工审阅算法的读者，解释 V5 从同步 PPG/IMU 到模型输入之间的三个阶段：信号预处理、质量与运动处理、特征和表征。每个可选计算方法分别解释，不把同一模块中的不同备选项合并成一个黑箱。分类模型本身的训练、最终概率聚合和 report 统计不在本文主体范围；运动检测模型属于质量处理，因此包含在内。

“直觉说明”用曲线、尺子、方向、滑动纸框等日常比喻解释目的；“代码与数学”保留算法名称、公式、形状、单位、默认参数和异常分支。比喻用于理解，公式与源码才定义实际计算。算法不必总能恢复真实脉搏；输出整齐或重复性高并不等于生理真实。

## 代码定位约定

完整路径相对 `final_pipeline_v5/`，例如 `src/ppg_frailty/signal/preprocess.py:381–411` 表示该文件第 381 到 411 行。章节若声明省略共同的 `src/ppg_frailty/` 前缀，则将此前缀补回定位；`configs/` 路径始终相对 V5 根目录。行号对应编写本文时的源码快照；以后代码移动时优先通过同节提供的函数或类名定位。附录列出所引用源文件的 SHA-256，以区分“位置改变”与“计算实现改变”。这些摘要仅是文档的追溯信息，不是运行时环境锁或执行门禁。

代码对照按有意义的连续计算语句分组：每组注明行范围、变量和数学动作；不为 import、空行、类型声明逐行重复中文翻译。涉及数组方向、掩码、边界、插值、截断、失败处理及训练集拟合的语句，同样属于算法解释。第三方函数只说明本地调用所确定的输入、参数和数学含义，不把其他实现或教科书默认值当作本项目实际代码。

有些方法保存在独立函数、专项比较或历史适配路线中，不一定能在当前 finalcase 上直接启用；相应章节会区分“可调用的底层方法”“通用配置入口”和“当前启用的流程”。仅注册、尚无可用模型的 learned denoiser 不能视为已经实现。

## 统一符号与数组方向

| 符号 | 含义 |
|---|---|
| $f_s$、$n$、$t_n=n/f_s$ | 每秒采样点数、从零开始的点序号和秒数 |
| $x[n]$、$N$ | 一条通道在第 $n$ 点的数值、总点数 |
| $C$、$W$、$T$ | 通道数、窗口数、每窗口采样点数 |
| $m[n]$ | 第 $n$ 点是否有效；不等同于填补后是否为有限数 |
| $\mu$、$\sigma$、$\operatorname{median}$ | 平均值、标准差、中位数；具体计算范围由各节规定 |
| $Q_{25}$、$Q_{75}$、IQR | 第 25/75 百分位数、二者之差 |
| $\epsilon$ | 防止除零或退化的正数，各算法的实际值不同 |
| $[a,b)$ | 含起点、不含终点的 Python 切片区间 |

源记录通常为“时间点 × 通道”，raw 模型窗口为“窗口 × 通道 × 时间点”。表征转换会改变轴顺序，不能把相同元素总数当成相同输入。PPG 保留原始计数的意义直到明确的标准化步骤；加速度和角速度先转换为章节指定的物理单位。

## 按真实数据依赖阅读流程

```text
原始记录与有效点标记
  ├─ 同 participant 的 B 静态记录 → 静态校准参数
  └─ PPG 缺失处理/滤波 + IMU 单位/偏置/可选重力分离
       → 400 Hz 的多种信号视图
            ├─ 质量/运动判断 → direct、rate-only 或丢弃等路线
            ├─ peak/PPI/PRV/形态/双光路与统计特征
            │    ├─ feature-vector
            │    └─ feature-matrix
            └─ 分窗及逐窗归一化 → raw 或 fusion 的波形输入
                 → 可选深度输入重采样与 fold 内输入变换
                 → 模型输入边界

cache 在确定性计算边界保存并读取数组，不另加一个信号处理公式。
```

这是数据依赖图，不表示所有分支串行执行。质量计算本身会调用峰与波形特征；特征章节解释这些被复用的内核。预处理章节为便于比较，集中说明重采样，但 finalcase 的深度输入重采样实际位于 raw 分窗及逐窗归一化之后。步骤顺序不同，即使使用同名算法，也可能产生不同结果。

## finalcase 与可选模块的关系

`configs/presets/finalcase.yaml` 与 `configs/studies/finalcase.yaml` 指向八通道 raw、全部 B/R/S/W 角色、5 秒窗口、2.5 秒步长、每文件最多 128 窗、深度输入 64 Hz 的方案。`quality.mode=off`、运动检测关闭、denoiser 关闭、reducer 为 identity。质量关闭不等于取消文件读取的物理有效性检查。

因此，本文对 SQI、运动检测、各种 reducer、peak/PRV/形态和 feature-vector/matrix 的详细介绍是在说明现有可选模块，不能解读为 finalcase 每次训练都计算并使用这些特征。`sensor_filter_only_no_gravity_removal` 的含义是“不去重力”，并非“不做同 participant B 校准”。最终以调用路径和 resolved 配置共同判断执行行为。

## 配套文本与 Word 使用

Word 主文件为 `docs/V5_ALGORITHM_GUIDE.docx`；本目录的 Markdown 是同内容的可审阅文本来源，不参与 pipeline 执行。使用 Word 的导航窗格或目录定位算法；目录页码可在 Word 中通过“更新整个目录”刷新。

在 V5 根目录已安装 Pandoc 时，可从文本重新生成 Word：

```bash
pandoc docs/algorithm_guide/00_reading_guide.md \
  docs/algorithm_guide/01_preprocessing.md \
  docs/algorithm_guide/02_cache.md \
  docs/algorithm_guide/02_quality_artifacts.md \
  docs/algorithm_guide/03_features_representations.md \
  docs/algorithm_guide/99_source_index.md \
  --from markdown --standalone --toc --toc-depth=3 \
  --output docs/V5_ALGORITHM_GUIDE.docx
```

本文不修改源码、默认值、采样率、模型权重或数据。行号和公式的静态核对不替代真实数据上的算法有效性验证，也不意味着每个可选模块均经过新一轮运行测试。
