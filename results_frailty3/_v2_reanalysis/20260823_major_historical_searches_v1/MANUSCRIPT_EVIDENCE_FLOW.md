# 论文证据链与写作顺序（Historical → V2 → Final locked）

## 结论先行

论文不能把所有历史与 V2 分数拼成一个总排行榜。历史 sweep、V2 调参、matched ablation、PTT cross-dataset benchmark 和最终 locked 5×5 回答的是不同问题，数据处理、模型选择权限和统计证据等级也不同。

四个历史 source 使用相同的五个 split seeds（42、10042、20042、30042、40042），且每个 seed 的五折 held-out participant roster 完全相同；这允许在协议兼容时做配对描述，但并不产生新的独立证据。尤其 20260608 与 20260625 的采样率、IMU 处理、class-weight 数据范围和基础正则不同，仍禁止合并排名。

建议把全文组织为一条逐步收窄的证据链：

1. 历史低可信搜索产生候选，不承担最终性能结论。
2. 历史 fixed-epoch 与 SQI/loss/feature 搜索产生机制假设。
3. V2 matched ablation 检查单个 pipeline 组件及算法模块。
4. V2 hyperparameter studies 冻结配置化 Inception route 的训练参数。
5. Stage1 在五条 representation/model route 间筛选；新增 Fusion 未运行前不得宣布冻结。
6. Stage5-pre 提供 motion/denoiser 外部 benchmark；Stage5 正式比较 SQI–motion–denoiser composition。
7. 所有选择冻结后，final locked 5×5 只做一次无进一步选择的内部确认。

目前证据链停在第 5–6 步之间：Stage1 需补跑默认 CNN Fusion；Stage5 SQI study 因五条短于 8 s 的 W1/W2 记录发生确定性失败。因此当前不能写“最终 representation 已冻结”“最终 SQI composition 已冻结”或“final model 已确认”。

完整数值索引见：

- `tables/paper_key_results.csv`
- `tables/paper_evidence_sequence.csv`
- `tables/v2_study_inventory.csv`
- `tables/paper_table_figure_plan.csv`
- `tables/confidence_methods.csv`

## Results 1：历史 architecture 搜索只负责产生候选

### Which test

- `20260527_1320_cnn_inceptionTime`
- `20260528_1045_shapeformer_0extra`
- 严格抽取共同合同：5 s、2.5 s hop（50% overlap）、patience 20、无 PPI/HRV extra input、相同 5 个 split seeds 和逐 fold participant rosters。

### 建议正文简表

| Model | Subject BA | Macro-F1 | BA repeat t-CI95 | Mean runtime |
|---|---:|---:|---:|---:|
| InceptionTime | 72.7 ± 6.0 | 72.3 ± 5.9 | [65.2, 80.2] | 397 s |
| CNN1D | 70.3 ± 4.5 | 70.8 ± 4.8 | [64.7, 75.9] | 173 s |
| ShapeFormer-PISD | 61.6 ± 4.5 | 60.5 ± 5.1 | [55.9, 67.2] | 4864 s |

正文只保留 subject-level 表；window/file 指标和 per-class 表放补充材料。配图使用 `early_three_model_ba_f1_boxplots.png` 和 log-scale runtime plot。

### 可以写的结论

InceptionTime 是 point-estimate leader，但与 CNN 的 5-repeat aggregate exact sign-flip test 不显著（BA raw P=.500；F1 P=.625）。ShapeFormer-PISD 在 5/5 matched repeats 均低于 CNN/InceptionTime，subject BA 分别低 8.7/11.1 percentage points，运行成本约为 CNN/InceptionTime 的 28.1×/12.3×。因此本项目不再让该历史 ShapeFormer-PISD 实现进入普通模型的共同大规模搜索；这属于效用/计算成本下的“不晋级”证据，不是对 ShapeFormer 算法族的普遍劣势证明。

### 必须同时写的限制

早期 archive 没有 participant-level OOF probability，不能合法补算 ROC/PR AUC、participant-cluster bootstrap、正式 participant permutation、calibration 或 t-SNE。并且 persisted fold history 与 archived generator 结构提示 held-out fold 被用于 best-epoch/early-stopping selection，应称 `legacy fold-held-out CV with selection contamination`，不能称 untouched OOF confirmation。

## Results 2：历史 fixed-epoch 搜索产生 regularization 假设

### Which test

`20260608_1206_overfitting_sweep_stage1_rank2`：186 个完整配置，每个 5 repeats，participant-grouped OOF，fixed epoch、无 early stopping。

### 建议正文简表

只展示 4–6 行：point leader `s1_085`、稳定性较好的 `s1_163`、ep10 baseline `s1_075`、50-epoch historical reference，以及一两个代表性 combined-regularization route。列仅保留：epoch、WD/dropout/LS、BA mean±SD、repeat t-CI、macro-F1、worst-class F1、train–validation BA gap。

### 结论

`s1_085` 的 BA 为 62.3 ± 7.0%，但相对同 epoch baseline 仅约 +1.85 pp，且来自 186-config post-selection。`s1_163` 点估计略低，却更稳定并有更好的 repeat-wise worst-class F1。50 epochs reference 与 5–15 epoch candidates 的差异同时混入长训练过拟合，不能被解释为某一正则化参数的纯效应。

因此本节只产生两个假设：固定短 epoch 值值得 V2 验证；WD/dropout/label smoothing 需要配置化联合研究。不要把 `s1_085` 写成 final winner。

## Results 3：历史 SQI/loss/feature 扩展产生 quality 假设

### Which test

`20260625_2320_overfitting_sweep_stage1_rank2`：129 个完整配置，每个 5 repeats。该 sweep 相比 20260608 同时改变了 64→400 Hz、ACC 重力处理、class-weight 数据范围和基础正则化，不能与 June 8 合并排名。

### 简表与结论

正文比较 `s1_122`（top50 SQI + quality weighting）与 `s1_102`（top50 SQI + mean probability），再加 matched baseline：两者 BA 均为 61.0%，但 `s1_102` 的 SD 为 2.1 pp，明显低于 `s1_122` 的 6.1 pp；quality weighting 只增加约 0.28 pp macro-F1，同时降低稳定性/worst-class evidence。

因此历史证据支持把 top50 quality selection 带入 V2 Stage5 作为假设，但不支持提前宣布 quality-weighted aggregation 优于 mean probability。manual features、loss 和 class-weight 也只能作为候选模块。

## Results 4：V2 matched component evidence

### Stage3 centered star

CNN 与 InceptionTime 必须分成两张 B0–B7 表，再给一张逐 profile 横向模型表。B2 对 CNN 为 +3.1 BA pp，但对 InceptionTime 为 −5.8 pp；B7 对两者均为小幅正向（CNN +0.9、Inception +0.6 pp）。这说明组件效应具有 architecture interaction，不能把某个 B profile 写成“对所有 DL 模型有效”。

推荐主文只画 B0-centered delta plot；完整 native BA/F1、worst-class F1 与 paired inference 放补充材料。Bridge run 只用于解释 V1→V2 迁移顺序，不与 centered star 合并成 ablation 结论。

### Peak detector ablation

使用 final v3：MSPTDfast v2.3 Python port 在 IR 上 recording median F1 99.7% [99.4,99.8]，PPI–RR RMSE 19.2 [15.8,27.7] ms，execution 0.006%；`aboy_project` 分别为 97.8% [90.8,98.8]、35.2 [31.0,38.4] ms、0.031%。IR F1 的 Holm–Sidak adjusted P=1.18×10⁻⁵；RED adjusted P=.00335。由此可把 MSPTDfast 冻结为默认模块，`aboy_project` 保留为显式 ablation。

## Results 5：V2 hyperparameter studies 冻结 configured Inception route

### Batch/LR

`b16_lr3e-4` 的 participant OOF BA/F1 为 59.5 ± 6.2% / 59.4 ± 6.3%，repeat t-CI 分别 [51.9,67.2]/[51.6,67.3]，cluster CI 更宽。它是 persisted development choice，但候选对比均无 Holm-adjusted superiority。因此论文措辞应为“selected for downstream evaluation under the predefined development rule”，不能写“proven optimal”。

### Regularization

R9 的 point BA/F1 为 58.9 ± 4.0% / 59.4 ± 4.6%，R2 为 58.1 ± 3.3% / 57.5 ± 4.1%；R9 vs R2 adjusted P=1.0。保持 persisted R2（WD=.001、dropout=.5、LS=.2）并把 R9 写成 sensitivity leader，避免在看到结果后重写选择历史。此 study 是三参数 joint-profile grid，不允许写出单个 WD/dropout/LS 的因果结论。

## Results 6：Stage1 冻结 representation/model（当前 pending）

现有四路线 run 的 point order 为 configured Full Inception 56.6 ± 5.2%、CompactCNN 51.5 ± 3.9%、Logistic feature vector 48.4 ± 3.2%、Matrix Small Inception 44.7 ± 4.3%；任何 declared contrast 都没有 Holm-adjusted superiority。新增的默认 `CNN Fusion` 尚无数据，所以现有结果只能作为 provisional screen。

Stage1 新 run 完成后，建议正文表仅保留五行和以下列：representation/model、BA/F1 mean±SD、participant-cluster CI、macro ROC-AUC、worst-fold BA、worst-class F1、参数量/推理时间。主图使用五路线 BA/F1+CI；ROC curves、confusion、calibration 放补充材料。

因为 configured Full Inception 已使用 Stage6 调参而其余四条为默认参数，本测试是“route screening with one configured candidate”，不是纯 representation 单因素消融。论文必须准确命名。

## Results 7：Stage5 冻结 quality composition（当前 pending）

### Stage5-pre 可用证据

在 matched reverse-ablation report 中，Frailty29-trained detector 的 Frailty29 file-level OOF BA/F1 为 96.8/96.8%，向 PTT22 转移为 78.4/71.2%，主要损失来自 motion sensitivity=.568。PTT22-trained detector 在 PTT OOF 为 100%，但向 Frailty29 转移时把全部数据判为 motion，file BA=.500、specificity=0，证明其 threshold/domain transfer 失败。PTT 结果应称 complete cross-dataset external benchmark，不是 untouched independent validation。

Denoiser 的 dynamic IR subject-macro endpoint 中，PCA-BSS 的 PPI–RR RMSE 为 71.4 ± 43.4 ms、F1 93.7 ± 5.3%；FastICA 为 71.7 ± 39.9 ms、F1 93.8 ± 5.0%，并有更多 reducer failures/更高运行成本。因此 PCA-BSS 是按预定 RMSE 主排序得到的 candidate，FastICA 保留并行 ablation。

### Stage5 正式 composition 尚不可解释

`20260823_111337...` 没有有效训练结果：五条 W1/W2 记录仅 6.88–7.66 s，短于 8 s SQI/peak window，所有 repeat/fold 会在长时间 CPU EKF 后重复失败。修复并重跑后，Stage5 reporter 应同时以 conditional 与 abstention-aware BA/F1、participant coverage、class-specific abstention、retained windows/files/participants、post-denoise Q_rate recovery、reducer failure、HR/PPI error 和 worst-class performance 决策。

后续 `20260823_205429...` 在任何 cell 完成前中止；`20260823_205510...` 仅完成 off/off 的单个诊断 fold，随后 SQI-only 再次以 `HR/PPI requires at least 8 seconds of observation` failed-closed，并在下一 case 中止。二者均只保留为失败审计，不能贡献 Stage5 效果分数，也不能改变上述 pending 状态。

## Results 8：Final locked 5×5

只有 Stage1 和 Stage5 均完成并形成书面 freeze manifest 后才运行。Final locked study 不再比较窗口、模型、SQI、motion threshold、denoiser、LR、batch 或正则化；它只估计整个冻结 pipeline 的内部表现。

主表应报告：participant OOF BA/macro-F1 mean±SD、repeat t-CI、participant-cluster bootstrap CI、macro ROC/PR AUC、worst-fold BA、worst-class recall/F1、ECE、coverage 和 operational cost。主图使用 ROC/PR、confusion matrix、calibration 和 repeat stability。任何 final 结果之后的再选择都必须另立新 study，不能回写本次确认。

## 统一 ranking 与可信度规则

1. Frailty primary ranking：participant-level abstention-aware balanced accuracy。
2. Secondary：participant macro-F1；随后审查 cluster LCB95、worst-fold BA、worst-class recall/F1、ECE、coverage 和 cost。
3. Denoiser primary endpoint：subject-macro PPI–RR RMSE；beat F1 是辅助 endpoint。
4. Motion detector 必须同时报告 window/file BA、macro-F1、sensitivity/specificity、ROC-AUC、PR-AUC；不能用 participant aggregation 替代用户指定的 file/window endpoints。
5. Repeat t-CI 使用 `mean ± t_(0.975,n−1)·s/√n`，但 repeated CV folds/participants 并非独立样本；这也是不能把普通 fold SD 当正式 CI 的原因。[Student (1908)](https://doi.org/10.1093/biomet/6.1.1)、[Bengio & Grandvalet (2004)](https://www.jmlr.org/papers/v5/grandvalet04a.html)。
6. V2 participant-cluster bootstrap 以 participant 为 cluster、保留其全部 repeats，10,000 resamples、seed 42、percentile 95% CI。[Efron (1979)](https://doi.org/10.1214/aos/1176344552)。
7. V2 paired P 以 participant 为 exchange unit、保留全部 repeats，100,000 permutations、seed 42；comparison-family×metric 内作 Holm correction。[Ojala & Garriga (2010)](https://www.jmlr.org/papers/v11/ojala10a.html)、[Holm (1979)](https://doi.org/10.2307/4615733)。
8. BA 用于不均衡类别的平均 recall；macro-F1 同权衡量各类 precision/recall；macro OvR ROC-AUC 只能由连续 OOF scores 计算。[Brodersen et al. (2010)](https://doi.org/10.1109/ICPR.2010.764)、[Sokolova & Lapalme (2009)](https://doi.org/10.1016/j.ipm.2009.03.002)、[Hand & Till (2001)](https://doi.org/10.1023/A:1010920819831)。
9. 调参 CV 的 point winner 不是无偏 final estimate；所有选择步骤必须嵌入 resampling，最终另做 locked confirmation。[Varma & Simon (2006)](https://doi.org/10.1186/1471-2105-7-91)、[Cawley & Talbot (2010)](https://www.jmlr.org/papers/v11/cawley10a.html)。

## 推荐的正文/补充材料分配

正文保留 6 张简表：历史三模型、历史假设摘要、Stage3 component delta、Stage6 freeze、Stage1 final five-route、Stage5 composition/final locked。正文图不超过 7 张：历史 BA/F1、历史 runtime、Stage3 centered delta、Stage1 route comparison、Stage5 detector transfer、Stage5 composition coverage-performance、final ROC/calibration panel。

完整 leaderboard、per-class metrics、所有 confusion matrices、所有 ROC/PR curves、learning curves、threshold/score/t-SNE、reproducibility splits/seeds、controlled/varied parameters 和 P-value families 放补充材料。这样能保留审计完整性，又不会让正文被数十个 development configurations 淹没。
