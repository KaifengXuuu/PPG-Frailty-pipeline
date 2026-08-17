# 52份代码/Notebook I/O 总索引 / Master Code and Notebook I/O Index

- 状态 / Status：`complete`
- 证据 / Evidence：`CODE_FILES.jsonl` 52行，全部逐字节至EOF并记录SHA；根目录29份、非根归档23份。
- 详细I/O / Detailed I/O：当前根文件见 `ROOT_FILE_IO_INVENTORY.md`；归档见 `ARCHIVED_CODE_IO_INVENTORY.md`；M0逐法见 `M0_METHOD_REGISTRY.md`与crosswalk。

## 1. 覆盖矩阵

| 分组 | 数量 | 人工I/O记录 | 逐脚本算法图 | 使用状态 |
|---|---:|---|---|---|
| M0根代码 | 16 | Root inventory + M0 registry/crosswalk | `m0/05_SCRIPT_ALGORITHM_ATLAS.md` | 历史审计完成；无严格有效full-waveform路线 |
| 非M0根Python/Notebook | 13 | Root inventory | `baseline/01_NON_M0_ROOT_SCRIPT_ATLAS.md` | 当前/未来TODO入口；部分存在泄漏、未执行或不可编译 |
| 非根归档代码/Notebook | 23 | Archived inventory | `baseline/02_ARCHIVED_SCRIPT_ATLAS.md` | 仅历史lineage和输出归因 |
| **合计** | **52** | **52/52** | **52/52** | 图覆盖另由严格JSON校验 |

## 2. M0根代码（16/16）

1. `funcs.py`
2. `ppg.py`
3. `cnnppg_v7.py`
4. `pttppg_pipeline_v7.py`
5. `pttppg_pipeline_v7_4_noleak_viz_ae.py`
6. `pttppg_denoiser_v8_masknet.py`
7. `pttppg_stage2_denoiser.py`
8. `pttppg_detector_v8_scores_audit_fix9.py`
9. `pttppg_denoiser_hybrid_core.py`
10. `pttppg_denoiser_hybrid_train.py`
11. `pttppg_denoiser_hybrid_preview.py`
12. `pttppg_denoiser_hybrid_ab_compare.py`
13. `pttppg_denoiser_hybrid_export_onnx.py`
14. `pttppg_denoiser_onnx_runtime.py`
15. `ppg_denoiser_dash_utils.py`
16. `ppg_peak_hr_gating_train.py`

## 3. 非M0根入口（13/13）

1. `frailty_3class_classifier.py`
2. `frailty_3class_cnn_fusion.py`
3. `frailty_3class_overfitting_sweep.py`
4. `frailty_3class_holdout_eval.py`
5. `analyze_sweep.py`
6. `shapeformer_port.py`
7. `asa_classifier.py`
8. `svm2_dataset_train.py`
9. `PPG_Analy_Visual_test.ipynb`
10. `ppg_analyse3.ipynb`
11. `ppg_analyse4_calib.ipynb`
12. `svm2_dataset_train.ipynb`
13. `template_test.ipynb`

## 4. 非根归档入口（23/23）

1. `archiv/frailty_3class_classifier - Copy_8channels_08062026.py`
2. `Arc/ppg_analy2.ipynb`
3. `Arc/ppg_with_detector_v8.py`
4. `Arc/ppg_with_detector_v8_npz_select.py`
5. `Arc/ppg_with_detector_v8_npz_select_viz.py`
6. `Arc/pttppg_dash.ipynb`
7. `Arc/pttppg_detector_v8_scores.py`
8. `Arc/pttppg_detector_v8_scores_audit.py`
9. `Arc/pttppg_detector_v8_scores_audit_fix2.py`
10. `Arc/pttppg_detector_v8_scores_audit_fix3.py`
11. `Arc/pttppg_detector_v8_scores_audit_fix6.py`
12. `Arc/pttppg_detector_v8_scores_audit_fix8.py`
13. `Arc/pttppg_pipeline_v7_2_noleak_viz.py`
14. `Arc/pttppg_stage1_detector.py`
15. `Arc/pttppg_pipeline_v7_3_noleak_viz.py`
16. `Arc/svm_dataset_train.ipynb`
17. `PPG_Testing_05_01_2026/Archive/ptt_ppg_dataset_analysis.py`
18. `PPG_Testing_05_01_2026/Archive/ptt_ppg_dataset_analysisv2.py`
19. `PPG_Testing_05_01_2026/Archive/ptt_ppg_dataset_analysis_esther.py`
20. `PPG_Testing_05_01_2026/Archive/ptt_ppg_dataset_analysis_fingertiponly.py`
21. `PPG_Testing_05_01_2026/Archive/7-8-2025/ptt_ppg_dataset_analysis_fingertiponly.py`
22. `PPG_Testing_05_01_2026/Archive/FilteredWalkTest/FilteredWalkTest.ipynb`
23. `PPG_Testing_05_01_2026/ptt_ppg_dataset_analysis_16July2025.py`

## 5. 主索引规则

1. 代码身份以manifest的路径、字节数和SHA为准；相同basename不自动视为同版本。
2. 归档lineage修复版本不重复计为独立科学方法；严格重复SHA只计一个算法实现。
3. “实际输出对应”必须来自路径/schema/字段或保存运行证据；只有相似代码时标为lineage而非精确生产者。
4. 每个新TODO开始重扫相关代码；manifest漂移后必须更新本索引和相应图册。

