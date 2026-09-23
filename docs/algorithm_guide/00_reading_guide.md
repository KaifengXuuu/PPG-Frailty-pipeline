---
title: "PPG Frailty Pipeline V6: A Step-by-Step Guide to Signal, Quality, and Representation Algorithms"
subtitle: "Intuition · Mathematical steps · Source-code correspondence"
lang: en
toc-title: "Contents"
---

# How to Read This Guide

This guide is for readers who want to review the algorithms manually. It explains three stages between synchronized PPG/IMU recordings and model inputs in V6: signal preprocessing, quality and motion processing, and features and representations. Each alternative computation is explained separately, rather than treating different options within a module as a single black box. Classifier training, final probability aggregation, and report statistics are outside the main scope; motion detection models are included because they belong to quality processing.

The “Intuition” sections use everyday images such as curves, rulers, directions, and sliding paper frames to explain the purpose. “Code and mathematics” retains algorithm names, formulas, shapes, units, default parameters, and exception branches. Analogies aid understanding; formulas and source code define the actual computation. Algorithms cannot always recover the true pulse: a neat or highly repeatable output is not proof of physiological accuracy.

## Source Location Convention

Full paths are relative to `final_pipeline_v6/`. For example, `src/ppg_frailty/signal/preprocess.py:381–411` means lines 381 through 411 of that file. Where a section omits the shared `src/ppg_frailty/` prefix, prepend it to locate the source; `configs/` paths are always relative to the V6 root. Line numbers refer to the source snapshot used when the guide was written. If code moves, locate the function or class named in the same section first. The appendix lists SHA-256 hashes of referenced source files to distinguish moved code from changed computations. These hashes are documentation provenance, not runtime environment locks or execution gates.

Code correspondence groups meaningful consecutive computational statements: each group identifies line ranges, variables, and mathematical operations. Imports, blank lines, and type declarations are not translated individually. Statements governing array orientation, masks, boundaries, interpolation, clipping, failure handling, and training-set fitting are also part of the algorithm explanation. For third-party functions, only the inputs, parameters, and mathematical meaning specified by the local call are described; other implementations or textbook defaults are not presented as this project's actual behavior.

Some methods exist as standalone functions, dedicated comparisons, or historical adaptation routes and may not be directly enabled in the current finalcase. Their sections distinguish callable low-level methods, general configuration entry points, and currently enabled workflows. A registered learned denoiser without an available model must not be treated as implemented.

## Common Notation and Array Orientation

| Symbol | Meaning |
|---|---|
| $f_s$, $n$, $t_n=n/f_s$ | Samples per second, zero-based sample index, and time in seconds |
| $x[n]$, $N$ | Value of a channel at sample $n$, and total sample count |
| $C$, $W$, $T$ | Channel count, window count, and samples per window |
| $m[n]$ | Whether sample $n$ is valid; this is not equivalent to being finite after filling |
| $\mu$, $\sigma$, $\operatorname{median}$ | Mean, standard deviation, and median; each section defines the population used |
| $Q_{25}$, $Q_{75}$, IQR | 25th/75th percentiles and their difference |
| $\epsilon$ | A positive constant preventing division by zero or degeneracy; its actual value varies by algorithm |
| $[a,b)$ | Python slice interval including the start and excluding the end |

Source recordings usually have shape “time samples × channels”; raw model windows have shape “windows × channels × time samples.” Representation conversion changes axis order; equal element counts do not imply equivalent inputs. PPG retains its raw-count meaning until an explicit normalization step. Acceleration and angular velocity are first converted to the physical units specified in the relevant section.

## Follow the Actual Data Dependencies

```text
Raw recording and valid-sample markers
  ├─ Same participant's static B recording → static calibration parameters
  └─ PPG gap handling/filtering + IMU units/bias/optional gravity separation
       → multiple signal views at 400 Hz
            ├─ quality/motion assessment → direct, rate-only, or discard routes
            ├─ peak/PPI/PRV/morphology/dual-optical-path and statistical features
            │    ├─ feature-vector
            │    └─ feature-matrix
            └─ windowing and per-window normalization → raw or fusion waveforms
                 → optional deep-input resampling and within-fold input transforms
                 → model-input boundary

The cache saves and reloads arrays at deterministic computation boundaries;
it does not introduce another signal-processing formula.
```

This is a dependency diagram, not a claim that all branches execute serially. Quality assessment itself calls peak and waveform features; the feature chapter explains those shared kernels. For comparison, the preprocessing chapter collects the resampling methods in one place, but finalcase deep-input resampling actually occurs after raw windowing and per-window normalization. A different order can yield different results even when algorithm names are identical.

## finalcase and the Optional Modules

`configs/presets/finalcase.yaml` and `configs/studies/finalcase.yaml` specify eight-channel raw input, all B/R/S/W roles, 5-second windows, a 2.5-second hop, at most 128 windows per file, and 64 Hz deep input. `quality.mode=off`, motion detection and denoising are disabled, and the reducer is identity. Disabling quality processing does not disable physical-validity checks during file reading.

Thus, the detailed descriptions of SQI, motion detection, reducers, peak/PRV/morphology, and feature-vector/matrix describe available optional modules; they do not mean that finalcase computes and uses all these features in every training run. `sensor_filter_only_no_gravity_removal` means “do not remove gravity,” not “omit same-participant B calibration.” Actual execution must be determined from both the call path and the resolved configuration.

## Companion Text and Word Document

The main Word document is `docs/V6_ALGORITHM_GUIDE.docx`. The Markdown files in this directory provide the same content as reviewable text and do not participate in pipeline execution. Use Word's navigation pane or table of contents to find algorithms; refresh page numbers with “Update entire table.”

With Pandoc installed, regenerate Word from the V6 root:

```bash
pandoc docs/algorithm_guide/00_reading_guide.md \
  docs/algorithm_guide/01_preprocessing.md \
  docs/algorithm_guide/02_cache.md \
  docs/algorithm_guide/02_quality_artifacts.md \
  docs/algorithm_guide/03_features_representations.md \
  docs/algorithm_guide/99_source_index.md \
  --from markdown --standalone --toc --toc-depth=3 \
  --output docs/V6_ALGORITHM_GUIDE.docx
```

This guide does not change source code, defaults, sampling rates, model weights, or data. Static checks of line references and formulas do not replace algorithm validation on real data or imply that every optional module has undergone a new execution test.
