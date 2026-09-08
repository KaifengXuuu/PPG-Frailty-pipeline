"""Leakage-explicit Stage 0 decision-bias oracle analysis.

This module deliberately operates after model fitting.  It reads persisted
participant-level OOF probabilities, averages the five repeat probabilities for
each participant, and searches a three-class additive bias on a non-negative
unit simplex.  Labels used to optimise the bias are also used to score it, so
the result is an upper-bound diagnostic and is never an estimand of predictive
performance.
"""
from __future__ import annotations
import csv
import hashlib
import json
import math
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from html import escape as html_escape
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
import yaml
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from ..data.schema import CANONICAL_CLASS_NAMES
from .role_scope_decomposition import _html_document, _html_table, _markdown_document

SCHEMA_VERSION = 'ppg_frailty.stage0_decision_bias_oracle.v1'
LEAKAGE_STATUS = 'intentional_label_leakage_same_29_participants_fit_and_score'
SCIENTIFIC_ROLE = 'decision_layer_recoverable_margin_upper_bound_only'
_TOLERANCE = 1e-12
@dataclass(frozen=True)
class DecisionBiasOraclePlan:
    plan_path: Path
    study_id: str
    source_study_dir: Path
    case_id: str
    prediction_file: Path | None
    prediction_level: str
    prediction_kind: str | None
    expected_participants: int
    expected_repeats: tuple[int, ...]
    expected_class_order: tuple[int, ...]
    repeat_aggregation: str
    bias_step: float
    substantial_upper_bound: float
    limited_upper_bound: float
    output_slug: str
    write_static_figures: bool
    write_excel_workbook: bool
@dataclass(frozen=True)
class ParticipantOracleDataset:
    participant_ids: tuple[str, ...]
    labels: np.ndarray
    probabilities: np.ndarray
    class_order: tuple[int, ...]
    repeats: tuple[int, ...]
    source_rows: int
    prediction_file: Path
    prediction_file_sha256: str
    config_hash: str
    aggregation_rule: str
@dataclass(frozen=True)
class BiasOracleResult:
    biases: np.ndarray
    balanced_accuracies: np.ndarray
    baseline_balanced_accuracy: float
    baseline_macro_f1: float
    oracle_balanced_accuracy: float
    oracle_macro_f1: float
    best_bias: tuple[float, ...]
    baseline_predictions: np.ndarray
    oracle_predictions: np.ndarray
    tied_optimum_count: int

    @property
    def recoverable_margin(self) -> float:
        return self.oracle_balanced_accuracy - self.baseline_balanced_accuracy
def _path(value: Any, *, field: str, allow_none: bool = False) -> Path | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{field} must be a non-empty path string')
    return Path(value)
def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f'{field} must be a mapping')
    return value
def load_decision_bias_oracle_plan(path: str | Path, *,
                                   pipeline_root: str | Path | None = None) -> DecisionBiasOraclePlan:
    plan_path = Path(path).resolve()
    payload = yaml.safe_load(plan_path.read_text(encoding='utf-8'))
    root = _mapping(payload, 'plan')
    if root.get('schema_version') != SCHEMA_VERSION:
        raise ValueError(f"unsupported Stage 0 schema: {root.get('schema_version')!r}; expected {SCHEMA_VERSION!r}")
    study = _mapping(root.get('study'), 'study')
    source = _mapping(root.get('source'), 'source')
    oracle = _mapping(root.get('oracle'), 'oracle')
    interpretation = _mapping(root.get('interpretation'), 'interpretation')
    output = _mapping(root.get('output'), 'output')
    base = Path(pipeline_root).resolve() if pipeline_root is not None else plan_path.parent

    def resolve(value: Any, *, field: str, allow_none: bool = False) -> Path | None:
        candidate = _path(value, field=field, allow_none=allow_none)
        if candidate is None:
            return None
        return candidate if candidate.is_absolute() else (base / candidate).resolve()

    repeats = tuple((int(value) for value in source.get('expected_repeats', ())))
    class_order = tuple((int(value) for value in source.get('expected_class_order', ())))
    if not repeats or len(set(repeats)) != len(repeats):
        raise ValueError('source.expected_repeats must contain unique repeat indices')
    if len(class_order) != 3 or len(set(class_order)) != 3:
        raise ValueError('Stage 0 requires exactly three unique class labels')
    expected_participants = int(source.get('expected_participants', 0))
    if expected_participants < 3:
        raise ValueError('source.expected_participants must be at least three')
    repeat_aggregation = str(source.get('repeat_aggregation', ''))
    if repeat_aggregation != 'arithmetic_mean_probability_per_participant':
        raise ValueError("Stage 0 repeat_aggregation must be 'arithmetic_mean_probability_per_participant'")
    if str(oracle.get('bias_parameterization')) != 'nonnegative_sum_one_simplex':
        raise ValueError("Stage 0 bias_parameterization must be 'nonnegative_sum_one_simplex'")
    if str(oracle.get('objective')) != 'balanced_accuracy':
        raise ValueError('Stage 0 objective must be balanced_accuracy')
    if str(oracle.get('prediction_rule')) != 'argmax_probability_plus_bias':
        raise ValueError('unsupported Stage 0 prediction_rule')
    if str(oracle.get('prediction_tie_break')) != 'first_class_in_declared_class_order':
        raise ValueError('unsupported Stage 0 prediction_tie_break')
    if str(oracle.get('optimum_tie_break')) != 'closest_to_equal_bias_then_lexicographic':
        raise ValueError('unsupported Stage 0 optimum_tie_break')
    step = float(oracle.get('step', 0.0))
    reciprocal = round(1.0 / step) if step > 0.0 else 0
    if reciprocal <= 0 or not math.isclose(step * reciprocal, 1.0, abs_tol=1e-12):
        raise ValueError('oracle.step must divide the unit simplex exactly')
    substantial = float(interpretation.get('substantial_upper_bound', 0.7))
    limited = float(interpretation.get('limited_upper_bound', 0.65))
    if not 0.0 <= limited < substantial <= 1.0:
        raise ValueError('interpretation thresholds must satisfy 0 <= limited < substantial <= 1')
    study_id = str(study.get('study_id', '')).strip()
    case_id = str(source.get('case_id', '')).strip()
    prediction_level = str(source.get('prediction_level', 'participant')).strip()
    output_slug = str(output.get('slug', 'stage0-decision-bias-oracle-v1')).strip()
    if not all((study_id, case_id, prediction_level, output_slug)):
        raise ValueError('study_id, case_id, prediction_level, and output.slug are required')
    return DecisionBiasOraclePlan(
        plan_path=plan_path, study_id=study_id, source_study_dir=resolve(source.get('study_dir'),
                                                                         field='source.study_dir'), case_id=case_id,
        prediction_file=resolve(source.get('prediction_file'), field='source.prediction_file',
                                allow_none=True), prediction_level=prediction_level,
        prediction_kind=None if source.get('prediction_kind') is None else str(source.get('prediction_kind')),
        expected_participants=expected_participants, expected_repeats=repeats, expected_class_order=class_order,
        repeat_aggregation=repeat_aggregation, bias_step=step, substantial_upper_bound=substantial,
        limited_upper_bound=limited, output_slug=output_slug,
        write_static_figures=bool(output.get('write_static_figures',
                                             True)), write_excel_workbook=bool(output.get('write_excel_workbook',
                                                                                          True)))
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()
def _resolve_prediction_file(plan: DecisionBiasOraclePlan) -> Path:
    if plan.prediction_file is not None:
        candidate = plan.prediction_file
        if not candidate.is_file():
            raise FileNotFoundError(f'Stage 0 prediction file not found: {candidate}')
        return candidate
    case_root = plan.source_study_dir / 'raw' / plan.case_id / 'attempts'
    candidates = sorted(case_root.glob('attempt_*/experiment/oof_subject_predictions.parquet'))
    if not candidates:
        staging = tuple(case_root.glob('attempt_*/*.staging*/**/oof_subject_predictions.parquet'))
        suffix = f'; found {len(staging)} fold/staging files, which Stage 0 refuses to use' if staging else ''
        raise FileNotFoundError(
            f'no completed root participant OOF file for case {plan.case_id!r} under {plan.source_study_dir}{suffix}')
    if len(candidates) > 1:
        raise ValueError('multiple completed attempts exist; set source.prediction_file explicitly: ' +
                         ', '.join((str(path) for path in candidates)))
    return candidates[0]
def load_participant_oracle_dataset(plan: DecisionBiasOraclePlan) -> ParticipantOracleDataset:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError('Stage 0 requires pandas and pyarrow') from exc
    prediction_file = _resolve_prediction_file(plan)
    frame = pd.read_parquet(prediction_file)
    required = {
        'participant_id', 'label', 'probabilities', 'repeat', 'level', 'retained', 'class_order', 'config_hash',
        'aggregation_rule'
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f'participant OOF file lacks fields: {missing}')
    selected = frame.loc[frame['level'] == plan.prediction_level].copy()
    if plan.prediction_kind is not None:
        if 'prediction_kind' not in selected.columns:
            raise ValueError('prediction_kind filter requested but field is absent')
        selected = selected.loc[selected['prediction_kind'] == plan.prediction_kind]
    selected = selected.loc[selected['retained'].astype(bool)]
    if selected.empty:
        raise ValueError('no retained participant OOF rows match the Stage 0 source filters')
    observed_orders = {tuple((int(value) for value in np.asarray(raw).tolist())) for raw in selected['class_order']}
    if observed_orders != {plan.expected_class_order}:
        raise ValueError(
            f'class order mismatch: observed={sorted(observed_orders)}, expected={plan.expected_class_order}')
    observed_repeats = tuple(sorted((int(value) for value in selected['repeat'].unique())))
    if observed_repeats != tuple(sorted(plan.expected_repeats)):
        raise ValueError(
            f'incomplete repeat roster: observed={observed_repeats}, expected={tuple(sorted(plan.expected_repeats))}')
    participant_ids = tuple(sorted((str(value) for value in selected['participant_id'].unique())))
    if len(participant_ids) != plan.expected_participants:
        raise ValueError(
            f'participant count mismatch: observed={len(participant_ids)}, expected={plan.expected_participants}')
    expected_rows = plan.expected_participants * len(plan.expected_repeats)
    if len(selected) != expected_rows:
        raise ValueError(f'participant-repeat row count mismatch: observed={len(selected)}, expected={expected_rows}')
    duplicate = selected.duplicated(subset=['participant_id', 'repeat'], keep=False)
    if bool(duplicate.any()):
        raise ValueError('participant OOF file has duplicate participant-repeat rows')
    config_hashes = tuple(sorted((str(value) for value in selected['config_hash'].unique())))
    aggregation_rules = tuple(sorted((str(value) for value in selected['aggregation_rule'].unique())))
    if len(config_hashes) != 1 or len(aggregation_rules) != 1:
        raise ValueError('Stage 0 source mixes config hashes or aggregation rules')
    labels: list[int] = []
    probabilities: list[np.ndarray] = []
    for participant_id in participant_ids:
        rows = selected.loc[selected['participant_id'].astype(str) == participant_id]
        repeats = tuple(sorted((int(value) for value in rows['repeat'])))
        if repeats != tuple(sorted(plan.expected_repeats)):
            raise ValueError(f'participant {participant_id!r} lacks a complete repeat roster')
        participant_labels = tuple((int(value) for value in rows['label'].unique()))
        if len(participant_labels) != 1 or participant_labels[0] not in plan.expected_class_order:
            raise ValueError(f'participant {participant_id!r} has inconsistent labels')
        matrix = np.asarray([np.asarray(value, dtype=np.float64) for value in rows['probabilities']], dtype=np.float64)
        if matrix.shape != (len(plan.expected_repeats), len(plan.expected_class_order)):
            raise ValueError(f'participant {participant_id!r} has invalid probability shape')
        if not np.isfinite(matrix).all() or (matrix < -_TOLERANCE).any():
            raise ValueError(f'participant {participant_id!r} has invalid probabilities')
        if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-06):
            raise ValueError(f'participant {participant_id!r} probabilities do not sum to one')
        mean_probability = matrix.mean(axis=0)
        mean_probability /= mean_probability.sum()
        labels.append(participant_labels[0])
        probabilities.append(mean_probability)
    observed_labels = set(labels)
    if observed_labels != set(plan.expected_class_order):
        raise ValueError(
            f'all three classes are required: observed={sorted(observed_labels)}, expected={plan.expected_class_order}')
    return ParticipantOracleDataset(participant_ids=participant_ids, labels=np.asarray(labels, dtype=np.int64),
                                    probabilities=np.asarray(probabilities,
                                                             dtype=np.float64), class_order=plan.expected_class_order,
                                    repeats=tuple(sorted(plan.expected_repeats)), source_rows=len(selected),
                                    prediction_file=prediction_file, prediction_file_sha256=_sha256(prediction_file),
                                    config_hash=config_hashes[0], aggregation_rule=aggregation_rules[0])
# Enumeration order is part of the deterministic optimum tie-break contract.
def enumerate_simplex_biases(step: float) -> np.ndarray:
    units = round(1.0 / float(step))
    if units <= 0 or not math.isclose(units * float(step), 1.0, abs_tol=1e-12):
        raise ValueError('step must divide one exactly')
    values = [(first / units, second / units, (units - first - second) / units) for first in range(units + 1)
              for second in range(units - first + 1)]
    return np.asarray(values, dtype=np.float64)
# Labels intentionally tune and score this leaked upper bound.
def search_decision_bias_oracle(labels: np.ndarray, probabilities: np.ndarray, *, class_order: Sequence[int],
                                step: float) -> BiasOracleResult:
    y = np.asarray(labels, dtype=np.int64)
    probability = np.asarray(probabilities, dtype=np.float64)
    order = np.asarray(tuple((int(value) for value in class_order)), dtype=np.int64)
    if probability.ndim != 2 or probability.shape != (y.size, order.size):
        raise ValueError('probabilities must have shape [participant, class]')
    if order.size != 3 or set(y.tolist()) != set(order.tolist()):
        raise ValueError('Stage 0 requires three represented classes')
    biases = enumerate_simplex_biases(step)
    prediction_indices = np.argmax(probability[:, np.newaxis, :] + biases[np.newaxis, :, :], axis=2)
    predictions = order[prediction_indices]
    recalls = np.empty((biases.shape[0], order.size), dtype=np.float64)
    for index, class_label in enumerate(order):
        mask = y == class_label
        recalls[:, index] = np.mean(predictions[mask] == class_label, axis=0)
    balanced_accuracies = recalls.mean(axis=1)
    maximum = float(np.max(balanced_accuracies))
    tied = np.flatnonzero(np.isclose(balanced_accuracies, maximum, rtol=0.0, atol=_TOLERANCE))
    equal = np.full(order.size, 1.0 / order.size, dtype=np.float64)
    best_index = min((int(value) for value in tied), key=lambda index: (float(np.sum(
        (biases[index] - equal)**2)), tuple((float(value) for value in biases[index]))))
    baseline_predictions = order[np.argmax(probability, axis=1)]
    oracle_predictions = predictions[:, best_index]
    return BiasOracleResult(biases=biases, balanced_accuracies=balanced_accuracies,
                            baseline_balanced_accuracy=float(balanced_accuracy_score(y, baseline_predictions)),
                            baseline_macro_f1=float(f1_score(y, baseline_predictions,
                                                             average='macro')), oracle_balanced_accuracy=maximum,
                            oracle_macro_f1=float(f1_score(y, oracle_predictions, average='macro')), best_bias=tuple(
                                (float(value)
                                 for value in biases[best_index])), baseline_predictions=baseline_predictions,
                            oracle_predictions=oracle_predictions, tied_optimum_count=int(tied.size))
def _interpret(plan: DecisionBiasOraclePlan, oracle_ba: float) -> tuple[str, str]:
    if oracle_ba >= plan.substantial_upper_bound:
        return (
            'substantial_decision_layer_headroom',
            'The leaked upper bound reaches the predeclared 0.70 threshold; decision-layer work is worth a leakage-free follow-up.'
        )
    if oracle_ba <= plan.limited_upper_bound:
        return (
            'limited_decision_layer_headroom',
            'The leaked upper bound is at or below the predeclared 0.65 ceiling; prioritise beat-level representation work.'
        )
    return (
        'intermediate_decision_layer_headroom',
        'The leaked upper bound lies between 0.65 and 0.70; the Stage 0 decision is inconclusive and any follow-up must use training-only bias fitting.'
    )
def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = list(dict.fromkeys((str(key) for row in rows for key in row)))
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
def _metric_rows(dataset: ParticipantOracleDataset, result: BiasOracleResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rule, prediction in (('unmodified_argmax', result.baseline_predictions), ('leaked_oracle_bias_argmax',
                                                                                  result.oracle_predictions)):
        precision, recall, f1, support = precision_recall_fscore_support(dataset.labels, prediction,
                                                                         labels=dataset.class_order, zero_division=0)
        for index, class_label in enumerate(dataset.class_order):
            rows.append({
                'decision_rule': rule, 'class_label': class_label,
                'class_name': CANONICAL_CLASS_NAMES.get(class_label, str(class_label)), 'support': int(support[index]),
                'precision': float(precision[index]), 'sensitivity_recall': float(recall[index]),
                'f1': float(f1[index]), 'scientific_role': SCIENTIFIC_ROLE
            })
    return rows
def _confusion_rows(dataset: ParticipantOracleDataset, result: BiasOracleResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rule, prediction in (('unmodified_argmax', result.baseline_predictions), ('leaked_oracle_bias_argmax',
                                                                                  result.oracle_predictions)):
        matrix = confusion_matrix(dataset.labels, prediction, labels=dataset.class_order)
        for true_index, true_label in enumerate(dataset.class_order):
            for predicted_index, predicted_label in enumerate(dataset.class_order):
                rows.append({
                    'decision_rule': rule, 'true_class': CANONICAL_CLASS_NAMES.get(true_label, str(true_label)),
                    'predicted_class': CANONICAL_CLASS_NAMES.get(predicted_label, str(predicted_label)),
                    'count': int(matrix[true_index, predicted_index])
                })
    return rows
# Plots consume the exact grid and confusion-table values.
def _plot_simplex(path: Path, result: BiasOracleResult, class_order: Sequence[int]) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    bias = result.biases
    x = bias[:, 1] + 0.5 * bias[:, 2]
    y = math.sqrt(3.0) / 2.0 * bias[:, 2]
    best = np.asarray(result.best_bias)
    best_x = best[1] + 0.5 * best[2]
    best_y = math.sqrt(3.0) / 2.0 * best[2]
    figure, axis = plt.subplots(figsize=(8.2, 7.0))
    scatter = axis.scatter(x, y, c=result.balanced_accuracies, cmap='viridis', s=18, linewidths=0)
    axis.scatter([best_x], [best_y], marker='*', s=220, c='red', label='oracle optimum')
    equal_x = 0.5
    equal_y = math.sqrt(3.0) / 6.0
    axis.scatter([equal_x], [equal_y], marker='x', s=80, c='white', linewidths=2, label='equal-bias location')
    vertices = ((0.0, 0.0), (1.0, 0.0), (0.5, math.sqrt(3.0) / 2.0))
    for (vx, vy), class_label in zip(vertices, class_order, strict=True):
        axis.text(vx, vy - 0.045 if vy == 0.0 else vy + 0.035,
                  f'b[{CANONICAL_CLASS_NAMES.get(class_label, class_label)}]=1', ha='center', va='center', fontsize=9)
    axis.plot([0, 1, 0.5, 0], [0, 0, math.sqrt(3) / 2, 0], color='black', lw=0.8)
    axis.set_title('Leaked Stage 0 bias-simplex balanced accuracy')
    axis.set_aspect('equal')
    axis.axis('off')
    axis.legend(loc='upper right', fontsize='small')
    figure.colorbar(scatter, ax=axis, label='balanced accuracy on the same 29 labels')
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(figure)
def _plot_confusions(path: Path, dataset: ParticipantOracleDataset, result: BiasOracleResult) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = [CANONICAL_CLASS_NAMES.get(value, str(value)) for value in dataset.class_order]
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    for axis, title, prediction in zip(axes, ('Unmodified argmax', 'Leaked oracle bias'),
                                       (result.baseline_predictions, result.oracle_predictions), strict=True):
        matrix = confusion_matrix(dataset.labels, prediction, labels=dataset.class_order)
        image = axis.imshow(matrix, cmap='Blues', vmin=0)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(column, row, str(matrix[row, column]), ha='center', va='center')
        axis.set_xticks(range(len(names)), names, rotation=30, ha='right')
        axis.set_yticks(range(len(names)), names)
        axis.set_xlabel('Predicted')
        axis.set_ylabel('True')
        axis.set_title(title)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle('Participant-level decisions on the same 29-person oracle roster')
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(figure)
def _markdown_table(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> str:
    def render(value: Any) -> str:
        if isinstance(value, float):
            return f'{value:.6f}'
        return str(value)

    header = '| ' + ' | '.join(fields) + ' |'
    divider = '| ' + ' | '.join(('---' for _ in fields)) + ' |'
    body = ['| ' + ' | '.join((render(row.get(field, '')) for field in fields)) + ' |' for row in rows]
    return '\n'.join([header, divider, *body])
# Exhaustive tables remain authoritative; pages are concise projections.
def _write_reports(output_dir: Path, plan: DecisionBiasOraclePlan, dataset: ParticipantOracleDataset,
                   result: BiasOracleResult, summary: Mapping[str, Any]) -> None:
    summary_rows = [summary]
    per_class_rows = _metric_rows(dataset, result)
    confusion_rows = _confusion_rows(dataset, result)
    participant_rows = []
    for index, participant_id in enumerate(dataset.participant_ids):
        row: dict[str, Any] = {
            'participant_id': participant_id, 'true_label': int(dataset.labels[index]),
            'true_class': CANONICAL_CLASS_NAMES.get(int(dataset.labels[index]), str(dataset.labels[index]))
        }
        for class_index, class_label in enumerate(dataset.class_order):
            row[f'mean_oof_probability_class_{class_label}'] = float(dataset.probabilities[index, class_index])
        row['baseline_prediction'] = int(result.baseline_predictions[index])
        row['oracle_prediction'] = int(result.oracle_predictions[index])
        row['decision_changed'] = bool(result.baseline_predictions[index] != result.oracle_predictions[index])
        participant_rows.append(row)
    grid_rows = [{
        'bias_class_0': float(bias[0]), 'bias_class_1': float(bias[1]), 'bias_class_2': float(bias[2]),
        'balanced_accuracy_same_29_labels': float(score),
        'is_selected_optimum': bool(np.allclose(bias, np.asarray(result.best_bias), atol=_TOLERANCE))
    } for bias, score in zip(result.biases, result.balanced_accuracies, strict=True)]
    component_rows = [{
        'stage': plan.study_id, 'component_type': 'source_model_probabilities', 'component_id': plan.case_id,
        'input_data': f'{plan.expected_participants} participants x {len(plan.expected_repeats)} OOF repeats; participant-level three-class probabilities',
        'fixed_parameters': f'level={plan.prediction_level}; prediction_kind={plan.prediction_kind}; repeat_aggregation={plan.repeat_aggregation}; aggregation_rule={dataset.aggregation_rule}'
    }, {
        'stage': plan.study_id, 'component_type': 'decision_bias_oracle',
        'component_id': 'three_class_additive_probability_bias_simplex_v1',
        'input_data': 'same 29 true labels and repeat-mean OOF probabilities',
        'fixed_parameters': f'b_c>=0; sum(b)=1; step={plan.bias_step:g}; objective=balanced_accuracy; rule=argmax(p_c+b_c); prediction_tie=first_class_order; optimum_tie=closest_to_equal_then_lexicographic'
    }]
    table_figure_rows = [{
        'table': 'stage0_summary.csv; bias_grid.csv', 'figure': 'bias_simplex_balanced_accuracy.png',
        'status': 'available' if plan.write_static_figures else 'N/A_disabled_by_plan',
        'relationship': 'The plotted colour is the BA column for every enumerated bias vector.'
    }, {
        'table': 'confusion_matrices.csv', 'figure': 'baseline_oracle_confusion_matrices.png',
        'status': 'available' if plan.write_static_figures else 'N/A_disabled_by_plan',
        'relationship': 'The two plotted matrices are direct renderings of the long-form counts.'
    }]
    tables = output_dir / 'tables'
    figures = output_dir / 'figures'
    tables.mkdir(parents=True)
    figures.mkdir(parents=True)
    for name, rows in (('stage0_summary', summary_rows), ('per_class_metrics', per_class_rows),
                       ('confusion_matrices', confusion_rows), ('participant_probabilities', participant_rows),
                       ('bias_grid', grid_rows), ('test_components', component_rows), ('table_figure_pairs',
                                                                                       table_figure_rows)):
        _write_csv(tables / f'{name}.csv', rows)
    if plan.write_static_figures:
        _plot_simplex(figures / 'bias_simplex_balanced_accuracy.png', result, dataset.class_order)
        _plot_confusions(figures / 'baseline_oracle_confusion_matrices.png', dataset, result)
    from ..reporting.tabular import write_excel_workbook_from_csv_directory, write_table_column_definitions
    write_table_column_definitions(tables, csv_directory=tables)
    if plan.write_excel_workbook:
        write_excel_workbook_from_csv_directory(tables / 'report_tables.xlsx', tables)
    component_markdown = '# Stage 0 test components\n\n' + _markdown_table(
        component_rows, ('component_type', 'component_id', 'input_data', 'fixed_parameters'))
    (output_dir / 'TEST_COMPONENTS.md').write_text(component_markdown + '\n', encoding='utf-8')
    result_payload = {
        'schema_version': SCHEMA_VERSION, 'study_id': plan.study_id, 'summary': dict(summary),
        'best_bias': list(result.best_bias), 'class_order': list(dataset.class_order), 'source': {
            'study_dir': str(plan.source_study_dir), 'case_id': plan.case_id,
            'prediction_file': str(dataset.prediction_file), 'prediction_file_sha256': dataset.prediction_file_sha256,
            'config_hash': dataset.config_hash, 'aggregation_rule': dataset.aggregation_rule,
            'source_participant_repeat_rows': dataset.source_rows,
            'participant_count_after_repeat_mean': len(dataset.participant_ids), 'repeats': list(dataset.repeats)
        }, 'oracle_contract': {
            'bias_parameterization': 'nonnegative_sum_one_simplex', 'step': plan.bias_step,
            'objective': 'balanced_accuracy', 'prediction_rule': 'argmax_c(p_c + b_c)',
            'prediction_tie_break': 'first_class_in_declared_class_order',
            'optimum_tie_break': 'closest_to_equal_bias_then_lexicographic', 'grid_points': int(result.biases.shape[0]),
            'tied_optimum_count': result.tied_optimum_count
        }, 'leakage_guard': {
            'status': LEAKAGE_STATUS, 'scientific_role': SCIENTIFIC_ROLE, 'eligible_as_predictive_performance': False,
            'eligible_for_model_selection': False, 'eligible_for_deployment': False,
            'eligible_for_threshold_or_bias_export': False,
            'confidence_interval': 'N/A_same_labels_used_to_fit_and_score_oracle',
            'p_value': 'N/A_same_labels_used_to_fit_and_score_oracle'
        }
    }
    (output_dir / 'stage0_result.json').write_text(
        json.dumps(result_payload, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    fields = ('model', 'baseline_balanced_accuracy', 'oracle_upper_bound_balanced_accuracy',
              'recoverable_margin_percentage_points', 'decisions_changed', 'best_bias', 'interpretation_status')
    per_class_fields = ('decision_rule', 'class_name', 'support', 'precision', 'sensitivity_recall', 'f1')
    figures_spec = (('Bias-simplex BA', 'bias_simplex_balanced_accuracy.png'),
                    ('Baseline and oracle confusion matrices', 'baseline_oracle_confusion_matrices.png'))
    figure_md = '\n\n'.join(
        (f'![{label}](figures/{name})' for label, name in figures_spec)) if plan.write_static_figures else ''
    figure_html = ''.join((f'<img src="figures/{name}" alt="{label}">'
                           for label, name in figures_spec)) if plan.write_static_figures else ''
    warning = '**LEAKED UPPER BOUND — NOT PREDICTIVE PERFORMANCE.** The same labels choose and score the bias; the result is ineligible for model selection, deployment, CI, or P value.'
    method = f'Average {len(dataset.repeats)} OOF probabilities per participant, enumerate the non-negative three-class unit simplex at step {plan.bias_step:g}, and apply argmax(p_c+b_c). Ties follow declared class order; tied optima use closest-to-equal then lexicographic order.'
    limits = 'Bias changes decision boundaries, not within-class score ordering. Any follow-up must fit bias on training participants and evaluate held-out people. Resampling the label-leaking optimisation would not estimate generalisation.'
    formulae = f'- Decision: y_hat_i(b) = argmax_c(p_ic + b_c).\n- Simplex: b_c >= 0, sum_c b_c = 1, step {plan.bias_step:g}.\n- BA: mean class recall; recoverable margin: 100 × (BA_oracle − BA_unmodified) pp.\n- Macro-F1: arithmetic mean of three one-vs-rest F1 values.'
    provenance = f'- Study/case: {plan.source_study_dir} / {plan.case_id}\n- OOF: {dataset.prediction_file}\n- SHA-256: {dataset.prediction_file_sha256}\n- Config: {dataset.config_hash}; aggregation: {dataset.aggregation_rule}\n- Grid points: {result.biases.shape[0]}; leakage: {LEAKAGE_STATUS}'
    sections = [('Status', warning), ('Result', _markdown_table(summary_rows, fields)),
                ('Interpretation', str(summary['interpretation'])), ('Method', method),
                ('Per-class oracle-roster results', _markdown_table(per_class_rows, per_class_fields)),
                ('Limits', limits), ('Formulae', formulae), ('Provenance', provenance), ('Figures', figure_md)]
    (output_dir / 'STUDY_SUMMARY.md').write_text(
        _markdown_document('Stage 0: decision-layer recoverable-margin oracle', sections), encoding='utf-8')
    html_sections = [('Result', _html_table(summary_rows, fields)),
                     ('Interpretation', f"<p>{html_escape(str(summary['interpretation']))}</p>"),
                     ('Per-class oracle-roster results', _html_table(per_class_rows, per_class_fields)),
                     ('Method and limits', f'<p>{html_escape(method)} {html_escape(limits)}</p>'),
                     ('Figures', figure_html), ('Provenance', f'<pre>{html_escape(provenance)}</pre>')]
    html_warning = '<strong>LEAKED UPPER BOUND — NOT PREDICTIVE PERFORMANCE.</strong> No CI or P value applies.'
    (output_dir / 'STUDY_SUMMARY.html').write_text(
        _html_document('Stage 0: decision-layer recoverable-margin oracle', html_sections, warning=html_warning),
        encoding='utf-8')
# Public entry resolves inputs, runs the oracle, and publishes immutable output.
def run_decision_bias_oracle(plan_path: str | Path, *, pipeline_root: str | Path, output_root: str | Path,
                             source_study_dir: str | Path | None = None, case_id: str | None = None,
                             prediction_file: str | Path | None = None, step: float | None = None) -> Path:
    plan = load_decision_bias_oracle_plan(plan_path, pipeline_root=pipeline_root)
    overrides = asdict(plan)
    overrides['plan_path'] = plan.plan_path
    if source_study_dir is not None:
        candidate = Path(source_study_dir)
        overrides['source_study_dir'] = candidate if candidate.is_absolute() else (Path(pipeline_root) /
                                                                                   candidate).resolve()
    if case_id is not None:
        overrides['case_id'] = str(case_id)
    if prediction_file is not None:
        candidate = Path(prediction_file)
        overrides['prediction_file'] = candidate if candidate.is_absolute() else (Path(pipeline_root) /
                                                                                  candidate).resolve()
    if step is not None:
        enumerate_simplex_biases(step)
        overrides['bias_step'] = float(step)
    plan = DecisionBiasOraclePlan(**overrides)
    dataset = load_participant_oracle_dataset(plan)
    result = search_decision_bias_oracle(dataset.labels, dataset.probabilities, class_order=dataset.class_order,
                                         step=plan.bias_step)
    status, interpretation = _interpret(plan, result.oracle_balanced_accuracy)
    now = datetime.now().astimezone()
    output_dir = Path(output_root).resolve() / f'{now:%Y%m%d_%H%M%S}_{plan.output_slug}'
    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(plan.plan_path, output_dir / 'study_plan.yaml')
    summary = {
        'model': plan.case_id, 'participants': len(dataset.participant_ids),
        'oof_repeats_per_participant': len(dataset.repeats),
        'baseline_balanced_accuracy': result.baseline_balanced_accuracy, 'baseline_macro_f1': result.baseline_macro_f1,
        'oracle_upper_bound_balanced_accuracy': result.oracle_balanced_accuracy,
        'oracle_macro_f1_same_labels': result.oracle_macro_f1,
        'recoverable_margin_percentage_points': 100.0 * result.recoverable_margin, 'decisions_changed': int(
            np.sum(result.baseline_predictions != result.oracle_predictions)), 'best_bias': '[' + ', '.join(
                (f'{value:.2f}' for value in result.best_bias)) + ']', 'grid_step': plan.bias_step,
        'grid_points': int(result.biases.shape[0]), 'tied_optimum_count': result.tied_optimum_count,
        'interpretation_status': status, 'interpretation': interpretation, 'leakage_status': LEAKAGE_STATUS,
        'eligible_as_predictive_performance': False, 'ci95': 'N/A_same_labels_used_to_fit_and_score_oracle',
        'p_value': 'N/A_same_labels_used_to_fit_and_score_oracle'
    }
    _write_reports(output_dir, plan, dataset, result, summary)
    return output_dir
__all__ = [
    'BiasOracleResult', 'DecisionBiasOraclePlan', 'LEAKAGE_STATUS', 'ParticipantOracleDataset', 'SCHEMA_VERSION',
    'SCIENTIFIC_ROLE', 'enumerate_simplex_biases', 'load_decision_bias_oracle_plan', 'load_participant_oracle_dataset',
    'run_decision_bias_oracle', 'search_decision_bias_oracle'
]
