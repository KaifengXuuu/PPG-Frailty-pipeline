"""One-pass analysis of persisted prediction and execution artifacts."""
from __future__ import annotations
import math
from dataclasses import asdict, dataclass, fields, replace
from statistics import fmean, pstdev, stdev
from typing import Any, Mapping, Sequence
import numpy as np
from scipy.stats import t as student_t
from sklearn.metrics import average_precision_score, roc_auc_score
from ..data.schema import CANONICAL_CLASS_NAMES
from ..training.aggregation import LINE_A_EQUAL_FILES, LINE_B_EQUAL_ROLE_FAMILIES, aggregate_hierarchy
from ..training.evaluator import evaluate_predictions_with_abstentions
from ..training.oof import OofPredictionRow
from .classification_diagnostics import classification_diagnostic_status_rows, classification_per_class_metric_rows, classification_roc_curve_rows, classification_tsne_rows, normalize_classification_rows
from .collect import CollectedStudy
from .conclusions import _number, paired_inference_against_reference
@dataclass(frozen=True)
class StudyAnalysis:
    case_summary: tuple[Mapping[str, Any], ...] = ()
    metric_distribution_summary: tuple[Mapping[str, Any], ...] = ()
    repeat_metrics: tuple[Mapping[str, Any], ...] = ()
    fold_metrics: tuple[Mapping[str, Any], ...] = ()
    per_class_metrics: tuple[Mapping[str, Any], ...] = ()
    confusion_matrices: tuple[Mapping[str, Any], ...] = ()
    confusion_counts: tuple[Mapping[str, Any], ...] = ()
    confusion_row_normalized: tuple[Mapping[str, Any], ...] = ()
    calibration_bins: tuple[Mapping[str, Any], ...] = ()
    paired_deltas: tuple[Mapping[str, Any], ...] = ()
    paired_participant_inference: tuple[Mapping[str, Any], ...] = ()
    aggregation_line_comparison: tuple[Mapping[str, Any], ...] = ()
    aggregation_line_repeat_metrics: tuple[Mapping[str, Any], ...] = ()
    aggregation_line_per_class_metrics: tuple[Mapping[str, Any], ...] = ()
    aggregation_view_comparison: tuple[Mapping[str, Any], ...] = ()
    aggregation_view_repeat_metrics: tuple[Mapping[str, Any], ...] = ()
    aggregation_view_per_class_metrics: tuple[Mapping[str, Any], ...] = ()
    aggregation_view_confusion_matrices: tuple[Mapping[str, Any], ...] = ()
    aggregation_hierarchy_coverage: tuple[Mapping[str, Any], ...] = ()
    legacy_bridge_numeric_ablation_report: tuple[Mapping[str, Any], ...] = ()
    legacy_bridge_execution_order_report: tuple[Mapping[str, Any], ...] = ()
    coverage: tuple[Mapping[str, Any], ...] = ()
    route_role_coverage: tuple[Mapping[str, Any], ...] = ()
    quality_distributions: tuple[Mapping[str, Any], ...] = ()
    predictive_leaderboard: tuple[Mapping[str, Any], ...] = ()
    worst_class_f1_stability: tuple[Mapping[str, Any], ...] = ()
    incomplete_cases: tuple[Mapping[str, Any], ...] = ()
    deployment_table: tuple[Mapping[str, Any], ...] = ()
    notes: tuple[str, ...] = ()
    denoiser_hr_record_pairs: tuple[Mapping[str, Any], ...] = ()
    denoiser_hr_comparison: tuple[Mapping[str, Any], ...] = ()
    repeat_per_class_metrics: tuple[Mapping[str, Any], ...] = ()
    per_class_metric_distribution_summary: tuple[Mapping[str, Any], ...] = ()
    aggregation_view_fold_metrics: tuple[Mapping[str, Any], ...] = ()
    stage3_star_absolute: tuple[Mapping[str, Any], ...] = ()
    stage3_star_contrasts: tuple[Mapping[str, Any], ...] = ()
    stage3_star_fold_contrasts: tuple[Mapping[str, Any], ...] = ()
    stage3_star_execution: tuple[Mapping[str, Any], ...] = ()
    stage3_star_inception_comparison: tuple[Mapping[str, Any], ...] = ()
    stage3_star_cnn_comparison: tuple[Mapping[str, Any], ...] = ()
    stage3_star_model_comparison: tuple[Mapping[str, Any], ...] = ()
    classification_prediction_scores: tuple[Mapping[str, Any], ...] = ()
    classification_roc_curves: tuple[Mapping[str, Any], ...] = ()
    classification_prediction_tsne: tuple[Mapping[str, Any], ...] = ()
    classification_diagnostic_status: tuple[Mapping[str, Any], ...] = ()
    classifier_per_class_results: tuple[Mapping[str, Any], ...] = ()
_SUMMARY_METRICS = ('balanced_accuracy', 'macro_f1', 'macro_roc_auc_ovr', 'macro_pr_auc_ovr', 'multiclass_log_loss', 'multiclass_brier',
                    'expected_calibration_error', 'worst_class_recall', 'worst_class_f1', 'coverage_rate',
                    'abstention_aware_balanced_accuracy', 'abstention_aware_macro_f1', 'abstention_aware_macro_recall',
                    'abstention_aware_macro_precision')
def _class_order(rows: Sequence[Mapping[str, Any]]) -> tuple[int, ...]:
    for row in rows:
        raw = row.get('class_order')
        if raw and (not isinstance(raw, (str, bytes))):
            return tuple((int(value) for value in raw))
        probabilities = row.get('probabilities')
        if probabilities and (not isinstance(probabilities, (str, bytes, Mapping))):
            return tuple(range(len(probabilities)))
    return tuple(sorted(CANONICAL_CLASS_NAMES))
def _roc_pr(labels: np.ndarray, probability: np.ndarray, order: tuple[int, ...]) -> tuple[float | None, float | None]:
    roc, pr = ([], [])
    for column, label in enumerate(order):
        target = labels == label
        if np.unique(target).size == 2:
            roc.append(float(roc_auc_score(target, probability[:, column])))
            pr.append(float(average_precision_score(target, probability[:, column])))
    return (fmean(roc) if roc else None, fmean(pr) if pr else None)
def _metric_row(case: str, rows: Sequence[Mapping[str, Any]], *, repeat: int | None = None, fold: int | None = None,
                ece_bins: int = 10) -> tuple[dict[str, Any], list[dict[str, Any]], list[list[int]] | None]:
    order = _class_order(rows)
    retained = [row for row in rows if bool(row.get('retained', True)) and row.get('probabilities')]
    dropped = [row for row in rows if not bool(row.get('retained', True))]
    base = {
        'case_id': case, 'repeat': repeat, 'fold': fold, 'n_total': len(rows), 'n_retained': len(retained), 'n_dropped': len(dropped),
        'coverage_rate': len(retained) / len(rows) if rows else None
    }
    if not rows:
        return ({**base, 'status': 'N/A_no_predictions'}, [], None)
    labels = np.asarray([int(row['label']) for row in retained], dtype=np.int64)
    probabilities = np.asarray([row['probabilities'] for row in retained], dtype=np.float64)
    dropped_labels = np.asarray([int(row['label']) for row in dropped], dtype=np.int64)
    aware = evaluate_predictions_with_abstentions(labels, probabilities, dropped_labels, class_order=order, ece_bins=ece_bins)
    conditional = aware.conditional_metrics
    result = dict(base)
    per_class: list[dict[str, Any]] = []
    matrix = None
    if conditional is not None:
        payload = asdict(conditional)
        matrix = [list(row) for row in conditional.confusion_matrix]
        for key, value in payload.items():
            if key not in {'per_class', 'confusion_matrix', 'class_order'}:
                result[key] = value
        roc, pr = _roc_pr(labels, probabilities, order)
        result.update(macro_roc_auc_ovr=roc, macro_pr_auc_ovr=pr, n_predictions=len(retained))
        aware_by_label = {item.label: item for item in aware.per_class}
        for item in conditional.per_class:
            extended = aware_by_label[item.label]
            per_class.append({
                'case_id': case, 'repeat': repeat, 'fold': fold, 'class_label': item.label,
                'class_name': CANONICAL_CLASS_NAMES.get(item.label,
                                                        str(item.label)), 'precision': item.precision, 'recall': item.recall, 'f1': item.f1,
                'support': item.support, 'abstention_aware_precision': extended.precision, 'abstention_aware_recall': extended.recall,
                'abstention_aware_f1': extended.f1, 'abstention_count': extended.abstention_count
            })
    result.update({
        'abstention_aware_balanced_accuracy': aware.balanced_accuracy, 'abstention_aware_macro_precision': aware.macro_precision,
        'abstention_aware_macro_recall': aware.macro_recall, 'abstention_aware_macro_f1': aware.macro_f1,
        'abstention_count': aware.n_abstained, 'abstention_counts_by_class': dict(aware.abstention_counts_by_class),
        'abstention_probability_metrics_scope': aware.probability_metrics_scope,
        'metric_source': 'persisted_participant_predictions_recomputed',
        'status': 'available' if retained else 'N/A_no_retained_predictions'
    })
    return (result, per_class, matrix)
def _distribution(rows: Sequence[Mapping[str, Any]], *, group_field: str = 'case_id') -> list[dict[str, Any]]:
    output = []
    groups = sorted({str(row.get(group_field, '')) for row in rows})
    for group in groups:
        selected = [row for row in rows if str(row.get(group_field, '')) == group]
        for metric in _SUMMARY_METRICS:
            values = [value for row in selected if (value := _number(row.get(metric))) is not None]
            if not values:
                continue
            count, mean = (len(values), fmean(values))
            spread = stdev(values) if count > 1 else 0.0
            half = float(student_t.ppf(0.975, count - 1) * spread / math.sqrt(count)) if count > 1 else 0.0
            output.append({
                group_field: group, 'metric': metric, 'n': count, 'mean': mean, 'population_sd': pstdev(values), 'sample_sd': spread,
                'minimum': min(values), 'maximum': max(values), 'repeat_t_ci95_low': mean - half, 'repeat_t_ci95_high': mean + half
            })
    return output
def _summary(case: str, repeats: Sequence[Mapping[str, Any]], folds: Sequence[Mapping[str, Any]], status: str, expected_repeats: int | None,
             expected_cells: int | None) -> dict[str, Any]:
    row: dict[str, Any] = {
        'case_id': case, 'status': status, 'repeat_count': len(repeats), 'fold_cell_count': len(folds),
        'complete_for_requested_execution': status in {'passed', 'success', 'complete', 'completed'}
        and (expected_repeats is None or len(repeats) == expected_repeats) and (expected_cells is None or len(folds) == expected_cells)
    }
    for metric in _SUMMARY_METRICS:
        values = [value for item in repeats if (value := _number(item.get(metric))) is not None]
        if values:
            row[f'participant_mean_{metric}'] = fmean(values)
            row[f'repeat_{metric}_population_sd'] = pstdev(values)
            row[f'repeat_{metric}_sample_sd'] = stdev(values) if len(values) > 1 else 0.0
    pooled, _, _ = _metric_row(case, [item for item in ()])
    return row
def _calibration(case: str, rows: Sequence[Mapping[str, Any]], bins: int) -> list[dict[str, Any]]:
    retained = [row for row in rows if bool(row.get('retained', True)) and row.get('probabilities')]
    if not retained:
        return []
    order = _class_order(retained)
    probabilities = np.asarray([row['probabilities'] for row in retained], dtype=float)
    labels = np.asarray([int(row['label']) for row in retained])
    predicted = np.asarray(order)[probabilities.argmax(1)]
    confidence = probabilities.max(1)
    correct = labels == predicted
    indices = np.minimum((confidence * bins).astype(int), bins - 1)
    return [{
        'case_id': case, 'bin_index': index, 'bin_lower': index / bins, 'bin_upper': (index + 1) / bins,
        'count': int(np.sum(indices == index)),
        'mean_confidence': float(np.mean(confidence[indices == index])) if np.any(indices == index) else None,
        'accuracy': float(np.mean(correct[indices == index])) if np.any(indices == index) else None
    } for index in range(bins)]
def _confusion_long(matrices: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    counts, normalized = ([], [])
    for source in matrices:
        value = np.asarray(source['confusion_matrix'], dtype=float)
        totals = value.sum(axis=1)
        for true in range(value.shape[0]):
            for predicted in range(value.shape[1]):
                base = {
                    'case_id': source.get('case_id'), 'true_class': true, 'predicted_class': predicted,
                    'metric_source': source.get('metric_source')
                }
                counts.append({**base, 'count': int(value[true, predicted])})
                normalized.append({**base, 'value': float(value[true, predicted] / totals[true]) if totals[true] else None})
    return (counts, normalized)
def _paired_deltas(repeats: Sequence[Mapping[str, Any]], reference: str | None) -> list[dict[str, Any]]:
    if not reference:
        return []
    lookup = {(str(row['case_id']), int(row['repeat'])): row for row in repeats if row.get('repeat') is not None}
    output = []
    for (case, repeat), row in sorted(lookup.items()):
        baseline = lookup.get((reference, repeat))
        if case == reference or baseline is None:
            continue
        result = {'reference_case_id': reference, 'case_id': case, 'repeat': repeat}
        for metric in ('balanced_accuracy', 'macro_f1', 'macro_roc_auc_ovr', 'macro_pr_auc_ovr'):
            a, b = (_number(row.get(metric)), _number(baseline.get(metric)))
            result[f'{metric}_delta'] = None if a is None or b is None else a - b
        output.append(result)
    return output
def _quality_tables(
        rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    coverage, distributions, pairs = ([], [], [])
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row.get('case_id',
                           '')), str(row.get('role', row.get('role_family',
                                                             'unknown'))), str(row.get('route_status', row.get('signal_route', 'unknown'))))
        grouped.setdefault(key, []).append(row)
        before = next((_number(row.get(key)) for key in ('direct_hr_bpm', 'hr_direct_bpm', 'pre_hr_bpm')), None)
        after = next((_number(row.get(key)) for key in ('denoised_hr_bpm', 'hr_post_bpm', 'post_hr_bpm')), None)
        if before is not None or after is not None:
            pairs.append({
                'case_id': key[0], 'participant_id': row.get('participant_id'), 'file_id': row.get('file_id'), 'role': key[1],
                'direct_hr_bpm': before, 'denoised_hr_bpm': after, 'delta_bpm': None if before is None or after is None else after - before
            })
    for (case, role, route), values in sorted(grouped.items()):
        retained = sum((bool(row.get('retained', True)) for row in values))
        coverage.append({
            'case_id': case, 'role': role, 'route': route, 'n_total': len(values), 'n_retained': retained,
            'coverage_rate': retained / len(values)
        })
        numeric = sorted({key for row in values for key, value in row.items() if _number(value) is not None} - {'repeat', 'fold', 'label'})
        for metric in numeric:
            points = [float(row[metric]) for row in values if _number(row.get(metric)) is not None]
            distributions.append({
                'case_id': case, 'role': role, 'route': route, 'metric': metric, 'n': len(points), 'mean': fmean(points),
                'sample_sd': stdev(points) if len(points) > 1 else 0.0, 'minimum': min(points), 'maximum': max(points)
            })
    denoiser = _distribution([{**row, 'case_id': str(row.get('case_id', '')), 'balanced_accuracy': row.get('delta_bpm')} for row in pairs])
    return (coverage, distributions, pairs, denoiser)
def _as_oof(row: Mapping[str, Any], line: str) -> OofPredictionRow:
    names = {item.name for item in fields(OofPredictionRow)}
    return replace(OofPredictionRow(**{key: value for key, value in row.items() if key in names}), aggregation_rule=line)
def _aggregation(
    collected: CollectedStudy, bins: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[
        str, Any]], list[str]]:
    summaries, repeats, folds, per_class, matrices, coverage, notes = ([], [], [], [], [], [], [])
    by_case: dict[str, list[Mapping[str, Any]]] = {}
    for row in collected.window_oof_rows:
        by_case.setdefault(str(row.get('case_id', '')), []).append(row)
    config = {str(row.get('case_id')): row.get('aggregation', {}) for row in collected.resolved_aggregation_configs}
    for case, source in sorted(by_case.items()):
        controls = config.get(case, {})
        controls = controls if isinstance(controls, Mapping) else {}
        weighted = bool(controls.get('quality_weighting', False))
        weight_source = controls.get('quality_weight_source', 'route_file_q_rate' if weighted else 'none')
        for line in (LINE_A_EQUAL_FILES, LINE_B_EQUAL_ROLE_FAMILIES):
            view = 'line_a_equal_files' if line == LINE_A_EQUAL_FILES else 'line_b_equal_role_families'
            try:
                result = aggregate_hierarchy((_as_oof(row, line) for row in source), balance_line=line, quality_weighted=weighted,
                                             quality_weight_source=str(weight_source))
            except (KeyError, TypeError, ValueError) as error:
                notes.append(f'{case}/{view}: aggregation replay unavailable: {error}')
                continue
            mappings = [asdict(row) for row in result.participant_rows]
            grouped: dict[int, list[Mapping[str, Any]]] = {}
            for row in mappings:
                grouped.setdefault(int(row['repeat']), []).append(row)
            current = []
            for repeat, rows in sorted(grouped.items()):
                metric, classes, matrix = _metric_row(case, rows, repeat=repeat, ece_bins=bins)
                metric['aggregation_view'] = view
                repeats.append(metric)
                current.append(metric)
                per_class += [{**row, 'aggregation_view': view} for row in classes]
            summary = {'case_id': case, 'aggregation_view': view, 'balance_line': line}
            for metric in _SUMMARY_METRICS:
                values = [value for row in current if (value := _number(row.get(metric))) is not None]
                if values:
                    summary[f'mean_{metric}'] = fmean(values)
            summaries.append(summary)
            pooled, _, matrix = _metric_row(case, mappings, ece_bins=bins)
            if matrix is not None:
                matrices.append(
                    {'case_id': case, 'aggregation_view': view, 'confusion_matrix': matrix, 'metric_source': 'report_reaggregation'})
            coverage += [{'case_id': case, 'aggregation_view': view, **asdict(row)} for row in result.coverage]
    return (summaries, repeats, folds, per_class, matrices, coverage, notes)
def analyze_study(collected: CollectedStudy) -> StudyAnalysis:
    manifest_cases = {str(row.get('case_id')): row for row in collected.manifest.get('cases', ()) if isinstance(row, Mapping)}
    statuses = {str(row.get('case_id')): str(row.get('status', 'unknown')).lower() for row in collected.case_records}
    for case in manifest_cases:
        statuses.setdefault(case, 'unknown')
    execution = collected.plan.get('execution', {}) if isinstance(collected.plan.get('execution'), Mapping) else {}
    expected_repeats = len(execution['repeats']) if isinstance(execution.get('repeats'), (list, tuple)) else None
    expected_cells = expected_repeats * len(execution['folds']) if expected_repeats is not None and isinstance(
        execution.get('folds'), (list, tuple)) else None
    options = collected.plan.get('report', {}) if isinstance(collected.plan.get('report'), Mapping) else {}
    bins = int(options.get('calibration_bins', 10))
    subject_by_case: dict[str, list[Mapping[str, Any]]] = {}
    for row in collected.subject_oof_rows:
        subject_by_case.setdefault(str(row.get('case_id', '')), []).append(row)
    all_repeat, fold_rows, per_class, matrices, calibration, summaries = ([], [], [], [], [], [])
    for case in sorted(set(manifest_cases) | set(subject_by_case) | set(statuses)):
        source = subject_by_case.get(case, [])
        by_repeat: dict[int, list[Mapping[str, Any]]] = {}
        by_fold: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
        for row in source:
            by_repeat.setdefault(int(row.get('repeat', 0)), []).append(row)
            by_fold.setdefault((int(row.get('repeat', 0)), int(row.get('fold', 0))), []).append(row)
        case_repeat = []
        for repeat, rows in sorted(by_repeat.items()):
            metric, classes, _ = _metric_row(case, rows, repeat=repeat, ece_bins=bins)
            case_repeat.append(metric)
            all_repeat.append(metric)
            per_class += [{**row, 'metric_source': 'participant_oof_per_repeat'} for row in classes]
        case_folds = []
        for (repeat, fold), rows in sorted(by_fold.items()):
            metric, _, _ = _metric_row(case, rows, repeat=repeat, fold=fold, ece_bins=bins)
            case_folds.append(metric)
            fold_rows.append(metric)
        pooled, classes, matrix = _metric_row(case, source, ece_bins=bins)
        per_class += [{**row, 'metric_source': 'participant_oof_pooled_repeats'} for row in classes]
        if matrix is not None:
            matrices.append({
                'case_id': case, 'class_order': list(_class_order(source)), 'confusion_matrix': matrix,
                'metric_source': 'participant_oof_pooled_repeats'
            })
        calibration += _calibration(case, source, bins)
        summary = _summary(case, case_repeat, case_folds, statuses.get(case, 'unknown'), expected_repeats, expected_cells)
        summary.update({key: value for key, value in pooled.items() if key not in {'case_id', 'repeat', 'fold', 'status'}})
        summary['frailty_classification_evaluation_scope'] = 'outer_heldout_participant_oof'
        summaries.append(summary)
    distribution = _distribution(all_repeat)
    leaderboard = sorted(
        (row for row in summaries if _number(
            row.get('participant_mean_abstention_aware_balanced_accuracy', row.get('participant_mean_balanced_accuracy'))) is not None),
        key=lambda row:
        (-float(row.get('participant_mean_abstention_aware_balanced_accuracy', row.get('participant_mean_balanced_accuracy', 0.0))),
         str(row['case_id'])))
    predictive = [{'predictive_rank': rank, **row, 'decision': 'manual_review_only_no_automatic_winner'}
                  for rank, row in enumerate(leaderboard[:int(options.get('top_k', 10))], 1)]
    worst = sorted(predictive, key=lambda row: (-float(row.get('worst_class_f1', 0.0) or 0.0), str(row['case_id'])))
    worst = [{'worst_class_f1_stability_rank': rank, **row} for rank, row in enumerate(worst[:10], 1)]
    reference = collected.manifest.get('reference_case_id', collected.manifest.get('reference_case'))
    reference = str(reference) if reference not in (None, '') else None
    scores = normalize_classification_rows(collected.subject_oof_rows, evaluation_id='participant_outer_oof',
                                           aggregation_level='participant')
    roc = classification_roc_curve_rows(scores, macro_grid_points=int(options.get('classification_roc_macro_grid_points', 201)))
    tsne = classification_tsne_rows(scores, random_state=int(options.get('classification_tsne_random_state', 42)), max_samples=int(
        options.get('classification_tsne_max_samples', 5000))) if scores else ()
    diagnostics = classification_diagnostic_status_rows(tuple(sorted(set(manifest_cases) | set(subject_by_case))), scores, roc, tsne)
    classifier_classes = classification_per_class_metric_rows(scores, class_names=CANONICAL_CLASS_NAMES)
    policy = next((row.get('evaluation_statistics')
                   for row in collected.resolved_aggregation_configs if isinstance(row.get('evaluation_statistics'), Mapping)), {})
    paired = paired_inference_against_reference(
        scores, reference_case_id=reference, comparison_family=str(collected.plan.get('study', {}).get(
            'study_id', 'declared_comparison')), inference_role='declared_reference_comparison', candidate_case_ids=tuple(
                sorted(set(subject_by_case) - ({reference} if reference else set()))), expected_repeats=tuple(execution.get('repeats', ()))
        or None, n_resamples=int(policy.get('paired_permutation_replicates', 100000)), bootstrap_resamples=int(
            policy.get('bootstrap_replicates', 10000)), seed=int(policy.get('seed', 42))) if reference and len(subject_by_case) > 1 else []
    counts, normalized = _confusion_long(matrices)
    route, quality, denoiser_pairs, denoiser_summary = _quality_tables(collected.quality_rows)
    aggregation, aggregation_repeats, aggregation_folds, aggregation_classes, aggregation_matrices, hierarchy_coverage, aggregation_notes = _aggregation(
        collected, bins)
    bridge = collected.plan.get('legacy_bridge')
    centered = isinstance(bridge, Mapping) and str(bridge.get('design', '')) == 'centered_star_v1'
    stage_absolute = aggregation if centered else []
    stage_contrasts = _paired_deltas(aggregation_repeats, reference) if centered else []
    legacy_numeric = aggregation if isinstance(bridge, Mapping) and (not centered) else []
    deployment = [{
        'case_id': row['case_id'], 'parameter_count': next(
            (item.get('parameter_count')
             for item in collected.cell_rows if str(item.get('case_id')) == row['case_id'] and item.get('parameter_count') is not None),
            None), 'deployment_readiness': 'measured' if any(
                (str(item.get('case_id')) == row['case_id'] and item.get('parameter_count') is not None
                 for item in collected.cell_rows)) else 'N/A_pending_hardware_evidence'
    } for row in summaries]
    coverage = [{
        'case_id': row['case_id'], 'n_total': row.get('n_total'), 'n_retained': row.get('n_retained'), 'n_dropped': row.get('n_dropped'),
        'coverage_rate': row.get('coverage_rate')
    } for row in summaries]
    notes = tuple(
        dict.fromkeys((*collected.limitations, *aggregation_notes,
                       *(('Classification t-SNE uses persisted probability vectors, not hidden features.', ) if tsne else ()))))
    return StudyAnalysis(
        case_summary=tuple(summaries), metric_distribution_summary=tuple(distribution), repeat_metrics=tuple(all_repeat),
        fold_metrics=tuple(fold_rows), per_class_metrics=tuple(per_class), confusion_matrices=tuple(matrices),
        confusion_counts=tuple(counts), confusion_row_normalized=tuple(normalized), calibration_bins=tuple(calibration),
        paired_deltas=tuple(_paired_deltas(all_repeat, reference)), paired_participant_inference=tuple(paired),
        aggregation_line_comparison=tuple(aggregation), aggregation_line_repeat_metrics=tuple(aggregation_repeats),
        aggregation_line_per_class_metrics=tuple(aggregation_classes), aggregation_view_comparison=tuple(aggregation),
        aggregation_view_repeat_metrics=tuple(aggregation_repeats), aggregation_view_fold_metrics=tuple(aggregation_folds),
        aggregation_view_per_class_metrics=tuple(aggregation_classes), aggregation_view_confusion_matrices=tuple(aggregation_matrices),
        aggregation_hierarchy_coverage=tuple(hierarchy_coverage), legacy_bridge_numeric_ablation_report=tuple(legacy_numeric),
        legacy_bridge_execution_order_report=tuple(collected.cell_rows if isinstance(bridge, Mapping) and (not centered) else ()),
        coverage=tuple(coverage), route_role_coverage=tuple(route), quality_distributions=tuple(quality),
        predictive_leaderboard=tuple(predictive), worst_class_f1_stability=tuple(worst), incomplete_cases=tuple(
            (row for row in summaries if not row['complete_for_requested_execution'])), deployment_table=tuple(deployment), notes=notes,
        denoiser_hr_record_pairs=tuple(denoiser_pairs), denoiser_hr_comparison=tuple(denoiser_summary), repeat_per_class_metrics=tuple(
            (row for row in per_class if row.get('repeat') is not None)), per_class_metric_distribution_summary=tuple(
                _distribution(per_class, group_field='class_name')), stage3_star_absolute=tuple(stage_absolute),
        stage3_star_contrasts=tuple(stage_contrasts), stage3_star_fold_contrasts=tuple(stage_contrasts),
        stage3_star_execution=tuple(collected.cell_rows if centered else ()), stage3_star_inception_comparison=tuple(
            (row for row in stage_absolute if 'inception' in str(row).lower())), stage3_star_cnn_comparison=tuple(
                (row for row in stage_absolute if 'cnn' in str(row).lower())), stage3_star_model_comparison=tuple(stage_absolute),
        classification_prediction_scores=tuple(scores), classification_roc_curves=tuple(roc), classification_prediction_tsne=tuple(tsne),
        classification_diagnostic_status=tuple(diagnostics), classifier_per_class_results=tuple(classifier_classes))
__all__ = ['StudyAnalysis', 'analyze_study']
