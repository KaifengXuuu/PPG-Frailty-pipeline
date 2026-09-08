"""Shared evidence tables and conservative, machine-readable conclusions."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from ..training.statistics import ParticipantPrediction, holm_adjust_by_family_metric, paired_participant_cluster_bootstrap, paired_participant_permutation
from .tabular import markdown_column_definitions_block

DEFAULT_PAIRED_PERMUTATION_RESAMPLES = 100000
DEFAULT_PAIRED_BOOTSTRAP_RESAMPLES = 10000
DEFAULT_REPORTING_RANDOM_SEED = 42
_MAX_HUMAN_TABLE_COLUMNS = 8
def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None
def _percent(value: Any) -> str:
    numeric = _number(value)
    return 'N/A' if numeric is None else f'{100.0 * numeric:.1f}'
def _mean_sd(mean: Any, sd: Any, *, repeat_count: Any = None) -> str:
    left = _number(mean)
    right = _number(sd)
    if left is None:
        return 'N/A'
    if right is None:
        try:
            count = int(repeat_count)
        except (TypeError, ValueError):
            count = 0
        suffix = f'; n={count} repeat' if count == 1 else ''
        return f'{100.0 * left:.1f} (SD N/A{suffix})'
    return f'{100.0 * left:.1f} ± {100.0 * right:.1f}'
def _ci(lower: Any, upper: Any) -> str:
    low = _number(lower)
    high = _number(upper)
    if low is None or high is None:
        return 'N/A'
    return f'[{100.0 * low:.1f}, {100.0 * high:.1f}]'
def classification_comparison_rows(case_summary: Sequence[Mapping[str, Any]], *,
                                   paired_inference: Sequence[Mapping[str, Any]] = ()) -> list[dict[str, Any]]:
    inference: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in paired_inference:
        if row.get('candidate_case_id') is None:
            continue
        key = (str(row.get('candidate_case_id')), str(row.get('metric')))
        previous = inference.get(key)
        if previous is not None and (str(previous.get('reference_case_id')) != str(row.get('reference_case_id'))
                                     or str(previous.get('comparison_family')) != str(row.get('comparison_family'))):
            raise ValueError(
                f'comparison table cannot collapse multiple inference families for candidate/metric {key}; emit separate comparison tables')
        inference[key] = row
    rows = [dict(row) for row in case_summary if _number(row.get('participant_mean_balanced_accuracy')) is not None]
    rows.sort(key=lambda row: (-float(row['participant_mean_balanced_accuracy']), -float(row.get('participant_mean_macro_f1', 0.0)),
                               str(row.get('case_id', ''))))
    output: list[dict[str, Any]] = []
    for rank, row in enumerate(rows, start=1):
        case_id = str(row.get('case_id', ''))
        ba_test = inference.get((case_id, 'balanced_accuracy'), {})
        f1_test = inference.get((case_id, 'macro_f1'), {})
        roc_test = inference.get((case_id, 'macro_roc_auc_ovr'), {})
        output.append({
            'rank': rank, 'case_id': case_id, 'status': row.get('status'),
            'complete_for_requested_execution': row.get('complete_for_requested_execution'), 'repeat_count': row.get('repeat_count'),
            'fold_cell_count': row.get('fold_cell_count'), 'balanced_accuracy_mean': row.get('participant_mean_balanced_accuracy'),
            'balanced_accuracy_sample_sd': row.get('repeat_balanced_accuracy_sample_sd'), 'balanced_accuracy_mean_sd_percent': _mean_sd(
                row.get('participant_mean_balanced_accuracy'), row.get('repeat_balanced_accuracy_sample_sd'),
                repeat_count=row.get('repeat_count')), 'balanced_accuracy_repeat_t_ci95_low': row.get('repeat_balanced_accuracy_ci95_low'),
            'balanced_accuracy_repeat_t_ci95_high': row.get('repeat_balanced_accuracy_ci95_high'),
            'balanced_accuracy_repeat_t_ci95_percent': _ci(row.get('repeat_balanced_accuracy_ci95_low'),
                                                           row.get('repeat_balanced_accuracy_ci95_high')),
            'balanced_accuracy_participant_cluster_ci95_low': row.get('participant_cluster_balanced_accuracy_ci95_low'),
            'balanced_accuracy_participant_cluster_ci95_high': row.get('participant_cluster_balanced_accuracy_ci95_high'),
            'balanced_accuracy_participant_cluster_ci95_percent': _ci(row.get('participant_cluster_balanced_accuracy_ci95_low'),
                                                                      row.get('participant_cluster_balanced_accuracy_ci95_high')),
            'macro_f1_mean': row.get('participant_mean_macro_f1'), 'macro_f1_sample_sd': row.get('repeat_macro_f1_sample_sd'),
            'macro_f1_mean_sd_percent': _mean_sd(row.get('participant_mean_macro_f1'), row.get('repeat_macro_f1_sample_sd'),
                                                 repeat_count=row.get('repeat_count')), 'macro_f1_repeat_t_ci95_percent': _ci(
                                                     row.get('repeat_macro_f1_ci95_low'), row.get('repeat_macro_f1_ci95_high')),
            'macro_f1_participant_cluster_ci95_low': row.get('participant_cluster_macro_f1_ci95_low'),
            'macro_f1_participant_cluster_ci95_high': row.get('participant_cluster_macro_f1_ci95_high'),
            'macro_f1_participant_cluster_ci95_percent': _ci(
                row.get('participant_cluster_macro_f1_ci95_low'),
                row.get('participant_cluster_macro_f1_ci95_high')), 'macro_roc_auc_ovr_mean': row.get('participant_mean_macro_roc_auc_ovr'),
            'macro_roc_auc_ovr_sample_sd': row.get('repeat_macro_roc_auc_ovr_sample_sd'), 'macro_roc_auc_ovr_mean_sd_percent': _mean_sd(
                row.get('participant_mean_macro_roc_auc_ovr'), row.get('repeat_macro_roc_auc_ovr_sample_sd'),
                repeat_count=row.get('repeat_count')), 'macro_roc_auc_ovr_repeat_t_ci95_percent': _ci(
                    row.get('repeat_macro_roc_auc_ovr_ci95_low'), row.get('repeat_macro_roc_auc_ovr_ci95_high')),
            'macro_roc_auc_ovr_participant_cluster_ci95_low': row.get('participant_cluster_macro_roc_auc_ovr_ci95_low'),
            'macro_roc_auc_ovr_participant_cluster_ci95_high': row.get('participant_cluster_macro_roc_auc_ovr_ci95_high'),
            'macro_roc_auc_ovr_participant_cluster_ci95_percent': _ci(
                row.get('participant_cluster_macro_roc_auc_ovr_ci95_low'),
                row.get('participant_cluster_macro_roc_auc_ovr_ci95_high')), 'participant_cluster_ci_applicability': row.get(
                    'participant_cluster_ci_applicability',
                    'N/A'), 'participant_cluster_ci_reason': row.get(
                        'participant_cluster_ci_reason',
                        'N/A_not_recorded_by_source_reporter'), 'macro_pr_auc_ovr_mean_sd_percent': _mean_sd(
                            row.get('participant_mean_macro_pr_auc_ovr'), row.get('repeat_macro_pr_auc_ovr_sample_sd'),
                            repeat_count=row.get('repeat_count')), 'macro_pr_auc_ovr_repeat_t_ci95_percent': _ci(
                                row.get('repeat_macro_pr_auc_ovr_ci95_low'),
                                row.get('repeat_macro_pr_auc_ovr_ci95_high')), 'worst_fold_balanced_accuracy_percent': _percent(
                                    row.get('worst_fold_balanced_accuracy')), 'worst_class_recall_percent': _percent(
                                        row.get('worst_class_recall')), 'worst_class_f1_percent': _percent(
                                            row.get('worst_class_f1')), 'expected_calibration_error': row.get('expected_calibration_error'),
            'paired_reference_case_id': ba_test.get('reference_case_id'),
            'ba_candidate_minus_reference': ba_test.get('candidate_minus_reference'),
            'ba_raw_two_sided_p': ba_test.get('raw_two_sided_p_value'), 'ba_holm_adjusted_p': ba_test.get('holm_adjusted_p_value'),
            'ba_holm_reject': ba_test.get('reject_null_after_holm'),
            'f1_candidate_minus_reference': f1_test.get('candidate_minus_reference'),
            'f1_raw_two_sided_p': f1_test.get('raw_two_sided_p_value'), 'f1_holm_adjusted_p': f1_test.get('holm_adjusted_p_value'),
            'f1_holm_reject': f1_test.get('reject_null_after_holm'), 'ba_paired_delta_cluster_ci95_percent': _ci(
                ba_test.get('participant_cluster_delta_ci95_low'), ba_test.get('participant_cluster_delta_ci95_high')),
            'ba_paired_delta_cluster_ci95_low': ba_test.get('participant_cluster_delta_ci95_low'),
            'ba_paired_delta_cluster_ci95_high': ba_test.get('participant_cluster_delta_ci95_high'),
            'f1_paired_delta_cluster_ci95_percent': _ci(f1_test.get('participant_cluster_delta_ci95_low'),
                                                        f1_test.get('participant_cluster_delta_ci95_high')),
            'f1_paired_delta_cluster_ci95_low': f1_test.get('participant_cluster_delta_ci95_low'),
            'f1_paired_delta_cluster_ci95_high': f1_test.get('participant_cluster_delta_ci95_high'),
            'roc_auc_candidate_minus_reference': roc_test.get('candidate_minus_reference'),
            'roc_auc_paired_delta_cluster_ci95_percent': _ci(roc_test.get('participant_cluster_delta_ci95_low'),
                                                             roc_test.get('participant_cluster_delta_ci95_high')),
            'roc_auc_paired_delta_cluster_ci95_low': roc_test.get('participant_cluster_delta_ci95_low'),
            'roc_auc_paired_delta_cluster_ci95_high': roc_test.get('participant_cluster_delta_ci95_high'),
            'roc_auc_p_value_applicability': roc_test.get('p_value_applicability',
                                                          'N/A_no_registered_roc_auc_permutation_test'), 'inference_role': ba_test.get(
                                                              'inference_role', 'N/A_no_eligible_paired_comparison')
        })
    return output
_CLASSIFICATION_COMPARISON_METRICS = (('balanced_accuracy', 'balanced_accuracy_mean_sd_percent', 'balanced_accuracy_repeat_t_ci95_percent',
                                       'balanced_accuracy_participant_cluster_ci95_percent'), ('macro_f1', 'macro_f1_mean_sd_percent',
                                                                                               'macro_f1_repeat_t_ci95_percent',
                                                                                               'macro_f1_participant_cluster_ci95_percent'),
                                      ('macro_roc_auc_ovr', 'macro_roc_auc_ovr_mean_sd_percent', 'macro_roc_auc_ovr_repeat_t_ci95_percent',
                                       'macro_roc_auc_ovr_participant_cluster_ci95_percent'),
                                      ('macro_pr_auc_ovr', 'macro_pr_auc_ovr_mean_sd_percent', 'macro_pr_auc_ovr_repeat_t_ci95_percent',
                                       None))
def _display_value(value: Any) -> Any:
    return 'N/A' if value is None else value
def _embedded_paired_inference_rows(comparison_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    metric_fields = (('balanced_accuracy', 'ba_candidate_minus_reference', 'ba_paired_delta_cluster_ci95_percent', 'ba_holm_adjusted_p',
                      'ba_holm_reject', None), ('macro_f1', 'f1_candidate_minus_reference', 'f1_paired_delta_cluster_ci95_percent',
                                                'f1_holm_adjusted_p', 'f1_holm_reject', None),
                     ('macro_roc_auc_ovr', 'roc_auc_candidate_minus_reference', 'roc_auc_paired_delta_cluster_ci95_percent', None, None,
                      'roc_auc_p_value_applicability'))
    for row in comparison_rows:
        reference = row.get('paired_reference_case_id')
        if reference is None:
            continue
        for metric, delta, interval, p_value, reject, applicability in metric_fields:
            adjusted_p = row.get(p_value) if p_value is not None else None
            output.append({
                'candidate_case_id': str(row.get('case_id', '')), 'reference_case_id': str(reference), 'metric': metric,
                'candidate_minus_reference': row.get(delta), 'participant_cluster_delta_ci95_percent': row.get(interval),
                'holm_adjusted_p_value': adjusted_p, 'reject_null_after_holm': row.get(reject) if reject is not None else None,
                'p_value_applicability': row.get(applicability)
                if applicability is not None else 'available' if _number(adjusted_p) is not None else 'N/A_no_eligible_paired_comparison'
            })
    return output
def classification_comparison_table_views(comparison_rows: Sequence[Mapping[str, Any]], *,
                                          paired_inference: Sequence[Mapping[str, Any]] = ()) -> dict[str, list[dict[str, Any]]]:
    source = [dict(row) for row in comparison_rows]
    ranking_performance = [{
        'case_id': str(row.get('case_id', '')), 'rank': row.get('rank'), 'balanced_accuracy_mean_sd_percent': _display_value(
            row.get('balanced_accuracy_mean_sd_percent')), 'macro_f1_mean_sd_percent': _display_value(row.get('macro_f1_mean_sd_percent')),
        'macro_roc_auc_ovr_mean_sd_percent': _display_value(row.get('macro_roc_auc_ovr_mean_sd_percent')),
        'macro_pr_auc_ovr_mean_sd_percent': _display_value(row.get('macro_pr_auc_ovr_mean_sd_percent')),
        'repeat_count': _display_value(row.get('repeat_count')),
        'complete_for_requested_execution': _display_value(row.get('complete_for_requested_execution'))
    } for row in source]
    if not ranking_performance:
        ranking_performance = [{
            'case_id': 'N/A_no_comparable_classifier_evidence', 'rank': 'N/A', 'balanced_accuracy_mean_sd_percent': 'N/A',
            'macro_f1_mean_sd_percent': 'N/A', 'macro_roc_auc_ovr_mean_sd_percent': 'N/A', 'macro_pr_auc_ovr_mean_sd_percent': 'N/A',
            'repeat_count': 'N/A', 'complete_for_requested_execution': 'N/A'
        }]
    uncertainty_ci: list[dict[str, Any]] = []
    for row in source:
        for metric, mean_sd, repeat_ci, cluster_ci in _CLASSIFICATION_COMPARISON_METRICS:
            cluster_value = row.get(cluster_ci) if cluster_ci is not None else None
            cluster_available = cluster_value is not None and cluster_value != 'N/A'
            uncertainty_ci.append({
                'case_id': str(row.get('case_id', '')), 'metric': metric, 'mean_sd_percent': _display_value(row.get(mean_sd)),
                'repeat_t_ci95_percent': _display_value(row.get(repeat_ci)),
                'participant_cluster_ci95_percent': _display_value(cluster_value),
                'participant_cluster_ci_applicability': 'available' if cluster_available else 'N/A_not_registered_for_macro_pr_auc'
                if cluster_ci is None else _display_value(row.get('participant_cluster_ci_applicability')),
                'participant_cluster_ci_reason': '' if cluster_available else 'participant_cluster_macro_pr_auc_interval_not_registered'
                if cluster_ci is None else _display_value(row.get('participant_cluster_ci_reason')),
                'repeat_count': _display_value(row.get('repeat_count'))
            })
    if not uncertainty_ci:
        uncertainty_ci = [{
            'case_id': 'N/A_no_comparable_classifier_evidence', 'metric': 'N/A', 'mean_sd_percent': 'N/A', 'repeat_t_ci95_percent': 'N/A',
            'participant_cluster_ci95_percent': 'N/A', 'participant_cluster_ci_applicability': 'N/A',
            'participant_cluster_ci_reason': 'no_comparable_classifier_evidence', 'repeat_count': 'N/A'
        }]
    raw_inference = [dict(row) for row in paired_inference] if paired_inference else _embedded_paired_inference_rows(source)
    paired_rows = [{
        'candidate_case_id': str(row.get('candidate_case_id', '')), 'reference_case_id': _display_value(row.get('reference_case_id')),
        'metric': str(row.get('metric', '')), 'candidate_minus_reference_percent': _percent(row.get('candidate_minus_reference')),
        'participant_cluster_delta_ci95_percent': _ci(row.get('participant_cluster_delta_ci95_low'),
                                                      row.get('participant_cluster_delta_ci95_high'))
        if 'participant_cluster_delta_ci95_low' in row or 'participant_cluster_delta_ci95_high' in row else _display_value(
            row.get('participant_cluster_delta_ci95_percent')), 'holm_adjusted_p_value': _display_value(row.get('holm_adjusted_p_value')),
        'reject_null_after_holm': _display_value(row.get('reject_null_after_holm')),
        'p_value_applicability': _display_value(row.get('p_value_applicability'))
    } for row in raw_inference]
    if not paired_rows:
        paired_rows = [{
            'candidate_case_id': 'N/A_no_eligible_paired_comparison', 'reference_case_id': 'N/A', 'metric': 'N/A',
            'candidate_minus_reference_percent': 'N/A', 'participant_cluster_delta_ci95_percent': 'N/A', 'holm_adjusted_p_value': 'N/A',
            'reject_null_after_holm': 'N/A', 'p_value_applicability': 'N/A_no_eligible_paired_comparison'
        }]
    robustness = [{
        'case_id': str(row.get('case_id', '')),
        'worst_fold_balanced_accuracy_percent': _display_value(row.get('worst_fold_balanced_accuracy_percent')),
        'worst_class_recall_percent': _display_value(row.get('worst_class_recall_percent')),
        'worst_class_f1_percent': _display_value(row.get('worst_class_f1_percent')),
        'expected_calibration_error': _display_value(row.get('expected_calibration_error')), 'status': _display_value(row.get('status')),
        'complete_for_requested_execution': _display_value(row.get('complete_for_requested_execution')),
        'fold_cell_count': _display_value(row.get('fold_cell_count'))
    } for row in source]
    if not robustness:
        robustness = [{
            'case_id': 'N/A_no_comparable_classifier_evidence', 'worst_fold_balanced_accuracy_percent': 'N/A',
            'worst_class_recall_percent': 'N/A', 'worst_class_f1_percent': 'N/A', 'expected_calibration_error': 'N/A', 'status': 'N/A',
            'complete_for_requested_execution': 'N/A', 'fold_cell_count': 'N/A'
        }]
    output = {
        'ranking_performance': ranking_performance, 'uncertainty_ci': uncertainty_ci, 'paired_inference': paired_rows,
        'robustness': robustness
    }
    if any((len(row) > 8 for rows in output.values() for row in rows)):
        raise AssertionError('compact classifier comparison tables exceed eight columns')
    return output
def paired_inference_against_reference(prediction_rows: Sequence[Mapping[str, Any]], *, reference_case_id: str, comparison_family: str,
                                       inference_role: str, candidate_case_ids: Sequence[str] | None = None,
                                       expected_repeats: Sequence[int] | None = None,
                                       expected_membership: Mapping[tuple[str, int], tuple[int, int]] | None = None,
                                       n_resamples: int = DEFAULT_PAIRED_PERMUTATION_RESAMPLES,
                                       bootstrap_resamples: int = DEFAULT_PAIRED_BOOTSTRAP_RESAMPLES,
                                       seed: int = DEFAULT_REPORTING_RANDOM_SEED) -> list[dict[str, Any]]:
    grouped: dict[str, list[ParticipantPrediction]] = {}
    authority: dict[str, dict[tuple[str, int], tuple[int, int, int]]] = {}
    class_order_by_case: dict[str, tuple[int, ...]] = {}
    seen_case_ids: set[str] = set()
    invalid_case_reasons: dict[str, str] = {}
    for row in prediction_rows:
        case_id = str(row.get('classifier_id', row.get('case_id', '')))
        if not case_id:
            continue
        seen_case_ids.add(case_id)
        try:
            participant_id = str(row.get('participant_id', '')).strip()
            repeat = int(row.get('repeat', -1))
            fold = int(row.get('fold', -1))
            split_seed = int(row.get('split_seed', -1))
            label = int(row.get('true_label', row.get('label', -1)))
            probabilities = tuple((float(value) for value in row.get('probabilities', ())))
            raw_order = row.get('class_order')
            class_order = tuple(
                (int(value)
                 for value in raw_order)) if raw_order is not None and (not isinstance(raw_order,
                                                                                       (str, bytes))) else tuple(range(len(probabilities)))
        except (TypeError, ValueError):
            invalid_case_reasons[case_id] = 'invalid_prediction_contract'
            continue
        key = (participant_id, repeat)
        if not case_id or not participant_id or repeat < 0 or (fold < 0) or (split_seed < 0) or (len(class_order) < 2) or (len(
                set(class_order)) != len(class_order)) or (label not in class_order) or (len(probabilities) != len(class_order)) or (
                    row.get('retained', True) is False):
            invalid_case_reasons[case_id] = 'invalid_or_abstained_participant_oof_contract'
            continue
        previous_order = class_order_by_case.setdefault(case_id, class_order)
        if previous_order != class_order:
            invalid_case_reasons[case_id] = 'inconsistent_class_order'
            continue
        if key in authority.setdefault(case_id, {}):
            invalid_case_reasons[case_id] = 'duplicate_participant_repeat_prediction'
            continue
        try:
            prediction = ParticipantPrediction(participant_id, label, repeat, probabilities)
        except ValueError:
            invalid_case_reasons[case_id] = 'invalid_probability_vector'
            continue
        grouped.setdefault(case_id, []).append(prediction)
        authority[case_id][key] = (fold, split_seed, label)
    reference_case_id = str(reference_case_id)
    declared_candidates = {str(value)
                           for value in candidate_case_ids if str(value) and str(value) != reference_case_id
                           } if candidate_case_ids is not None else (seen_case_ids | set(grouped)) - {reference_case_id}

    def na_rows(candidate_case_id: str, reason: str) -> list[dict[str, Any]]:
        comparison_id = f'{candidate_case_id}_vs_{reference_case_id}'
        return [{
            'comparison_family': comparison_family, 'comparison_id': comparison_id, 'reference_case_id': reference_case_id,
            'candidate_case_id': candidate_case_id, 'metric': metric, 'candidate_minus_reference': None,
            'participant_cluster_delta_ci95_low': None, 'participant_cluster_delta_ci95_high': None,
            'bootstrap_resamples': bootstrap_resamples, 'bootstrap_valid_resamples': None, 'bootstrap_seed': seed,
            'bootstrap_cluster_unit': 'participant_with_all_repeats', 'bootstrap_interval_method': 'percentile_two_sided_95',
            'raw_two_sided_p_value': None, 'n_resamples': None, 'seed': seed, 'participant_count': None, 'repeat_count': None,
            'exchange_unit': 'participant_with_all_repeats', 'test_method': 'N/A_incompatible_declared_pair',
            'p_value_applicability': f'N/A_{reason}', 'comparison_contract_status': f'N/A_{reason}', 'inference_role': inference_role,
            'automatic_selection': False, 'holm_adjusted_p_value': None, 'holm_rank': None, 'holm_family_size': None, 'alpha': None,
            'reject_null_after_holm': None,
            'interpretation': 'Declared pair retained as N/A because compatible matched participant OOF evidence is unavailable.'
        } for metric in ('balanced_accuracy', 'macro_f1', 'macro_roc_auc_ovr')]

    reference_reason = invalid_case_reasons.get(reference_case_id)
    if reference_reason is None and reference_case_id not in grouped:
        reference_reason = 'reference_has_no_valid_participant_oof'
    if reference_reason is not None:
        return [
            row for candidate_case_id in sorted(declared_candidates) for row in na_rows(candidate_case_id, f'reference_{reference_reason}')
        ]
    reference = tuple(grouped[reference_case_id])
    reference_authority = authority[reference_case_id]
    reference_class_order = class_order_by_case[reference_case_id]
    observed_reference_repeats = {repeat for _participant_id, repeat in reference_authority}
    declared_repeat_set = {int(value) for value in expected_repeats} if expected_repeats is not None else observed_reference_repeats
    if observed_reference_repeats != declared_repeat_set:
        reason = 'reference_declared_repeat_roster_mismatch'
        return [row for candidate_case_id in sorted(declared_candidates) for row in na_rows(candidate_case_id, reason)]
    if any((len(
        {split_seed
         for (_participant_id, row_repeat), (_fold, split_seed, _label) in reference_authority.items() if row_repeat == repeat}) != 1
            for repeat in declared_repeat_set)):
        reason = 'reference_repeat_has_multiple_split_seeds'
        return [row for candidate_case_id in sorted(declared_candidates) for row in na_rows(candidate_case_id, reason)]
    if expected_membership is not None:
        observed_membership = {key: (value[0], value[1]) for key, value in reference_authority.items()}
        normalized_expected = {(str(key[0]), int(key[1])): (int(value[0]), int(value[1])) for key, value in expected_membership.items()}
        if observed_membership != normalized_expected:
            reason = 'reference_frozen_split_registry_roster_mismatch'
            return [row for candidate_case_id in sorted(declared_candidates) for row in na_rows(candidate_case_id, reason)]
    raw_values: dict[tuple[str, str, str], float] = {}
    raw_rows: dict[tuple[str, str], dict[str, Any]] = {}
    unavailable_rows: list[dict[str, Any]] = []
    for case_id in sorted(declared_candidates):
        reason = invalid_case_reasons.get(case_id)
        if reason is None and case_id not in grouped:
            reason = 'candidate_has_no_valid_participant_oof'
        if reason is None and authority.get(case_id) != reference_authority:
            reason = 'participant_repeat_fold_split_seed_label_roster_mismatch'
        if reason is None and class_order_by_case.get(case_id) != reference_class_order:
            reason = 'candidate_reference_class_order_mismatch'
        if reason is not None:
            unavailable_rows.extend(na_rows(case_id, reason))
            continue
        comparison_id = f'{case_id}_vs_{reference_case_id}'
        for metric in ('balanced_accuracy', 'macro_f1', 'macro_roc_auc_ovr'):
            interval = paired_participant_cluster_bootstrap(reference, tuple(grouped[case_id]), class_order=reference_class_order,
                                                            metric=metric, n_resamples=bootstrap_resamples, seed=seed)
            permutation = None
            if metric in {'balanced_accuracy', 'macro_f1'}:
                permutation = paired_participant_permutation(reference, tuple(grouped[case_id]), class_order=reference_class_order,
                                                             metric=metric, n_resamples=n_resamples, seed=seed)
                raw_values[comparison_family, metric, comparison_id] = permutation.two_sided_p_value
            raw_rows[comparison_id, metric] = {
                'comparison_family': comparison_family, 'comparison_id': comparison_id, 'reference_case_id': reference_case_id,
                'candidate_case_id': case_id, 'metric': metric, 'candidate_minus_reference': interval.observed_candidate_minus_reference,
                'participant_cluster_delta_ci95_low': interval.ci95_lower, 'participant_cluster_delta_ci95_high': interval.ci95_upper,
                'bootstrap_resamples': interval.n_resamples, 'bootstrap_valid_resamples': interval.valid_resamples,
                'bootstrap_seed': interval.seed, 'bootstrap_cluster_unit': interval.cluster_unit,
                'bootstrap_interval_method': interval.interval_method, 'bootstrap_implementation_version': interval.implementation_version,
                'bootstrap_rng_contract': interval.rng_contract,
                'raw_two_sided_p_value': None if permutation is None else permutation.two_sided_p_value,
                'n_resamples': None if permutation is None else permutation.n_resamples, 'seed': seed,
                'participant_count': interval.n_participants, 'repeat_count': interval.n_repeats,
                'exchange_unit': interval.cluster_unit if permutation is None else permutation.exchange_unit,
                'test_method': 'paired_participant_cluster_bootstrap_ci_only'
                if permutation is None else 'paired_participant_cluster_bootstrap_and_permutation',
                'permutation_implementation_version': None if permutation is None else permutation.implementation_version,
                'permutation_rng_contract': None if permutation is None else permutation.rng_contract,
                'p_value_applicability': 'N/A_no_registered_roc_auc_permutation_test'
                if permutation is None else 'available_two_sided_participant_cluster_permutation',
                'comparison_contract_status': 'matched_complete_roster', 'inference_role': inference_role, 'automatic_selection': False
            }
    adjusted = {(row.comparison_id, row.metric): row for row in holm_adjust_by_family_metric(raw_values, alpha=0.05)} if raw_values else {}
    output: list[dict[str, Any]] = list(unavailable_rows)
    for key, row in sorted(raw_rows.items()):
        holm = adjusted.get(key)
        output.append({
            **row, 'holm_adjusted_p_value': None if holm is None else holm.adjusted_p_value,
            'holm_rank': None if holm is None else holm.rank, 'holm_family_size': None if holm is None else holm.family_size,
            'alpha': None if holm is None else holm.alpha, 'reject_null_after_holm': None if holm is None else holm.reject_null,
            'interpretation': 'exploratory post-selection contrast; CI/P are not posterior confidence and do not alter the persisted winner'
            if inference_role.startswith('exploratory') else 'paired outer-OOF comparison; no automatic selection'
        })
    return output
def holm_adjust_paired_inference_rows(rows: Sequence[Mapping[str, Any]], *, alpha: float = 0.05) -> list[dict[str, Any]]:
    raw_values: dict[tuple[str, str, str], float] = {}
    for row in rows:
        raw_p = _number(row.get('raw_two_sided_p_value'))
        if raw_p is None:
            continue
        key = (str(row.get('comparison_family', '')), str(row.get('metric', '')), str(row.get('comparison_id', '')))
        if not all(key):
            raise ValueError('Holm-adjustable inference row lacks family/metric/id')
        if key in raw_values:
            raise ValueError(f'duplicate paired-inference Holm key: {key}')
        raw_values[key] = raw_p
    adjusted = {(row.comparison_family, row.metric, row.comparison_id): row
                for row in holm_adjust_by_family_metric(raw_values, alpha=alpha)} if raw_values else {}
    output: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        key = (str(row.get('comparison_family', '')), str(row.get('metric', '')), str(row.get('comparison_id', '')))
        result = adjusted.get(key)
        if result is None:
            row.update(
                {'holm_adjusted_p_value': None, 'holm_rank': None, 'holm_family_size': None, 'alpha': None, 'reject_null_after_holm': None})
        else:
            row.update({
                'holm_adjusted_p_value': result.adjusted_p_value, 'holm_rank': result.rank, 'holm_family_size': result.family_size,
                'alpha': result.alpha, 'reject_null_after_holm': result.reject_null
            })
        output.append(row)
    return output
def _repeat_metrics(rows: Sequence[ParticipantPrediction], *, class_order: tuple[int, ...] = (0, 1, 2)) -> dict[str, float]:
    labels = np.asarray([row.label for row in rows], dtype=np.int64)
    probabilities = np.asarray([row.probabilities for row in rows], dtype=np.float64)
    if set(labels.tolist()) != set(class_order):
        raise ValueError('repeat does not contain every declared class')
    predicted = np.asarray(class_order, dtype=np.int64)[probabilities.argmax(axis=1)]
    return {
        'balanced_accuracy': float(balanced_accuracy_score(labels, predicted)),
        'macro_f1': float(f1_score(labels, predicted, labels=np.asarray(class_order), average='macro', zero_division=0)),
        'macro_roc_auc_ovr': float(
            np.mean([
                roc_auc_score(labels == class_label, probabilities[:, class_index]) for class_index, class_label in enumerate(class_order)
            ]))
    }
def paired_repeat_deltas_against_reference(
        prediction_rows: Sequence[Mapping[str, Any]], *, reference_case_id: str, comparison_family: str, comparison_role: str,
        candidate_case_ids: Sequence[str] | None = None, expected_repeats: Sequence[int] | None = None,
        expected_membership: Mapping[tuple[str, int], tuple[int, int]] | None = None) -> list[dict[str, Any]]:
    grouped: dict[str, dict[tuple[str, int], ParticipantPrediction]] = {}
    authority: dict[str, dict[tuple[str, int], tuple[int, int, int]]] = {}
    class_order_by_case: dict[str, tuple[int, ...]] = {}
    invalid: dict[str, str] = {}
    seen_case_ids: set[str] = set()
    for row in prediction_rows:
        case_id = str(row.get('classifier_id', row.get('case_id', '')))
        if not case_id:
            continue
        seen_case_ids.add(case_id)
        try:
            participant_id = str(row.get('participant_id', '')).strip()
            repeat = int(row.get('repeat', -1))
            fold = int(row.get('fold', -1))
            split_seed = int(row.get('split_seed', -1))
            label = int(row.get('true_label', row.get('label', -1)))
            probabilities = tuple((float(value) for value in row.get('probabilities', ())))
            raw_order = row.get('class_order')
            class_order = tuple(
                (int(value)
                 for value in raw_order)) if raw_order is not None and (not isinstance(raw_order,
                                                                                       (str, bytes))) else tuple(range(len(probabilities)))
        except (TypeError, ValueError):
            invalid[case_id] = 'invalid_prediction_contract'
            continue
        key = (participant_id, repeat)
        if not participant_id or repeat < 0 or fold < 0 or (split_seed < 0) or (len(class_order) < 2) or (len(
                set(class_order)) != len(class_order)) or (label not in class_order) or (len(probabilities) != len(class_order)) or (
                    row.get('retained', True) is False) or (key in grouped.setdefault(case_id, {})):
            invalid[case_id] = 'invalid_or_duplicate_participant_oof_contract'
            continue
        previous_order = class_order_by_case.setdefault(case_id, class_order)
        if previous_order != class_order:
            invalid[case_id] = 'inconsistent_class_order'
            continue
        try:
            grouped[case_id][key] = ParticipantPrediction(participant_id, label, repeat, probabilities)
        except ValueError:
            invalid[case_id] = 'invalid_probability_vector'
            continue
        authority.setdefault(case_id, {})[key] = (fold, split_seed, label)
    reference_id = str(reference_case_id)
    declared_candidates = {str(value)
                           for value in candidate_case_ids if str(value) and str(value) != reference_id
                           } if candidate_case_ids is not None else (seen_case_ids | set(grouped)) - {reference_id}

    def unavailable_row(candidate_id: str, reason: str, repeat: int | None) -> dict[str, Any]:
        row: dict[str, Any] = {
            'comparison_family': comparison_family, 'comparison_id': f'{candidate_id}_vs_{reference_id}',
            'comparison_role': comparison_role, 'reference_case_id': reference_id, 'candidate_case_id': candidate_id, 'repeat': repeat,
            'split_seed': None, 'matched_participant_count': None, 'matched_roster_sha256': None,
            'comparison_contract_status': f'N/A_{reason}', 'difference_direction': 'candidate_minus_reference', 'automatic_selection': False
        }
        for metric in ('balanced_accuracy', 'macro_f1', 'macro_roc_auc_ovr'):
            row[f'reference_{metric}'] = None
            row[f'candidate_{metric}'] = None
            row[f'{metric}_delta'] = None
        return row

    reference_reason = invalid.get(reference_id)
    if reference_reason is None and reference_id not in grouped:
        reference_reason = 'reference_has_no_valid_participant_oof'
    declared_repeats = sorted({int(value) for value in expected_repeats}) if expected_repeats is not None else None
    if reference_reason is not None:
        repeats = declared_repeats or [None]
        return [
            unavailable_row(candidate_id, f'reference_{reference_reason}', repeat) for candidate_id in sorted(declared_candidates)
            for repeat in repeats
        ]
    reference_authority = authority[reference_id]
    reference_by_key = grouped[reference_id]
    reference_class_order = class_order_by_case[reference_id]
    if expected_membership is not None:
        observed_membership = {key: (value[0], value[1]) for key, value in reference_authority.items()}
        normalized_expected = {(str(key[0]), int(key[1])): (int(value[0]), int(value[1])) for key, value in expected_membership.items()}
        if observed_membership != normalized_expected:
            repeats = declared_repeats or [None]
            return [
                unavailable_row(candidate_id, 'reference_frozen_split_registry_roster_mismatch', repeat)
                for candidate_id in sorted(declared_candidates) for repeat in repeats
            ]
    output: list[dict[str, Any]] = []
    for case_id in sorted(declared_candidates):
        comparison_id = f'{case_id}_vs_{reference_id}'
        contract_reason = invalid.get(case_id)
        if contract_reason is None and case_id not in grouped:
            contract_reason = 'candidate_has_no_valid_participant_oof'
        if contract_reason is None and authority.get(case_id) != reference_authority:
            contract_reason = 'participant_repeat_fold_split_seed_label_roster_mismatch'
        if contract_reason is None and class_order_by_case.get(case_id) != reference_class_order:
            contract_reason = 'candidate_reference_class_order_mismatch'
        if contract_reason is not None:
            output.extend((unavailable_row(case_id, contract_reason, repeat) for repeat in declared_repeats or [None]))
            continue
        candidate_by_key = grouped[case_id]
        observed_repeats = sorted({repeat for _participant, repeat in reference_by_key})
        repeats = declared_repeats or observed_repeats
        for repeat in repeats:
            keys = sorted((key for key in reference_by_key if key[1] == repeat))
            if not keys:
                output.append(unavailable_row(case_id, 'declared_repeat_missing_from_matched_oof', repeat))
                continue
            reference_rows = [reference_by_key[key] for key in keys]
            candidate_rows = [candidate_by_key[key] for key in keys]
            split_seeds = {reference_authority[key][1] for key in keys}
            if len(split_seeds) != 1:
                output.append(unavailable_row(case_id, 'repeat_has_multiple_split_seeds', repeat))
                continue
            try:
                reference_metrics = _repeat_metrics(reference_rows, class_order=reference_class_order)
                candidate_metrics = _repeat_metrics(candidate_rows, class_order=reference_class_order)
                if not all((math.isfinite(value) for value in (*reference_metrics.values(), *candidate_metrics.values()))):
                    raise ValueError('repeat metric is non-finite')
            except ValueError:
                unavailable = unavailable_row(case_id, 'repeat_lacks_all_classes_for_macro_roc_auc', repeat)
                unavailable['matched_participant_count'] = len(keys)
                unavailable['split_seed'] = next(iter(split_seeds))
                output.append(unavailable)
                continue
            roster_payload = [[key[0], repeat, *reference_authority[key]] for key in keys]
            roster_sha256 = hashlib.sha256(
                json.dumps(roster_payload, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()
            row: dict[str, Any] = {
                'comparison_family': comparison_family, 'comparison_id': comparison_id, 'comparison_role': comparison_role,
                'reference_case_id': reference_id, 'candidate_case_id': case_id, 'repeat': repeat,
                'split_seed': next(iter(split_seeds)) if len(split_seeds) == 1 else None, 'matched_participant_count': len(keys),
                'matched_roster_sha256': roster_sha256, 'comparison_contract_status': 'matched_complete_roster',
                'difference_direction': 'candidate_minus_reference', 'automatic_selection': False
            }
            for metric in ('balanced_accuracy', 'macro_f1', 'macro_roc_auc_ovr'):
                row[f'reference_{metric}'] = reference_metrics[metric]
                row[f'candidate_{metric}'] = candidate_metrics[metric]
                row[f'{metric}_delta'] = candidate_metrics[metric] - reference_metrics[metric]
            output.append(row)
    return output
def classification_conclusion_rows(
    comparison_rows: Sequence[Mapping[str, Any]], *, selected_case_id: str | None, selection_basis: str, study_role: str,
    planned_case_count: int | None = None, incomplete_case_count: int = 0,
    inference_reference_case_ids: Sequence[str] = ()) -> list[dict[str, Any]]:
    rows = list(comparison_rows)
    if not rows:
        return [{
            'angle': 'completion', 'leading_or_selected_case': None, 'finding': 'No comparable completed classifier result is available.',
            'confidence': 'not_established', 'confidence_scope': 'selection_superiority', 'selection_effect': 'none'
        }]
    point_top = str(rows[0]['case_id'])
    selected = str(selected_case_id) if selected_case_id else None
    complete = [row for row in rows if row.get('complete_for_requested_execution') is True]
    expected_cases = int(planned_case_count) if planned_case_count is not None else len(rows)
    all_complete = len(complete) == len(rows) == expected_cases and int(incomplete_case_count) == 0
    complete_requested_execution = bool(complete) and all(
        (int(row.get('repeat_count') or 0) >= 1 and int(row.get('fold_cell_count') or 0) >= int(row.get('repeat_count') or 0)
         for row in complete))
    cluster_ci_available = all((row.get('balanced_accuracy_participant_cluster_ci95_percent') != 'N/A'
                                and row.get('macro_f1_participant_cluster_ci95_percent') != 'N/A' and
                                (row.get('macro_roc_auc_ovr_participant_cluster_ci95_percent') != 'N/A')
                                for row in complete)) if complete else False
    reference_ids = {str(value) for value in inference_reference_case_ids if str(value).strip()}
    if selected:
        reference_ids.add(selected)
    p_rows = [
        row for row in rows
        if str(row.get('paired_reference_case_id')) in reference_ids and str(row.get('case_id')) != str(row.get('paired_reference_case_id'))
    ]
    tested_p_rows = [row for row in p_rows if _number(row.get('ba_raw_two_sided_p')) is not None]
    significant_better_challengers = [
        row for row in tested_p_rows
        if row.get('ba_holm_reject') is True and _number(row.get('ba_candidate_minus_reference')) is not None and (
            float(row['ba_candidate_minus_reference']) > 0.0)
    ]
    confirmatory_p = bool(tested_p_rows) and all(
        (str(row.get('inference_role', '')) == 'declared_reference_confirmatory' for row in tested_p_rows))
    f1_direction_conflicts = [
        row for row in tested_p_rows
        if _number(row.get('ba_candidate_minus_reference')) is not None and _number(row.get('f1_candidate_minus_reference')) is not None and
        (float(row['ba_candidate_minus_reference']) * float(row['f1_candidate_minus_reference']) < 0.0)
    ]
    if selected is None:
        superiority_confidence = 'not_established_no_selection'
    elif selected != point_top:
        superiority_confidence = 'low_metric_disagreement'
    elif significant_better_challengers:
        superiority_confidence = 'low_challenger_significantly_better'
    elif f1_direction_conflicts:
        superiority_confidence = 'low_ba_f1_direction_disagreement'
    elif not confirmatory_p:
        superiority_confidence = 'low_no_confirmatory_paired_superiority_test'
    else:
        superiority_confidence = 'low_no_adjusted_significant_difference_does_not_establish_superiority'
    evidence_integrity = 'evidence_completeness_high_with_cluster_ci_not_superiority' if all_complete and complete_requested_execution and cluster_ci_available else 'evidence_completeness_moderate_reduced_resource_or_ci' if all_complete else 'evidence_completeness_low_incomplete_execution'
    selected_text = selected or 'none'
    return [{
        'angle': 'point_estimates', 'leading_or_selected_case': point_top,
        'finding': f"Highest participant-OOF BA is {point_top}: {rows[0]['balanced_accuracy_mean_sd_percent']} percent; Macro-F1 {rows[0]['macro_f1_mean_sd_percent']} percent; macro ROC-AUC {rows[0]['macro_roc_auc_ovr_mean_sd_percent']} percent.",
        'confidence': 'descriptive', 'confidence_scope': 'point_estimate_only', 'selection_effect': 'none_by_itself'
    }, {
        'angle': 'uncertainty', 'leading_or_selected_case': point_top,
        'finding': 'Repeat t-CI and participant-cluster percentile CI are reported separately; marginal CI overlap is not used as a significance test.',
        'confidence': evidence_integrity, 'confidence_scope': 'evidence_completeness_not_superiority',
        'selection_effect': 'supports_precision_audit_only'
    }, {
        'angle': 'paired_inference', 'leading_or_selected_case': selected,
        'finding': f"{len(tested_p_rows)} candidate contrasts use {tested_p_rows[0].get('inference_role')} paired participant-cluster P values with metric-wise Holm correction."
        if tested_p_rows else 'No eligible paired P-value family is available; superiority is not established.',
        'confidence': 'confirmatory' if confirmatory_p else 'exploratory_or_unavailable',
        'confidence_scope': 'inference_design_not_superiority_probability', 'selection_effect': 'none_automatic'
    }, {
        'angle': 'robustness', 'leading_or_selected_case': point_top,
        'finding': f"Worst-fold BA={rows[0]['worst_fold_balanced_accuracy_percent']}%; worst-class F1={rows[0]['worst_class_f1_percent']}%. These stress metrics can disagree with mean BA ranking.",
        'confidence': evidence_integrity, 'confidence_scope': 'evidence_completeness_not_superiority',
        'selection_effect': 'secondary_review'
    }, {
        'angle': 'selection', 'leading_or_selected_case': selected,
        'finding': f"Persisted choice={selected_text} by {selection_basis}; participant-OOF point-estimate top={point_top}; agreement={(selected == point_top if selected else 'N/A')}. This is a {study_role} choice, not an independent final-test winner.",
        'confidence': superiority_confidence, 'confidence_scope': 'selection_superiority',
        'selection_effect': 'retain_persisted_choice_without_rewriting_history'
    }]
def _markdown_table(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return 'N/A — no rows.'
    fields = list(dict.fromkeys((str(key) for row in rows for key in row)))
    if len(fields) > _MAX_HUMAN_TABLE_COLUMNS:
        raise ValueError(f'human-facing result-interpretation table has {len(fields)} columns; maximum is {_MAX_HUMAN_TABLE_COLUMNS}')
    lines = ['| ' + ' | '.join(fields) + ' |', '|' + '|'.join(('---' for _ in fields)) + '|']
    for row in rows:
        lines.append('| ' + ' | '.join((str(row.get(field, '')).replace('|', '\\|').replace('\n', ' ') for field in fields)) + ' |')
    lines.extend(('', markdown_column_definitions_block(fields)))
    return '\n'.join(lines)
def write_result_interpretation(root: str | Path, *, comparison_rows: Sequence[Mapping[str, Any]], conclusion_rows: Sequence[Mapping[str,
                                                                                                                                     Any]],
                                paired_inference: Sequence[Mapping[str, Any]] = (), split_classification_comparison: bool = False,
                                title: str = 'Result interpretation and selection confidence') -> Path:
    comparison_sections: list[str]
    if split_classification_comparison:
        comparison_views = classification_comparison_table_views(comparison_rows, paired_inference=paired_inference)
        comparison_sections = [
            "The lossless compatibility evidence remains in the caller's machine-readable comparison table. The report uses separate narrow projections so ranking, uncertainty, inference, and robustness are not conflated.",
            '', '### Ranking and performance', '',
            _markdown_table(comparison_views['ranking_performance']), '', '### Uncertainty and 95% confidence intervals', '',
            _markdown_table(comparison_views['uncertainty_ci']), '', '### Paired inference', '',
            _markdown_table(comparison_views['paired_inference']), '', '### Robustness', '',
            _markdown_table(comparison_views['robustness'])
        ]
    else:
        comparison_sections = [_markdown_table(comparison_rows)]
    target = Path(root) / 'RESULT_INTERPRETATION.md'
    target.write_text(
        '\n'.join([
            f'# {title}', '',
            'P values below are null-hypothesis tail probabilities, not a posterior probability that a candidate is best. Repeat t-CIs and participant-cluster bootstrap CIs answer different uncertainty questions and are labeled separately.',
            'A participant-cluster CI resamples participant IDs with replacement within true-class strata and carries all repeat OOF predictions for each draw. Metrics are recomputed per repeat and averaged equally; the 95% interval is the 2.5th/97.5th percentile. Paired intervals apply the same participant draw to candidate and reference before forming candidate minus reference.',
            'These intervals quantify participant-sampling uncertainty conditional on the observed dataset, frozen folds, fitted predictions, and reporting estimand. They do not include dataset shift, model-selection uncertainty, or the probability that one classifier is superior.',
            '', '## Comprehensive comparison', '', *comparison_sections, '', '## Conclusions by evidence angle', '',
            _markdown_table(conclusion_rows), ''
        ]), encoding='utf-8')
    return target
__all__ = [
    'DEFAULT_PAIRED_BOOTSTRAP_RESAMPLES', 'DEFAULT_PAIRED_PERMUTATION_RESAMPLES', 'DEFAULT_REPORTING_RANDOM_SEED',
    'classification_comparison_rows', 'classification_comparison_table_views', 'classification_conclusion_rows',
    'holm_adjust_paired_inference_rows', 'paired_inference_against_reference', 'paired_repeat_deltas_against_reference',
    'write_result_interpretation'
]
