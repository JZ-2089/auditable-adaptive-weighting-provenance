#!/usr/bin/env python3
"""Create the locked paired analysis after every confirmation run is evaluated."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from scipy import stats

from run_confirmation import ARMS, ROOT, SEEDS, atomic_json, now


OUTCOMES = (
    'direct_training_relative_l2', 'direct_validation_relative_l2',
    'direct_test_relative_l2', 'continuation_validation_relative_l2',
    'continuation_test_relative_l2', 'terminal_relative_l2',
)


def exact_sign_flip_p(differences: np.ndarray) -> float:
    """One-sided paired randomisation p-value for a negative mean difference."""
    observed = float(np.mean(differences))
    means = [float(np.mean(np.asarray(signs) * differences))
             for signs in itertools.product((-1.0, 1.0), repeat=differences.size)]
    return float(np.mean(np.asarray(means) <= observed + 1e-15))


def paired_summary(treatment: np.ndarray, control: np.ndarray, seed: int) -> dict:
    differences = treatment - control
    n = differences.size
    mean = float(np.mean(differences))
    sem = float(stats.sem(differences))
    radius = float(stats.t.ppf(0.975, n - 1) * sem)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(100000, n))
    bootstrap = np.mean(differences[indices], axis=1)
    return {
        'n': int(n), 'treatment_mean': float(np.mean(treatment)),
        'control_mean': float(np.mean(control)), 'paired_mean_difference': mean,
        'paired_median_difference': float(np.median(differences)),
        'mean_difference_95pct_t_ci': [mean - radius, mean + radius],
        'paired_bootstrap_95pct_ci': [float(np.quantile(bootstrap, .025)),
                                      float(np.quantile(bootstrap, .975))],
        'wins_lower_error': int(np.sum(differences < 0)),
        'ties': int(np.sum(differences == 0)),
        'one_sided_exact_sign_flip_p': exact_sign_flip_p(differences),
        'relative_mean_change_percent': float(100 * mean / np.mean(control)),
        'paired_differences': differences.tolist(),
    }


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues, key=pvalues.get)
    adjusted, running = {}, 0.0
    count = len(ordered)
    for rank, name in enumerate(ordered):
        running = max(running, (count - rank) * pvalues[name])
        adjusted[name] = min(1.0, running)
    return adjusted


def main() -> None:
    rows = []
    for seed in SEEDS:
        for arm in ARMS:
            run = ROOT / 'runs' / f'seed_{seed}' / arm
            record = json.loads((run / 'record.json').read_text())
            metrics = json.loads((run / 'metrics.json').read_text())
            diagnostics = json.loads((run / 'output/diagnostics.json').read_text())
            if record.get('status') != 'complete' or metrics.get('phase') != 'confirmation':
                raise RuntimeError(f'Incomplete confirmation record: seed={seed}, arm={arm}')
            row = {key: metrics[key] for key in OUTCOMES}
            row.update(
                seed=seed, arm=arm, training_seconds=record['training_seconds'],
                nonfinite_updates=diagnostics.get('nonfinite_updates', 0),
                clipped_gradient_updates=diagnostics.get('clipped_gradient_updates', 0),
                completed_updates=diagnostics.get('completed_updates', 0),
                adaptive_updates=len(diagnostics.get('adaptive_updates', [])),
            )
            rows.append(row)
    values = {(row['seed'], row['arm']): row for row in rows}
    summary = {
        'schema_version': 1, 'created_utc': now(), 'confirmatory': True,
        'expected_runs': len(SEEDS) * len(ARMS), 'completed_runs': len(rows),
        'primary_outcome': 'direct_test_relative_l2', 'rows': rows,
        'by_arm': {}, 'paired_contrasts': {},
    }
    for arm in ARMS:
        selected = [values[(seed, arm)] for seed in SEEDS]
        summary['by_arm'][arm] = {
            key: {'mean': float(np.mean([row[key] for row in selected])),
                  'median': float(np.median([row[key] for row in selected])),
                  'standard_deviation': float(np.std([row[key] for row in selected], ddof=1))}
            for key in OUTCOMES
        }
    contrasts = (('grouped_rate1', 'zero_rk4'), ('fixed_rate1', 'zero_rk4'),
                 ('grouped_rate1', 'fixed_rate1'))
    for outcome_index, outcome in enumerate(OUTCOMES):
        summary['paired_contrasts'][outcome] = {}
        for contrast_index, (treatment, control) in enumerate(contrasts):
            treatment_values = np.asarray([values[(seed, treatment)][outcome] for seed in SEEDS])
            control_values = np.asarray([values[(seed, control)][outcome] for seed in SEEDS])
            name = f'{treatment}_minus_{control}'
            summary['paired_contrasts'][outcome][name] = paired_summary(
                treatment_values, control_values, 90210 + 100 * outcome_index + contrast_index)
    direct = summary['paired_contrasts']['direct_test_relative_l2']
    declared = ('grouped_rate1_minus_zero_rk4', 'fixed_rate1_minus_zero_rk4')
    adjusted = holm({name: direct[name]['one_sided_exact_sign_flip_p'] for name in declared})
    for name in declared:
        direct[name]['holm_adjusted_one_sided_p'] = adjusted[name]
    atomic_json(ROOT / 'confirmation_summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

