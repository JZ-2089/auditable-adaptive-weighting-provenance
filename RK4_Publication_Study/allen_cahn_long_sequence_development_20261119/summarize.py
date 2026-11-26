#!/usr/bin/env python3
"""Summarize the paired three-seed long-sequence development study."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from run_study import ROOT, SEEDS, SCENARIOS, ARMS, atomic_json


OUTCOMES = (
    "direct_training_relative_l2", "direct_terminal_relative_l2",
    "interface_mean_relative_l2", "interface_max_relative_l2",
    "direct_future_0.2_relative_l2", "direct_future_0.4_relative_l2",
    "direct_future_0.8_relative_l2", "continuation_future_0.8_relative_l2",
)


def main() -> None:
    rows, values = [], {}
    for scenario in SCENARIOS:
        for seed in SEEDS:
            for arm in ARMS:
                run = ROOT / "runs" / scenario / f"seed_{seed}" / arm
                record = json.loads((run / "record.json").read_text())
                if record.get("status") != "complete":
                    raise RuntimeError(f"Incomplete run: {scenario}, {seed}, {arm}")
                metrics = json.loads((run / "metrics.json").read_text())
                row = {"scenario": scenario, "seed": seed, "arm": arm,
                       **{name: metrics[name] for name in OUTCOMES}}
                rows.append(row)
                values[(scenario, seed, arm)] = row
    summary = {"schema_version": 1, "development": True, "rows": rows,
               "by_scenario": {}, "paired_contrasts": {}}
    for scenario in SCENARIOS:
        summary["by_scenario"][scenario] = {}
        summary["paired_contrasts"][scenario] = {}
        for arm in ARMS:
            selected = [values[(scenario, seed, arm)] for seed in SEEDS]
            summary["by_scenario"][scenario][arm] = {
                name: {"mean": float(np.mean([row[name] for row in selected])),
                       "values": [row[name] for row in selected]}
                for name in OUTCOMES
            }
        for treatment, control in (("grouped_rate1", "fixed_rate1"),
                                   ("fixed_rate1", "zero_rk4"),
                                   ("grouped_rate1", "zero_rk4")):
            label = f"{treatment}_minus_{control}"
            summary["paired_contrasts"][scenario][label] = {}
            for name in OUTCOMES:
                differences = np.asarray([
                    values[(scenario, seed, treatment)][name]
                    - values[(scenario, seed, control)][name] for seed in SEEDS
                ])
                summary["paired_contrasts"][scenario][label][name] = {
                    "paired_mean_difference": float(np.mean(differences)),
                    "paired_differences": differences.tolist(),
                    "wins_lower_error": int(np.sum(differences < 0)),
                }
    primary = "direct_training_relative_l2"
    adaptive_differences = [
        summary["paired_contrasts"][scenario]["grouped_rate1_minus_fixed_rate1"]
        [primary]["paired_mean_difference"] for scenario in SCENARIOS
    ]
    block_counts = np.asarray([SCENARIOS[name]["blocks"] for name in SCENARIOS], dtype=float)
    summary["adaptive_minus_fixed_training_error_trend"] = {
        "blocks": block_counts.tolist(), "mean_differences": adaptive_differences,
        "linear_slope_per_block": float(np.polyfit(block_counts, adaptive_differences, 1)[0]),
        "qualification": "Three-seed development trend; not confirmatory inference.",
    }
    atomic_json(ROOT / "analysis/summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
