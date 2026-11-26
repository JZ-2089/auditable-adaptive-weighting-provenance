#!/usr/bin/env python3
"""Audit frozen Allen--Cahn evidence and generate the DIM paper artefacts.

The script reads only hash-registered copies under ``inputs/``.  It validates
the matched design before writing tables, figures, a compact evidence summary,
and hashes for every generated output.  Run it from any working directory.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "inputs"
OUTPUT = ROOT / "generated"
EXPECTED_CONFIRMATION_SEEDS = [5101, 5113, 5147, 5167, 5191, 5227, 5261, 5281, 5309, 5333]
EXPECTED_LONG_SEEDS = [6101, 6131, 6173]
ARMS = ["zero_rk4", "fixed_rate1", "grouped_rate1"]
# Wall-clock PDF dates would change output hashes on an otherwise identical
# rebuild. Keep version metadata, but omit dates from these derived figures.
PDF_METADATA = {"CreationDate": None, "ModDate": None}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(name: str):
    return json.loads((INPUT / name).read_text(encoding="utf-8"))


def fmt(value: float, digits: int = 5) -> str:
    if value == 0:
        return "0"
    if abs(value) < 1e-4:
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def audit_sources() -> dict:
    registry = json.loads((ROOT / "evidence_sources.json").read_text(encoding="utf-8"))
    checked = []
    for item in registry["sources"]:
        path = ROOT / item["copied_path"]
        actual = sha256(path)
        if actual != item["sha256"]:
            raise RuntimeError(f"hash mismatch for {path}: {actual} != {item['sha256']}")
        checked.append({"path": item["copied_path"], "sha256": actual, "status": "verified"})
    return {"count": len(checked), "files": checked}


def audit_confirmation(summary: dict, manifest: dict, lock: dict) -> dict:
    if not summary.get("confirmatory"):
        raise RuntimeError("confirmation summary is not marked confirmatory")
    if (summary.get("expected_runs"), summary.get("completed_runs")) != (30, 30):
        raise RuntimeError("confirmation campaign is incomplete")
    if manifest.get("status") != "confirmatory_locked":
        raise RuntimeError("confirmation manifest is not locked")
    if manifest.get("seeds") != EXPECTED_CONFIRMATION_SEEDS:
        raise RuntimeError("unexpected confirmation seeds")
    if sorted(manifest.get("arms", {})) != sorted(ARMS):
        raise RuntimeError("unexpected confirmation arms")
    if lock.get("selected_method") != "grouped_rate1":
        raise RuntimeError("the copied selection lock does not select grouped_rate1")

    rows = summary["rows"]
    observed = {(int(row["seed"]), row["arm"]) for row in rows}
    expected = {(seed, arm) for seed in EXPECTED_CONFIRMATION_SEEDS for arm in ARMS}
    if observed != expected or len(rows) != 30:
        raise RuntimeError("confirmation rows do not form the expected complete matched design")
    if any(row["completed_updates"] != 20000 for row in rows):
        raise RuntimeError("confirmation update budgets differ")
    if any(row["nonfinite_updates"] != 0 or row["clipped_gradient_updates"] != 0 for row in rows):
        raise RuntimeError("confirmation contains non-finite or clipped-gradient updates")
    for row in rows:
        expected_updates = 384 if row["arm"] == "grouped_rate1" else 0
        if row["adaptive_updates"] != expected_updates:
            raise RuntimeError("unexpected adaptive-update count")
    return {
        "status": "verified_complete",
        "runs": 30,
        "paired_seeds": 10,
        "completed_updates_per_run": 20000,
        "nonfinite_updates": 0,
        "clipped_gradient_updates": 0,
        "grouped_adaptive_updates_per_run": 384,
        "primary_registered_contrast": lock["primary_contrast"],
        "second_predeclared_contrast": lock["secondary_predeclared_contrast"],
        "grouped_vs_fixed_status": "secondary contrast stated in the frozen protocol",
    }


def audit_long_sequence(summary: dict, manifest: dict) -> dict:
    if not summary.get("development") or manifest.get("status") != "development_locked":
        raise RuntimeError("long-sequence evidence is not locked development evidence")
    if manifest.get("seeds") != EXPECTED_LONG_SEEDS:
        raise RuntimeError("unexpected long-sequence seeds")
    rows = summary["rows"]
    scenarios = ["blocks_04", "blocks_08", "blocks_12"]
    observed = {(row["scenario"], int(row["seed"]), row["arm"]) for row in rows}
    expected = {(scenario, seed, arm) for scenario in scenarios for seed in EXPECTED_LONG_SEEDS for arm in ARMS}
    if observed != expected or len(rows) != 27:
        raise RuntimeError("long-sequence rows do not form the expected complete design")
    if manifest["common"].get("explicit_causal_loss_weighting") is not False:
        raise RuntimeError("long-sequence causal-weighting status changed")
    return {
        "status": "verified_complete_development_only",
        "runs": 27,
        "paired_seeds": 3,
        "block_counts": [4, 8, 12],
        "explicit_causal_loss_weighting": False,
    }


def audit_controller_mechanism(evidence: dict) -> dict:
    if evidence.get("inferential_status") != "descriptive_diagnostic":
        raise RuntimeError("controller evidence is not marked descriptive_diagnostic")
    verification = evidence.get("trace_verification", {})
    required_true = (
        "all_required_trace_values_finite",
        "exact_update_schedule_match",
        "json_npz_nan_layout_match",
    )
    if any(verification.get(key) is not True for key in required_true):
        raise RuntimeError("controller trace verification failed")
    if verification.get("seeds_verified") != 10 or verification.get("total_updates_verified") != 3840:
        raise RuntimeError("controller trace count differs from the complete campaign")
    if verification.get("maximum_json_npz_absolute_discrepancy") != 0.0:
        raise RuntimeError("controller JSON and NPZ traces disagree")
    if verification.get("maximum_weighted_ratio_reconstruction_discrepancy") != 0.0:
        raise RuntimeError("controller weighted-ratio reconstruction failed")
    return {
        "inferential_status": evidence["inferential_status"],
        "seeds_verified": verification["seeds_verified"],
        "adaptive_updates_verified": verification["total_updates_verified"],
        "all_required_trace_values_finite": True,
        "maximum_json_npz_absolute_discrepancy": 0.0,
        "maximum_physics_budget_error": verification["maximum_physics_budget_error"],
        "maximum_constraint_budget_error": verification["maximum_constraint_budget_error"],
    }


def write_confirmation_table(summary: dict) -> None:
    contrast_key = "grouped_rate1_minus_fixed_rate1"
    rows = [
        ("Training interval", "direct_training_relative_l2"),
        ("Validation interval", "direct_validation_relative_l2"),
        ("Direct test interval", "direct_test_relative_l2"),
        ("Continuation test interval", "continuation_test_relative_l2"),
        ("Terminal state", "terminal_relative_l2"),
    ]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering\scriptsize",
        r"\caption{Fresh-seed ten-pair Allen--Cahn comparison of grouped and fixed RK4 weighting. Differences are grouped minus fixed; negative values favour grouped weighting. The 95\% interval is the two-sided paired $t$ interval. This controller contrast is secondary under the frozen protocol.}",
        r"\label{tab:ac-confirmatory}",
        r"\begin{adjustbox}{max width=\linewidth}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Metric & Fixed mean & Grouped mean & Paired difference & 95\% CI & Wins/10\\",
        r"\midrule",
    ]
    for label, metric in rows:
        c = summary["paired_contrasts"][metric][contrast_key]
        lo, hi = c["mean_difference_95pct_t_ci"]
        lines.append(
            f"{label} & {fmt(c['control_mean'])} & {fmt(c['treatment_mean'])} & "
            f"{fmt(c['paired_mean_difference'])} & [{fmt(lo)}, {fmt(hi)}] & {c['wins_lower_error']}/10\\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}", ""])
    (OUTPUT / "table_confirmatory_grouped_vs_fixed.tex").write_text("\n".join(lines), encoding="utf-8")


def write_long_table(summary: dict) -> None:
    rows = []
    for blocks in (4, 8, 12):
        key = f"blocks_{blocks:02d}"
        contrast = summary["paired_contrasts"][key]["grouped_rate1_minus_fixed_rate1"]
        entries = []
        for metric in (
            "direct_training_relative_l2",
            "interface_mean_relative_l2",
            "direct_future_0.8_relative_l2",
            "continuation_future_0.8_relative_l2",
        ):
            item = contrast[metric]
            entries.extend([fmt(item["paired_mean_difference"]), f"{item['wins_lower_error']}/3"])
        rows.append((blocks, entries))
    lines = [
        r"\begin{table}[htbp]",
        r"\centering\scriptsize",
        r"\caption{Three-seed Allen--Cahn long-sequence development study. Entries are grouped-minus-fixed mean differences with the number of seeds favouring grouped weighting in parentheses. Negative differences indicate smaller error. These results are exploratory.}",
        r"\label{tab:ac-long-sequence}",
        r"\begin{adjustbox}{max width=\linewidth}",
        r"\begin{tabular}{ccccc}",
        r"\toprule",
        r"Blocks & Training & Interface mean & Direct future $0.8$ & Continuation $0.8$\\",
        r"\midrule",
    ]
    for blocks, e in rows:
        lines.append(
            f"{blocks} & {e[0]} ({e[1]}) & {e[2]} ({e[3]}) & {e[4]} ({e[5]}) & {e[6]} ({e[7]})\\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}", ""])
    (OUTPUT / "table_long_sequence.tex").write_text("\n".join(lines), encoding="utf-8")


def confirmation_values(summary: dict, metric: str):
    by = {(int(row["seed"]), row["arm"]): row[metric] for row in summary["rows"]}
    fixed = np.array([by[(seed, "fixed_rate1")] for seed in EXPECTED_CONFIRMATION_SEEDS])
    grouped = np.array([by[(seed, "grouped_rate1")] for seed in EXPECTED_CONFIRMATION_SEEDS])
    return fixed, grouped


def plot_confirmation(summary: dict) -> None:
    panels = [
        ("direct_training_relative_l2", "Training interval"),
        ("direct_validation_relative_l2", "Validation interval"),
        ("direct_test_relative_l2", "Direct test interval"),
        ("continuation_test_relative_l2", "Continuation test interval"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.5), constrained_layout=True)
    for ax, (metric, title) in zip(axes.flat, panels):
        fixed, grouped = confirmation_values(summary, metric)
        for a, b in zip(fixed, grouped):
            ax.plot([0, 1], [a, b], color="0.72", lw=0.9, zorder=1)
        ax.scatter(np.zeros_like(fixed), fixed, s=24, color="#2962a3", label="Fixed", zorder=2)
        ax.scatter(np.ones_like(grouped), grouped, s=24, color="#d55e00", label="Grouped", zorder=2)
        ax.plot([0, 1], [fixed.mean(), grouped.mean()], color="black", lw=2.2, marker="D", ms=4.5, zorder=3)
        ax.set_xticks([0, 1], ["Fixed", "Grouped"])
        ax.set_xlim(-0.25, 1.25)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Relative $L^2$ error")
        ax.grid(axis="y", alpha=0.22)
    fig.suptitle("Matched Allen--Cahn controller comparison (10 fresh seeds)", fontsize=11)
    fig.savefig(OUTPUT / "figure_confirmatory_paired.pdf", bbox_inches="tight", metadata=PDF_METADATA)
    fig.savefig(OUTPUT / "figure_confirmatory_paired.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_long_sequence(summary: dict) -> None:
    metrics = [
        ("direct_training_relative_l2", "Training interval"),
        ("interface_mean_relative_l2", "Mean interface error"),
        ("direct_future_0.8_relative_l2", "Direct future horizon 0.8"),
        ("continuation_future_0.8_relative_l2", "Continuation horizon 0.8"),
    ]
    block_counts = [4, 8, 12]
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.5), constrained_layout=True)
    for ax, (metric, title) in zip(axes.flat, metrics):
        seed_series = []
        means = []
        for blocks in block_counts:
            item = summary["paired_contrasts"][f"blocks_{blocks:02d}"]["grouped_rate1_minus_fixed_rate1"][metric]
            seed_series.append(np.asarray(item["paired_differences"], dtype=float))
            means.append(float(item["paired_mean_difference"]))
        data = np.vstack(seed_series)
        for seed_index in range(data.shape[1]):
            ax.plot(block_counts, data[:, seed_index], color="0.68", lw=1.0, marker="o", ms=3)
        ax.plot(block_counts, means, color="#7b3294", lw=2.2, marker="D", ms=4.5, label="Mean")
        ax.axhline(0.0, color="black", lw=0.8, ls="--")
        ax.set_xticks(block_counts)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Temporal blocks")
        ax.set_ylabel("Grouped $-$ fixed")
        ax.grid(axis="y", alpha=0.22)
    fig.suptitle("Allen--Cahn long-sequence development study (3 seeds)", fontsize=11)
    fig.savefig(OUTPUT / "figure_long_sequence_effects.pdf", bbox_inches="tight", metadata=PDF_METADATA)
    fig.savefig(OUTPUT / "figure_long_sequence_effects.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_evidence_summary(confirmation: dict, long_sequence: dict, controller: dict, audits: dict) -> None:
    metrics = [
        "direct_training_relative_l2",
        "direct_validation_relative_l2",
        "direct_test_relative_l2",
        "continuation_test_relative_l2",
        "terminal_relative_l2",
    ]
    matched = {
        metric: confirmation["paired_contrasts"][metric]["grouped_rate1_minus_fixed_rate1"]
        for metric in metrics
    }
    long_matched = {
        scenario: long_sequence["paired_contrasts"][scenario]["grouped_rate1_minus_fixed_rate1"]
        for scenario in ("blocks_04", "blocks_08", "blocks_12")
    }
    compact = {
        "schema_version": 1,
        "scope": "DIM adaptive-weighting manuscript evidence",
        "interpretation": {
            "ten_seed_grouped_vs_fixed": "secondary matched contrast in a locked independent campaign",
            "long_sequence": "three-seed development evidence; not confirmatory",
            "fixed_vs_zero": "outside the DIM claim graph; the current companion addresses the numerical question with a distinct cohort",
        },
        "audits": audits,
        "ten_seed_grouped_minus_fixed": matched,
        "long_sequence_grouped_minus_fixed": long_matched,
        "controller_descriptive_diagnostic": {
            "diagnostic_scope": controller["diagnostic_scope"],
            "inferential_status": controller["inferential_status"],
            "pooled_summary": controller["pooled_summary"],
            "by_block": controller["by_block"],
            "qualification": controller["qualification"],
        },
    }
    (OUTPUT / "dim_evidence_summary.json").write_text(json.dumps(compact, indent=2) + "\n", encoding="utf-8")


def write_output_hashes() -> None:
    files = sorted(path for path in OUTPUT.iterdir() if path.is_file() and path.name != "generated_outputs.sha256")
    text = "".join(f"{sha256(path)}  generated/{path.name}\n" for path in files)
    (OUTPUT / "generated_outputs.sha256").write_text(text, encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source_audit = audit_sources()
    confirmation = load_json("confirmation_summary.json")
    confirmation_manifest = load_json("confirmation_manifest.json")
    selection_lock = load_json("selection_lock.json")
    long_sequence = load_json("long_sequence_summary.json")
    long_manifest = load_json("long_sequence_manifest.json")
    controller = load_json("controller_mechanism_evidence.json")
    audits = {
        "source_hashes": source_audit,
        "confirmation": audit_confirmation(confirmation, confirmation_manifest, selection_lock),
        "long_sequence": audit_long_sequence(long_sequence, long_manifest),
        "controller_mechanism": audit_controller_mechanism(controller),
    }
    write_confirmation_table(confirmation)
    write_long_table(long_sequence)
    plot_confirmation(confirmation)
    plot_long_sequence(long_sequence)
    write_evidence_summary(confirmation, long_sequence, controller, audits)
    shutil.copy2(INPUT / "source_table_controller_mechanism.tex", OUTPUT / "table_controller_mechanism.tex")
    shutil.copy2(INPUT / "source_figure_controller_mechanism.pdf", OUTPUT / "figure_controller_mechanism.pdf")
    shutil.copy2(INPUT / "source_figure_controller_mechanism.png", OUTPUT / "figure_controller_mechanism.png")
    write_output_hashes()
    print(json.dumps(audits, indent=2))


if __name__ == "__main__":
    main()
