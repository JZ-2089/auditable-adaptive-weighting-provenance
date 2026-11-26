#!/usr/bin/env python3
"""Run the fixed small audit comparison without changing any evidence files.

The comparisons concern configured validation layers, not competing standards.
Only Python's standard library and the adjacent profile checker are required.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from profile_validation import validate_links, validate_profile


ROOT = Path(__file__).resolve().parent
SPEC_NAME = "audit_comparison_spec.json"
RESULT_NAME = "audit_comparison_results.json"
TABLE_NAME = "audit_comparison_table.tex"
ENTITY_LISTS = (
    "protocols", "configurations", "spatial_domains", "runs", "checkpoints",
    "predictions", "evaluations", "controller_traces", "metrics", "claims",
    "artifacts", "generations",
)


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def export_record(record: dict) -> bytes:
    return (json.dumps(record, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def load_artifacts(audit: dict) -> tuple[dict, dict]:
    """Read packaged artifacts once; preserve the original expected digests."""
    payloads, inventory = {}, {}
    for artifact in audit["artifacts"]:
        if artifact["path_scope"] != "packaged":
            continue
        path = (ROOT / artifact["path"]).resolve()
        # Restrict the test to the existing packaged evidence tree.
        if not path.is_relative_to(ROOT):
            raise ValueError(f"Packaged artifact outside package: {artifact['id']}")
        artifact_id = artifact["id"]
        payloads[artifact_id] = path.read_bytes()
        inventory[artifact_id] = {
            "sha256": artifact["sha256"], "bytes": artifact["bytes"],
            "path": artifact["path"],
        }
    return payloads, inventory


def inventory_errors(payloads: dict, inventory: dict) -> list[str]:
    errors = []
    for artifact_id, expected in inventory.items():
        payload = payloads.get(artifact_id)
        if payload is None:
            errors.append(f"Missing artifact: {artifact_id}")
            continue
        if len(payload) != expected["bytes"]:
            errors.append(f"Byte count mismatch: {artifact_id}")
        if digest(payload) != expected["sha256"]:
            errors.append(f"SHA-256 mismatch: {artifact_id}")
    return errors


def mutate(audit: dict, payloads: dict, case: dict) -> list[str]:
    """Apply exactly the operation recorded in the specification."""
    operation = case["operation"]
    if operation == "identity":
        return []
    if operation == "reverse_top_level_entity_lists":
        for collection in ENTITY_LISTS:
            audit[collection].reverse()
        return ["/" + name for name in ENTITY_LISTS]
    if operation == "delete_first_prediction_mechanism":
        del audit["predictions"][0]["mechanism"]
        return ["/predictions/0/mechanism"]
    if operation == "extend_first_evaluation_end_beyond_prediction_by_one":
        evaluation = audit["evaluations"][0]
        prediction = next(item for item in audit["predictions"] if item["id"] == evaluation["prediction_id"])
        evaluation["interval"]["end"] = prediction["temporal_domain"]["end"] + 1.0
        return ["/evaluations/0/interval/end"]
    if operation == "replace_first_prediction_checkpoint_with_missing_id":
        if any(item["id"] == "checkpoint:missing" for item in audit["checkpoints"]):
            raise ValueError("Broken-link fixture ID unexpectedly exists")
        audit["predictions"][0]["checkpoint_id"] = "checkpoint:missing"
        return ["/predictions/0/checkpoint_id"]
    if operation == "delete_first_claim_limiting_evidence_ids":
        del audit["claims"][0]["limiting_evidence_ids"]
        return ["/claims/0/limiting_evidence_ids"]
    if operation == "promote_first_claim_status_without_changing_linked_metrics":
        claim = audit["claims"][0]
        if claim["inferential_status"] != "locked_confirmatory_secondary":
            raise ValueError("Status fixture requires the original secondary claim")
        claim["inferential_status"] = "locked_confirmatory_primary"
        return ["/claims/0/inferential_status"]
    if operation == "append_whitespace_to_first_packaged_metrics_artifact":
        artifact = next(item for item in audit["artifacts"]
                        if item["id"].startswith("artifact:metrics:")
                        and item["path_scope"] == "packaged")
        payloads[artifact["id"]] += b" \n"
        return [artifact["id"] + " (bytes only)"]
    if operation == "replace_first_claim_statement_with_false_generality":
        audit["claims"][0]["statement"] = case["replacement"]
        return ["/claims/0/statement"]
    raise ValueError(f"Unknown specified operation: {operation}")


def layer_outcome(common_errors: list[str], report: dict | None = None) -> dict:
    errors = list(common_errors)
    if report is not None:
        errors.extend(report["errors"])
        if bool(report["passed"]) != (not report["errors"]):
            raise ValueError("Profile checker returned inconsistent passed/errors fields")
    return {"accepted": not errors, "errors": errors}


def run_comparison() -> dict:
    spec_bytes = (ROOT / SPEC_NAME).read_bytes()
    specification = json.loads(spec_bytes)
    source_bytes = (ROOT / specification["source_record"]).read_bytes()
    original = json.loads(source_bytes)
    payloads, inventory = load_artifacts(original)
    # A dirty source package would confound every comparison. Stop rather than
    # misreport package-maintenance errors as successful fault detection.
    original_errors = inventory_errors(payloads, inventory)
    if original_errors:
        raise ValueError("Source package integrity failed before comparison: " + "; ".join(original_errors))
    cases = []
    for definition in specification["cases"]:
        audit = copy.deepcopy(original)
        # Bytes objects are immutable, so this shallow dictionary copy isolates
        # byte replacement while avoiding copies of all unaffected artifacts.
        case_payloads = dict(payloads)
        changed = mutate(audit, case_payloads, definition)
        record_bytes = export_record(audit)
        case_inventory = dict(inventory)
        case_inventory["audit-record-export"] = {
            "sha256": digest(record_bytes), "bytes": len(record_bytes),
            "path": "in-memory/audit_record.json",
        }
        case_payloads["audit-record-export"] = record_bytes
        common_errors = inventory_errors(case_payloads, case_inventory)
        outcomes = {
            "inventory": layer_outcome(common_errors),
            "links": layer_outcome(common_errors, validate_links(audit)),
            "profile": layer_outcome(common_errors, validate_profile(audit)),
        }
        cases.append({
            "id": definition["id"], "kind": definition["kind"],
            "label": definition["label"], "changed_locations": changed,
            "record_export_sha256": digest(record_bytes),
            "record_checksum_refreshed": True,
            "artifact_checksum_retained": definition["kind"] == "integrity_fault",
            "outcomes": outcomes,
        })
    controls = [case for case in cases if case["kind"] == "valid_control"]
    faults = [case for case in cases if case["kind"] in {"structural_fault", "integrity_fault"}]
    probes = [case for case in cases if case["kind"] == "blind_spot_probe"]
    summary = {}
    for baseline in specification["baselines"]:
        key = baseline["id"]
        summary[key] = {
            "faults_detected": sum(not case["outcomes"][key]["accepted"] for case in faults),
            "faults_total": len(faults),
            "false_positives": sum(not case["outcomes"][key]["accepted"] for case in controls),
            "valid_controls_total": len(controls),
            "blind_spot_probes_rejected": sum(not case["outcomes"][key]["accepted"] for case in probes),
            "blind_spot_probes_total": len(probes),
        }
    return {
        "result_schema_version": "1.0.0",
        "record_id": original["record_id"],
        "specification_sha256": digest(spec_bytes),
        "source_record_sha256": digest(source_bytes),
        "runner_sha256": digest(Path(__file__).read_bytes()),
        "profile_checker_sha256": digest((ROOT / "profile_validation.py").read_bytes()),
        "schema_validator_sha256": digest((ROOT / "validate_provenance.py").read_bytes()),
        "schema_sha256": digest((ROOT / "audit_record.schema.json").read_bytes()),
        "packaged_artifacts_checked": len(inventory),
        "external_artifacts_excluded": sum(item["path_scope"] != "packaged" for item in original["artifacts"]),
        "baselines": specification["baselines"],
        "cases": cases,
        "summary": summary,
        "interpretation": specification["reporting"]["interpretation"],
    }


def make_table(result: dict) -> str:
    lines = [
        "% Generated by reproducibility/provenance/audit_comparison.py; do not edit.",
        r"\begin{table}[t]",
        r"\centering\small",
        r"\caption{Small audit comparison on copied records. All three layers use the same artifact inventory and hashes; the additional checks are identifier links and application-profile rules, respectively. D: rejected; A: accepted. The false-narrative probe is outside the six structural/integrity faults.}",
        r"\label{tab:audit-comparison}",
        r"\begin{tabular}{p{0.49\linewidth}ccc}",
        r"\toprule",
        r"Record or injected defect & Inventory & Links & Profile \\",
        r"\midrule",
    ]
    for case in result["cases"]:
        labels = ["A" if case["outcomes"][key]["accepted"] else "D" for key in ("inventory", "links", "profile")]
        lines.append(case["label"] + " & " + " & ".join(labels) + r" \\")
    lines.append(r"\midrule")
    values = [result["summary"][key] for key in ("inventory", "links", "profile")]
    lines.append("Structural/integrity faults detected & " + " & ".join(f"{value['faults_detected']}/{value['faults_total']}" for value in values) + r" \\")
    lines.append("Valid controls rejected & " + " & ".join(f"{value['false_positives']}/{value['valid_controls_total']}" for value in values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="Write the deterministic JSON report and TeX table")
    mode.add_argument("--check", action="store_true", help="Verify that saved JSON and TeX match recomputation")
    args = parser.parse_args()
    result = run_comparison()
    outputs = {
        RESULT_NAME: json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        TABLE_NAME: make_table(result),
    }
    if args.write:
        for filename, payload in outputs.items():
            (ROOT / filename).write_text(payload, encoding="utf-8")
    if args.check:
        different = [filename for filename, payload in outputs.items()
                     if not (ROOT / filename).exists()
                     or (ROOT / filename).read_text(encoding="utf-8") != payload]
        if different:
            parser.exit(1, "Audit comparison outputs need rebuilding: " + ", ".join(different) + "\n")
    print(json.dumps({"summary": result["summary"], "packaged_artifacts_checked": result["packaged_artifacts_checked"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
