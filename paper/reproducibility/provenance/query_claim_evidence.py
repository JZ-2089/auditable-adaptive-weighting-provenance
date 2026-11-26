#!/usr/bin/env python3
"""Dependency-free claim traversals and explicitly labelled competency answers.

Claims C1, C2, ... are scientific statements, not competency-question IDs.
Each claim query can contribute to several of CQ1--CQ4. CQ5 byte verification
and CQ6 schema/gap validation are delegated explicitly to validator layers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def index(items: list[dict]) -> dict[str, dict]:
    return {item["id"]: item for item in items}


def unique(values):
    return list(dict.fromkeys(values))


def trace_evidence(audit: dict, claim: dict, evidence_ids: list[str]) -> dict:
    metrics = index(audit["metrics"])
    evaluations = index(audit["evaluations"])
    predictions = index(audit["predictions"])
    controller_traces = index(audit["controller_traces"])
    runs = index(audit["runs"])
    configurations = index(audit["configurations"])
    protocols = index(audit["protocols"])

    metric_ids = [identifier for identifier in evidence_ids if identifier in metrics]
    evaluation_ids = [identifier for identifier in evidence_ids if identifier in evaluations]
    prediction_ids = [identifier for identifier in evidence_ids if identifier in predictions]
    controller_trace_ids = [identifier for identifier in evidence_ids if identifier in controller_traces]
    direct_protocol_ids = [identifier for identifier in evidence_ids if identifier in protocols]
    for metric_id in metric_ids:
        evaluation_ids.extend(metrics[metric_id]["evaluation_ids"])
        prediction_ids.extend(metrics[metric_id]["prediction_ids"])
        controller_trace_ids.extend(metrics[metric_id]["controller_trace_ids"])
    evaluation_ids = unique(evaluation_ids)
    for evaluation_id in evaluation_ids:
        prediction_ids.append(evaluations[evaluation_id]["prediction_id"])
    prediction_ids = unique(prediction_ids)
    controller_trace_ids = unique(controller_trace_ids)

    def path_record(*, evidence_id: str, metric_id: str | None, evaluation_id: str | None, prediction_id: str | None, controller_trace_id: str | None, run_id: str) -> dict:
        run = runs[run_id]
        config = configurations[run["config_id"]]
        return {
            "claim_id": claim["id"],
            "evidence_id": evidence_id,
            "metric_id": metric_id,
            "evaluation_id": evaluation_id,
            "prediction_id": prediction_id,
            "controller_trace_id": controller_trace_id,
            "native_spatial_domain_id": predictions[prediction_id]["native_spatial_domain_id"] if prediction_id else None,
            "evaluation_spatial_domain_id": predictions[prediction_id]["evaluation_spatial_domain_id"] if prediction_id else None,
            "run_id": run_id,
            "configuration_id": config["id"],
            "configuration_digest": config["digest"],
            "protocol_id": run["protocol_id"],
        }

    paths = []
    for metric_id in metric_ids:
        metric = metrics[metric_id]
        if metric["evaluation_ids"]:
            for evaluation_id in metric["evaluation_ids"]:
                evaluation = evaluations[evaluation_id]
                prediction_id = evaluation["prediction_id"]
                run_id = evaluation["run_id"]
                paths.append(path_record(evidence_id=metric_id, metric_id=metric_id, evaluation_id=evaluation_id, prediction_id=prediction_id, controller_trace_id=None, run_id=run_id))
        else:
            for prediction_id in metric["prediction_ids"]:
                run_id = predictions[prediction_id]["run_id"]
                paths.append(path_record(evidence_id=metric_id, metric_id=metric_id, evaluation_id=None, prediction_id=prediction_id, controller_trace_id=None, run_id=run_id))
        for trace_id in metric["controller_trace_ids"]:
            trace = controller_traces[trace_id]
            paths.append(path_record(evidence_id=metric_id, metric_id=metric_id, evaluation_id=None, prediction_id=None, controller_trace_id=trace_id, run_id=trace["run_id"]))
    for evaluation_id in [identifier for identifier in evidence_ids if identifier in evaluations]:
        evaluation = evaluations[evaluation_id]
        prediction_id = evaluation["prediction_id"]
        run_id = evaluation["run_id"]
        paths.append(path_record(evidence_id=evaluation_id, metric_id=None, evaluation_id=evaluation_id, prediction_id=prediction_id, controller_trace_id=None, run_id=run_id))
    for prediction_id in [identifier for identifier in evidence_ids if identifier in predictions]:
        run_id = predictions[prediction_id]["run_id"]
        paths.append(path_record(evidence_id=prediction_id, metric_id=None, evaluation_id=None, prediction_id=prediction_id, controller_trace_id=None, run_id=run_id))
    for trace_id in [identifier for identifier in evidence_ids if identifier in controller_traces]:
        trace = controller_traces[trace_id]
        paths.append(path_record(evidence_id=trace_id, metric_id=None, evaluation_id=None, prediction_id=None, controller_trace_id=trace_id, run_id=trace["run_id"]))

    path_keys = set()
    deduplicated_paths = []
    for path in paths:
        key = tuple(path.values())
        if key not in path_keys:
            path_keys.add(key)
            deduplicated_paths.append(path)
    run_ids = unique(path["run_id"] for path in deduplicated_paths)
    configuration_ids = unique(path["configuration_id"] for path in deduplicated_paths)
    protocol_ids = unique([path["protocol_id"] for path in deduplicated_paths] + direct_protocol_ids)
    return {
        "evidence_ids": evidence_ids,
        "metric_ids": metric_ids,
        "evaluation_ids": evaluation_ids,
        "prediction_ids": prediction_ids,
        "controller_trace_ids": controller_trace_ids,
        "run_ids": run_ids,
        "configuration_ids": configuration_ids,
        "configurations": [{"id": config_id, "digest": configurations[config_id]["digest"]} for config_id in configuration_ids],
        "direct_protocol_ids": direct_protocol_ids,
        "protocol_ids": protocol_ids,
        "protocols": [{"id": protocol_id, "campaign": protocols[protocol_id]["campaign"]} for protocol_id in protocol_ids],
        "paths": deduplicated_paths,
    }


def build_competency_results(audit: dict, queries: list[dict]) -> list[dict]:
    """Answer the actual questions without claiming unperformed validation."""
    runs = index(audit["runs"])
    predictions = index(audit["predictions"])
    evaluations = index(audit["evaluations"])
    path_runs = unique(path["run_id"] for query in queries for boundary in ("supporting", "limiting") for path in query[boundary]["paths"])
    prediction_ids = unique(identifier for query in queries for boundary in ("supporting", "limiting") for identifier in query[boundary]["prediction_ids"])
    evaluation_ids = unique(identifier for query in queries for boundary in ("supporting", "limiting") for identifier in query[boundary]["evaluation_ids"])
    return [
        {"competency_question_id": "CQ1", "question": "Which protocol, configuration, seed, run and checkpoint produced a scored prediction?",
         "execution": "record_query", "results": [{key: runs[identifier][key] for key in ("id", "protocol_id", "config_id", "config_digest", "seed", "arm", "checkpoint_id")} for identifier in path_runs]},
        {"competency_question_id": "CQ2", "question": "Which prediction mechanism and domain were evaluated?",
         "execution": "record_query", "results": [{key: predictions[identifier][key] for key in ("id", "mechanism", "temporal_domain", "native_spatial_domain_id", "evaluation_spatial_domain_id", "output_field", "status")} for identifier in prediction_ids]},
        {"competency_question_id": "CQ3", "question": "Which reference, interval, metric and aggregation rule generated the value?",
         "execution": "record_query", "results": [{key: evaluations[identifier][key] for key in ("id", "prediction_id", "reference_artifact_id", "interval", "metric", "aggregation", "value", "source_artifact_id", "source_pointer")} for identifier in evaluation_ids]},
        {"competency_question_id": "CQ4", "question": "What supports and limits each claim, and what inferential status applies?",
         "execution": "record_query", "results": [{"claim_id": query["claim_id"], "query_id": query["query_id"], "inferential_status": query["inferential_status"], "supporting_evidence_ids": query["supporting"]["evidence_ids"], "limiting_evidence_ids": query["limiting"]["evidence_ids"]} for query in queries]},
        {"competency_question_id": "CQ5", "question": "Do available bytes and generator derivations match the declared inventory?",
         "execution": "delegated_validation", "performed_by_query": False,
         "checks": ["artifact_hashes", "generation_records", "ro_crate"],
         "command": "python validate_provenance.py --verify-external",
         "qualification": "Querying declared hashes does not verify bytes or execute generators; the separate integrity and frozen-study report states what was checked."},
        {"competency_question_id": "CQ6", "question": "Which required fields or roles are absent or explicitly unavailable?",
         "execution": "declared_gaps_and_delegated_validation", "schema_validation_performed_by_query": False,
         "results": [{"run_id": identifier, "unavailable_prediction_roles": runs[identifier]["unavailable_prediction_roles"]} for identifier in path_runs if runs[identifier]["unavailable_prediction_roles"]],
         "checks": ["profile_schema"], "command": "python validate_provenance.py --mode profile --record RECORD.json",
         "qualification": "Explicit unavailable roles are reported here. Schema validation identifies absent required fields; the supplied legacy gap audit is a separate documented assessment."},
    ]


def build_query_results(audit: dict, claim_ids: list[str] | None = None) -> dict:
    claims = index(audit["claims"])
    selected = list(claims) if claim_ids is None else claim_ids
    queries = []
    for claim_id in selected:
        if claim_id not in claims:
            raise KeyError(f"unknown claim: {claim_id}")
        claim = claims[claim_id]
        supporting = trace_evidence(audit, claim, claim["supporting_evidence_ids"])
        limiting = trace_evidence(audit, claim, claim["limiting_evidence_ids"])
        covered = ["CQ1"]
        if supporting["prediction_ids"] or limiting["prediction_ids"]:
            covered.append("CQ2")
        if supporting["evaluation_ids"] or limiting["evaluation_ids"]:
            covered.append("CQ3")
        covered.append("CQ4")
        queries.append(
            {
                "query_id": f"claim-evidence:{claim_id}",
                "competency_question_ids": covered,
                "claim_id": claim_id,
                "claim_statement": claim["statement"],
                "direction": claim["direction"],
                "inferential_status": claim["inferential_status"],
                "supporting": supporting,
                "limiting": limiting,
            }
        )
    return {
        "schema_version": "2.0.0",
        "audit_record_id": audit["record_id"],
        "description": "Claim identifiers are independent of six competency questions; traversals answer CQ1--CQ4 and report the separate verification requirements of CQ5--CQ6.",
        "competency_results": build_competency_results(audit, queries),
        "queries": queries,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--claim", action="append", help="claim identifier to query; may be repeated")
    parser.add_argument("--write", action="store_true", help="write competency_query_results.json")
    args = parser.parse_args()
    audit = json.loads((ROOT / "allen_cahn_audit_record.json").read_text(encoding="utf-8"))
    result = build_query_results(audit, args.claim)
    payload = json.dumps(result, indent=2) + "\n"
    if args.write:
        (ROOT / "competency_query_results.json").write_text(payload, encoding="utf-8")
        print(f"wrote {len(result['queries'])} claim traversal(s), with separate CQ1--CQ6 coverage")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
