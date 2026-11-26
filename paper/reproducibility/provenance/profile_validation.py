#!/usr/bin/env python3
"""Study-independent structural checks for the supplied application profile.

These functions inspect an in-memory record. They do not verify file bytes,
interpret scientific prose, recompute numerical outcomes, or enforce the
Allen--Cahn campaign's run counts, seed list, or conclusions. Filesystem
integrity and frozen-study regression are separate validator layers.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COLLECTIONS = (
    "protocols", "configurations", "spatial_domains", "runs", "checkpoints",
    "predictions", "evaluations", "controller_traces", "metrics", "claims",
    "artifacts", "generations",
)
LINK_TARGETS = {
    "protocol_id": "protocols", "run_id": "runs", "config_id": "configurations",
    "configuration_id": "configurations", "checkpoint_id": "checkpoints",
    "prediction_id": "predictions", "prediction_ids": "predictions",
    "evaluation_ids": "evaluations", "controller_trace_ids": "controller_traces",
    "native_spatial_domain_id": "spatial_domains",
    "evaluation_spatial_domain_id": "spatial_domains", "metric_ids": "metrics",
    "generator_id": "artifacts", "source_activity_id": "generations",
    "input_ids": "application", "output_ids": "application",
    "supporting_evidence_ids": "application", "limiting_evidence_ids": "application",
    "prov_activity_id": "prov_activities", "source_record_id": "application",
}


def _check(name, errors, detail):
    return {"name": name, "status": "failed" if errors else "passed", "errors": errors, "detail": detail}


def _report(checks):
    errors = [error for check in checks for error in check["errors"]]
    return {"passed": not errors, "status": "failed" if errors else "passed", "checks": checks, "errors": errors}


def validate_links(audit: dict) -> dict:
    """Check identifiers and existing link endpoints, without requiring fields.

    This intentionally does not enforce scientific metadata requirements. A
    missing mechanism or missing disclosure field belongs to profile checking.
    """
    indices = {}
    errors = []
    application = {}
    for collection in COLLECTIONS:
        items = audit.get(collection, [])
        if not isinstance(items, list):
            errors.append(f"{collection}: expected record collection")
            items = []
        index = {}
        for position, item in enumerate(items):
            if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                errors.append(f"{collection}[{position}]: missing identifier")
                continue
            identifier = item["id"]
            if identifier in application:
                errors.append(f"duplicate application identifier: {identifier}")
            application[identifier] = item
            index[identifier] = item
        indices[collection] = index
    indices["application"] = application
    prov = audit.get("prov", {})
    seen_prov_ids = set()
    for collection in ("entities", "activities", "agents"):
        indices[f"prov_{collection}"] = {}
        for item in prov.get(collection, []):
            if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                errors.append(f"prov.{collection}: missing identifier")
                continue
            if item["id"] in seen_prov_ids:
                errors.append(f"duplicate PROV identifier: {item['id']}")
            seen_prov_ids.add(item["id"])
            indices[f"prov_{collection}"][item["id"]] = item
    prov_ids = set().union(*(set(indices[f"prov_{collection}"]) for collection in ("entities", "activities", "agents")))

    def visit(value, path):
        if isinstance(value, dict):
            for key, item in value.items():
                target = LINK_TARGETS.get(key)
                if key == "artifact_id" or key.endswith("_artifact_id") or key.endswith("_artifact_ids"):
                    target = "artifacts"
                if target:
                    references = item if isinstance(item, list) else [item]
                    for reference in references:
                        if reference is not None and (not isinstance(reference, str) or reference not in indices[target]):
                            errors.append(f"{path}.{key}: unresolved {target} reference {reference!r}")
                if key in {"subject_id", "object_id"} and item not in prov_ids:
                    errors.append(f"{path}.{key}: unresolved PROV endpoint {item!r}")
                visit(item, f"{path}.{key}")
        elif isinstance(value, list):
            for position, item in enumerate(value):
                visit(item, f"{path}[{position}]")

    for collection in COLLECTIONS:
        visit(audit.get(collection, []), collection)
    visit(prov, "prov")
    for kind in ("entities", "activities"):
        for item in prov.get(kind, []):
            if item.get("record_id") not in application:
                errors.append(f"{item.get('id')}: unresolved PROV application record {item.get('record_id')!r}")
    return _report([_check("identifier_links", errors, "Existing application references and PROV endpoints resolve; no required-field or outcome assertions.")])


def validate_profile(audit: dict, schema: dict | None = None) -> dict:
    """Validate schema, links and generic metadata consistency in memory.

    A valid record may describe positive, negative or uncertain results and
    any study size permitted by the application profile. Success means that
    the tested declarations agree, not that they are scientifically true.
    """
    # Imported lazily so the command-line validator can reuse this module.
    from validate_provenance import SchemaValidator

    schema = schema if schema is not None else json.loads((ROOT / "audit_record.schema.json").read_text(encoding="utf-8"))
    validator = SchemaValidator(schema)
    validator.check(audit, schema)
    checks = [_check("profile_schema", validator.errors, "Required profile fields, types and controlled vocabularies; no case-specific cardinalities.")]
    checks.extend(validate_links(audit)["checks"])
    if any(check["errors"] for check in checks):
        return _report(checks)
    idx = {name: {item["id"]: item for item in audit[name]} for name in COLLECTIONS}

    errors = []
    for run in audit["runs"]:
        protocol = idx["protocols"][run["protocol_id"]]
        config = idx["configurations"][run["config_id"]]
        checkpoint = idx["checkpoints"][run["checkpoint_id"]]
        if checkpoint["run_id"] != run["id"] or config["run_id"] != run["id"]:
            errors.append(f"{run['id']}: checkpoint/configuration belongs to another run")
        if run["campaign"] != protocol["campaign"] or config["campaign"] != run["campaign"]:
            errors.append(f"{run['id']}: campaign does not agree with protocol/configuration")
        if run["config_digest"] != config["digest"]:
            errors.append(f"{run['id']}: configuration digest differs")
        for collection, field in (("predictions", "prediction_ids"), ("evaluations", "evaluation_ids")):
            if any(idx[collection][identifier]["run_id"] != run["id"] for identifier in run[field]):
                errors.append(f"{run['id']}: {field} includes a different run")
    for trace in audit["controller_traces"]:
        run = idx["runs"][trace["run_id"]]
        if trace["protocol_id"] != run["protocol_id"] or trace["configuration_id"] != run["config_id"]:
            errors.append(f"{trace['id']}: trace protocol/configuration differs from run")
    checks.append(_check("run_configuration_consistency", errors, "Run, checkpoint, configuration and controller-trace declarations agree."))

    errors = []
    def valid_domain(domain, identifier):
        if domain["start"] > domain["end"]:
            errors.append(f"{identifier}: reversed temporal domain")
        if domain["closure"] == "point" and domain["start"] != domain["end"]:
            errors.append(f"{identifier}: point domain has unequal endpoints")
        if domain["closure"] == "left-open-right-closed" and domain["start"] == domain["end"]:
            errors.append(f"{identifier}: empty half-open domain")
    for prediction in audit["predictions"]:
        run = idx["runs"][prediction["run_id"]]
        checkpoint = idx["checkpoints"][prediction["checkpoint_id"]]
        if checkpoint["run_id"] != run["id"] or prediction["checkpoint_id"] != run["checkpoint_id"]:
            errors.append(f"{prediction['id']}: checkpoint does not belong to the declared run")
        if prediction["id"] not in run["prediction_ids"]:
            errors.append(f"{prediction['id']}: missing run backlink")
        for field in ("native_spatial_domain_id", "evaluation_spatial_domain_id"):
            if idx["spatial_domains"][prediction[field]]["campaign"] != run["campaign"]:
                errors.append(f"{prediction['id']}: {field} belongs to another campaign")
        for identifier in prediction["evaluation_ids"]:
            if idx["evaluations"][identifier]["prediction_id"] != prediction["id"]:
                errors.append(f"{prediction['id']}: evaluation backlink mismatch")
        valid_domain(prediction["temporal_domain"], prediction["id"])
    for evaluation in audit["evaluations"]:
        prediction = idx["predictions"][evaluation["prediction_id"]]
        if evaluation["run_id"] != prediction["run_id"] or evaluation["id"] not in prediction["evaluation_ids"]:
            errors.append(f"{evaluation['id']}: prediction/run backlink mismatch")
        if evaluation["id"] not in idx["runs"][evaluation["run_id"]]["evaluation_ids"]:
            errors.append(f"{evaluation['id']}: missing run backlink")
        interval, domain = evaluation["interval"], prediction["temporal_domain"]
        valid_domain(interval, evaluation["id"])
        if interval["start"] < domain["start"] or interval["end"] > domain["end"]:
            errors.append(f"{evaluation['id']}: evaluation interval extends outside prediction temporal domain")
        if interval["start"] == domain["start"] and domain["closure"] == "left-open-right-closed" and interval["closure"] != "left-open-right-closed":
            errors.append(f"{evaluation['id']}: evaluation includes an excluded prediction endpoint")
    checks.append(_check("prediction_evaluation_consistency", errors, "Prediction checkpoints, spatial domains, evaluation backlinks and interval containment agree."))

    errors = []
    acceptable_sources = {
        "locked_confirmatory_primary": {"locked_confirmatory_primary"},
        "locked_confirmatory_secondary": {"locked_confirmatory_primary", "locked_confirmatory_secondary"},
        "contextual_component_control": {"locked_confirmatory_primary", "locked_confirmatory_secondary", "contextual_component_control"},
        "development_only": {"locked_confirmatory_primary", "locked_confirmatory_secondary", "contextual_component_control", "development_only"},
        "descriptive_diagnostic": set(audit["inferential_status_vocabulary"]),
    }
    for claim in audit["claims"]:
        linked_metrics = set(claim["metric_ids"]) | (set(claim["supporting_evidence_ids"]) & set(idx["metrics"]))
        for identifier in sorted(linked_metrics):
            metric = idx["metrics"][identifier]
            if metric["inferential_status"] not in acceptable_sources[claim["inferential_status"]]:
                errors.append(f"{claim['id']}: {claim['inferential_status']} claim exceeds linked metric {identifier} status {metric['inferential_status']}")
        if not set(claim["metric_ids"]).issubset(set(claim["supporting_evidence_ids"]) | set(claim["limiting_evidence_ids"])):
            errors.append(f"{claim['id']}: declared metric lacks an explicit supporting or limiting evidence role")
    for metric in audit["metrics"]:
        for evaluation_id in metric["evaluation_ids"]:
            if idx["evaluations"][evaluation_id]["prediction_id"] not in metric["prediction_ids"]:
                errors.append(f"{metric['id']}: scored prediction missing from linked prediction list")
        if "ci95" in metric and metric["ci95"][0] > metric["ci95"][1]:
            errors.append(f"{metric['id']}: confidence interval endpoints are reversed")
        if "wins_lower_error" in metric and "n" in metric and metric["wins_lower_error"] > metric["n"]:
            errors.append(f"{metric['id']}: win count exceeds number of replicates")
    checks.append(_check("claim_evidence_consistency", errors, "Explicit evidence roles and inferential-status bounds agree; no outcome direction or prose truth is inferred."))

    errors = []
    for generation in audit["generations"]:
        if generation["generator_digest"] != idx["artifacts"][generation["generator_id"]]["sha256"]:
            errors.append(f"{generation['id']}: declared generator digest differs from artifact inventory")
    prov = audit["prov"]
    ids_by_type = {kind: {item["id"] for item in prov[kind]} for kind in ("entities", "activities", "agents")}
    signatures = {
        "prov:used": ("activities", "entities"),
        "prov:wasGeneratedBy": ("entities", "activities"),
        "prov:wasAssociatedWith": ("activities", "agents"),
        "prov:wasInformedBy": ("activities", "activities"),
        "prov:wasDerivedFrom": ("entities", "entities"),
    }
    for collection in COLLECTIONS:
        expected_class = "prov:Activity" if collection in {"runs", "generations"} else "prov:Entity"
        if prov["record_class_mapping"].get(collection) != expected_class:
            errors.append(f"{collection}: incorrect application-to-PROV class mapping")
        if collection == "generations":
            # Training generations reuse the run activity; other generations
            # have their own activity. The explicit activity link is required.
            activities = {item["id"]: item for item in prov["activities"]}
            for generation in audit[collection]:
                activity = activities[generation["prov_activity_id"]]
                if activity["record_id"] not in {generation["id"], generation.get("run_id")}:
                    errors.append(f"{generation['id']}: generating activity projects another application record")
            continue
        projected = prov["activities"] if expected_class == "prov:Activity" else prov["entities"]
        for item in audit[collection]:
            if sum(projection["record_id"] == item["id"] for projection in projected) != 1:
                errors.append(f"{item['id']}: expected exactly one {expected_class} projection")
    for relation in prov["relations"]:
        subject_type, object_type = signatures[relation["predicate"]]
        if relation["subject_id"] not in ids_by_type[subject_type] or relation["object_id"] not in ids_by_type[object_type]:
            errors.append(f"{relation['predicate']}: PROV endpoint types do not match {subject_type} -> {object_type}")
    checks.append(_check("generation_prov_consistency", errors, "Generator digests, application-to-PROV coverage and relation endpoint types agree; byte verification is separate."))
    return _report(checks)
