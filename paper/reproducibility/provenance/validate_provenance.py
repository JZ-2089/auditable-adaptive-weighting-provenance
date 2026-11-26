#!/usr/bin/env python3
"""Validate the DIM provenance bundle without third-party dependencies.

Default study validation is portable and never overwrites validation_report.json.
The --mode profile path checks in-memory schema and semantic consistency only;
it deliberately omits file integrity and frozen Allen--Cahn regression.
The default --mode study retains all integrity and case regression checks.
Use ``--verify-external --write-report`` only in the source workspace to create
the formal report that records both packaged and external hash verification.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import math
import re
import struct
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def npz_vector_summary(path: Path, member: str) -> dict:
    """Read one uniform floating-point vector from an NPZ without NumPy."""
    with zipfile.ZipFile(path) as archive:
        payload = archive.read(f"{member}.npy")
    if payload[:6] != b"\x93NUMPY":
        raise AssertionError(f"invalid NPY member: {path}:{member}")
    major = payload[6]
    if major == 1:
        header_length, header_start = struct.unpack("<H", payload[8:10])[0], 10
    elif major in {2, 3}:
        header_length, header_start = struct.unpack("<I", payload[8:12])[0], 12
    else:
        raise AssertionError(f"unsupported NPY version: {path}:{member}")
    header = ast.literal_eval(payload[header_start:header_start + header_length].decode("latin1").strip())
    shape = tuple(header["shape"])
    formats = {"<f8": "<d", ">f8": ">d", "<f4": "<f", ">f4": ">f", "|f8": "=d", "|f4": "=f"}
    if len(shape) != 1 or header.get("fortran_order") or header["descr"] not in formats:
        raise AssertionError(f"unsupported spatial vector: {path}:{member}")
    values = [item[0] for item in struct.iter_unpack(formats[header["descr"]], payload[header_start + header_length:])]
    spacing = float(values[1] - values[0])
    if any(not math.isclose(values[index] - values[index - 1], spacing, rel_tol=1e-11, abs_tol=1e-13) for index in range(2, len(values))):
        raise AssertionError(f"non-uniform spatial vector: {path}:{member}")
    return {
        "lower_bound": float(values[0]),
        "last_coordinate": float(values[-1]),
        "upper_bound_exclusive": float(values[0] + spacing * len(values)),
        "points": len(values),
        "spacing": spacing,
        "period": float(spacing * len(values)),
        "endpoint_duplicate_excluded": True,
    }


def json_pointer(value, pointer: str):
    current = value
    for token in pointer.lstrip("/").split("/") if pointer != "/" else []:
        token = token.replace("~1", "/").replace("~0", "~")
        current = current[int(token)] if isinstance(current, list) else current[token]
    return current


class SchemaValidator:
    """Draft 2020-12 subset covering every keyword used by this package."""

    def __init__(self, root_schema: dict):
        self.root = root_schema
        self.errors: list[str] = []

    def resolve(self, reference: str):
        if not reference.startswith("#/"):
            self.errors.append(f"unsupported non-local $ref: {reference}")
            return {}
        return json_pointer(self.root, reference[1:])

    @staticmethod
    def type_ok(value, expected: str) -> bool:
        return {
            "object": isinstance(value, dict),
            "array": isinstance(value, list),
            "string": isinstance(value, str),
            "integer": isinstance(value, int) and not isinstance(value, bool),
            "number": isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value),
            "null": value is None,
            "boolean": isinstance(value, bool),
        }.get(expected, False)

    def check(self, value, schema: dict, path: str = "$"):
        if "$ref" in schema:
            self.check(value, self.resolve(schema["$ref"]), path)
            return
        if "oneOf" in schema:
            outcomes = []
            for candidate in schema["oneOf"]:
                nested = SchemaValidator(self.root)
                nested.check(value, candidate, path)
                outcomes.append(not nested.errors)
            if sum(outcomes) != 1:
                self.errors.append(f"{path}: oneOf matched {sum(outcomes)} branches")
            return
        if "not" in schema:
            nested = SchemaValidator(self.root)
            nested.check(value, schema["not"], path)
            if not nested.errors:
                self.errors.append(f"{path}: matched forbidden schema")
        for candidate in schema.get("allOf", []):
            self.check(value, candidate, path)
        if "if" in schema:
            condition = SchemaValidator(self.root)
            condition.check(value, schema["if"], path)
            branch = schema.get("then") if not condition.errors else schema.get("else")
            if branch:
                self.check(value, branch, path)
        expected = schema.get("type")
        if expected and not self.type_ok(value, expected):
            self.errors.append(f"{path}: expected {expected}, got {type(value).__name__}")
            return
        if "const" in schema and value != schema["const"]:
            self.errors.append(f"{path}: value differs from const")
        if "enum" in schema and value not in schema["enum"]:
            self.errors.append(f"{path}: value not in enum")
        if isinstance(value, str):
            if len(value) < schema.get("minLength", 0):
                self.errors.append(f"{path}: string shorter than minLength")
            if "pattern" in schema and re.search(schema["pattern"], value) is None:
                self.errors.append(f"{path}: string does not match pattern {schema['pattern']}")
            if schema.get("format") == "date-time":
                try:
                    datetime.fromisoformat(value.replace("Z", "+00:00"))
                except ValueError:
                    self.errors.append(f"{path}: invalid date-time")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if "minimum" in schema and value < schema["minimum"]:
                self.errors.append(f"{path}: below minimum")
            if "maximum" in schema and value > schema["maximum"]:
                self.errors.append(f"{path}: above maximum")
            if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
                self.errors.append(f"{path}: not above exclusiveMinimum")
        if isinstance(value, list):
            if len(value) < schema.get("minItems", 0):
                self.errors.append(f"{path}: fewer than minItems")
            if "maxItems" in schema and len(value) > schema["maxItems"]:
                self.errors.append(f"{path}: more than maxItems")
            if schema.get("uniqueItems"):
                encoded = [json.dumps(item, sort_keys=True) for item in value]
                if len(encoded) != len(set(encoded)):
                    self.errors.append(f"{path}: duplicate array items")
            if isinstance(schema.get("items"), dict):
                for index, item in enumerate(value):
                    self.check(item, schema["items"], f"{path}[{index}]")
        if isinstance(value, dict):
            if len(value) < schema.get("minProperties", 0):
                self.errors.append(f"{path}: fewer than minProperties")
            for key in schema.get("required", []):
                if key not in value:
                    self.errors.append(f"{path}: missing required property {key}")
            properties = schema.get("properties", {})
            for key, item in value.items():
                if key in properties:
                    self.check(item, properties[key], f"{path}.{key}")
                elif schema.get("additionalProperties") is False:
                    self.errors.append(f"{path}: unexpected property {key}")
                elif isinstance(schema.get("additionalProperties"), dict):
                    self.check(item, schema["additionalProperties"], f"{path}.{key}")


def unique_index(items: list[dict], label: str) -> dict[str, dict]:
    result = {}
    for item in items:
        identifier = item["id"]
        if identifier in result:
            raise AssertionError(f"duplicate {label} id: {identifier}")
        result[identifier] = item
    return result


def load_query_module():
    path = ROOT / "query_claim_evidence.py"
    spec = importlib.util.spec_from_file_location("query_claim_evidence", path)
    if spec is None or spec.loader is None:
        raise AssertionError("cannot load competency-query program")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["profile", "study"], default="study", help="profile: generic metadata only; study: profile plus bytes and frozen-case regression")
    parser.add_argument("--record", type=Path, help="audit JSON to validate; alternative records require --mode profile")
    parser.add_argument("--verify-external", action="store_true", help="hash external workspace artifacts as well as packaged files")
    parser.add_argument("--write-report", action="store_true", help="write formal validation_report.json; requires --verify-external")
    args = parser.parse_args()
    if args.record and args.mode != "profile":
        parser.error("--record is for generic --mode profile; study mode audits the frozen local bundle")
    if args.mode == "profile" and (args.verify_external or args.write_report):
        parser.error("profile mode checks metadata only; use study mode for byte verification and the formal report")
    if args.write_report and not args.verify_external:
        parser.error("--write-report requires --verify-external so a formal report cannot downgrade external verification")

    checks: list[dict] = []

    def passed(name: str, detail: str):
        checks.append({"name": name, "status": "passed", "detail": detail})

    schema = json.loads((ROOT / "audit_record.schema.json").read_text(encoding="utf-8"))
    audit_path = args.record or ROOT / "allen_cahn_audit_record.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    from profile_validation import validate_profile
    profile_report = validate_profile(audit, schema)
    if args.mode == "profile":
        print(json.dumps({"validation_mode": "profile", "filesystem_integrity_performed": False, "frozen_study_regression_performed": False, **profile_report}, indent=2))
        return 0 if profile_report["passed"] else 1
    if not profile_report["passed"]:
        raise AssertionError("generic profile validation failed:\n" + "\n".join(profile_report["errors"][:100]))
    if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        raise AssertionError("schema does not declare JSON Schema Draft 2020-12")
    validator = SchemaValidator(schema)
    validator.check(audit, schema)
    if validator.errors:
        raise AssertionError("schema validation failed:\n" + "\n".join(validator.errors[:100]))
    passed("json_schema", "Draft 2020-12 schema validates the v2 audit record and enforces the minimum record fields")

    protocols = unique_index(audit["protocols"], "protocol")
    configurations = unique_index(audit["configurations"], "configuration")
    spatial_domains = unique_index(audit["spatial_domains"], "spatial domain")
    runs = unique_index(audit["runs"], "run")
    checkpoints = unique_index(audit["checkpoints"], "checkpoint")
    predictions = unique_index(audit["predictions"], "prediction")
    evaluations = unique_index(audit["evaluations"], "evaluation")
    controller_traces = unique_index(audit["controller_traces"], "controller trace")
    metrics = unique_index(audit["metrics"], "metric")
    claims = unique_index(audit["claims"], "claim")
    artifacts = unique_index(audit["artifacts"], "artifact")
    generations = unique_index(audit["generations"], "generation")
    if (len(configurations), len(spatial_domains), len(runs), len(checkpoints), len(predictions), len(evaluations), len(controller_traces), len(metrics), len(claims)) != (57, 4, 57, 57, 315, 258, 10, 17, 4):
        raise AssertionError(f"unexpected core counts: runs={len(runs)}, checkpoints={len(checkpoints)}, predictions={len(predictions)}, evaluations={len(evaluations)}, metrics={len(metrics)}, claims={len(claims)}")
    passed("record_counts", "57 configurations/runs/checkpoints, four spatial domains, 315 mechanism/domain-specific predictions, 258 evaluations, ten controller traces, 17 metrics and four claims are present")

    all_application_records = {}
    for collection in (protocols, configurations, spatial_domains, checkpoints, predictions, evaluations, controller_traces, metrics, claims, artifacts):
        overlap = set(all_application_records) & set(collection)
        if overlap:
            raise AssertionError(f"application record ids are not globally unique: {sorted(overlap)[:3]}")
        all_application_records.update(collection)
    for collection_name, collection in [
        ("protocol", protocols), ("configuration", configurations), ("spatial domain", spatial_domains), ("run", runs), ("checkpoint", checkpoints), ("prediction", predictions),
        ("evaluation", evaluations), ("controller trace", controller_traces), ("metric", metrics), ("claim", claims), ("artifact", artifacts), ("generation", generations),
    ]:
        for identifier, item in collection.items():
            if set(item["identity"]) != {"label", "type", "role"} or not all(item["identity"].values()):
                raise AssertionError(f"incomplete identity for {collection_name}: {identifier}")
    passed("minimum_identity", "every application record has non-empty label/type/role identity fields")

    expected_confirm = {(seed, arm) for seed in [5101, 5113, 5147, 5167, 5191, 5227, 5261, 5281, 5309, 5333] for arm in ["zero_rk4", "fixed_rate1", "grouped_rate1"]}
    expected_long = {(scenario, seed, arm) for scenario in ["blocks_04", "blocks_08", "blocks_12"] for seed in [6101, 6131, 6173] for arm in ["zero_rk4", "fixed_rate1", "grouped_rate1"]}
    observed_confirm = {(run["seed"], run["arm"]) for run in runs.values() if run["protocol_id"] == "protocol:confirmatory"}
    observed_long = {(run["scenario"], run["seed"], run["arm"]) for run in runs.values() if run["protocol_id"] == "protocol:long-sequence"}
    if observed_confirm != expected_confirm or observed_long != expected_long:
        raise AssertionError("57-run matrix is incomplete or contains unexpected cells")
    passed("run_matrix", "57/57 cells present: 30 confirmatory and 27 long-sequence development runs")

    expected_roles = {
        "protocol:confirmatory": {"direct full-series", "training", "validation", "direct-test", "numerical-continuation-test", "terminal-state"},
        "protocol:long-sequence": {"direct full-series", "training", "direct-test", "numerical-continuation-test", "terminal-state"},
    }
    for run in runs.values():
        if run["protocol_id"] not in protocols:
            raise AssertionError(f"run points to missing protocol: {run['id']}")
        if run["checkpoint_id"] not in checkpoints or checkpoints[run["checkpoint_id"]]["run_id"] != run["id"]:
            raise AssertionError(f"run/checkpoint backlink failed: {run['id']}")
        run_predictions = [predictions[prediction_id] for prediction_id in run["prediction_ids"]]
        run_evaluations = [evaluations[evaluation_id] for evaluation_id in run["evaluation_ids"]]
        if any(item["run_id"] != run["id"] for item in run_predictions + run_evaluations):
            raise AssertionError(f"run backlink mismatch: {run['id']}")
        roles = {item["identity"]["role"] for item in run_predictions}
        if roles != expected_roles[run["protocol_id"]]:
            raise AssertionError(f"prediction roles differ for {run['id']}: {sorted(roles)}")
        if run["protocol_id"] == "protocol:long-sequence" and run["unavailable_prediction_roles"] != ["validation"]:
            raise AssertionError(f"long run does not declare unavailable validation: {run['id']}")
        if run["protocol_id"] == "protocol:confirmatory" and run["unavailable_prediction_roles"]:
            raise AssertionError(f"confirmatory run unexpectedly declares missing prediction role: {run['id']}")
        controlled_artifact = artifacts[checkpoints[run["checkpoint_id"]]["parameter_state_source_artifact_id"]]
        controlled = json.loads((ROOT / controlled_artifact["path"]).read_text(encoding="utf-8"))
        if run["config_id"] not in configurations:
            raise AssertionError(f"run points to missing configuration entity: {run['id']}")
        config = configurations[run["config_id"]]
        config_source = artifacts[config["source_artifact_id"]]
        config_value = json_pointer(json.loads((ROOT / config_source["path"]).read_text(encoding="utf-8")), config["source_pointer"])
        if run["config_id"] != f"config:{run['config_digest']}" or canonical_digest(controlled["config"]) != run["config_digest"] or canonical_digest(config_value) != config["digest"]:
            raise AssertionError(f"run configuration digest mismatch: {run['id']}")
        if config["run_id"] != run["id"] or config["campaign"] != run["campaign"] or config["digest"] != run["config_digest"]:
            raise AssertionError(f"configuration entity/run backlink mismatch: {run['id']}")
        if int(controlled["completed_updates"]) != run["update_count"] or len(controlled["block_records"]) != run["block_count"]:
            raise AssertionError(f"run update/block counts differ from controlled record: {run['id']}")
        if int(controlled["nonfinite_updates"]) != run["nonfinite_updates"] or int(controlled["clipped_gradient_updates"]) != run["clipped_gradient_updates"]:
            raise AssertionError(f"run stability counts differ from controlled record: {run['id']}")
        timing = run["timing_source"]
        timing_artifact = artifacts[timing["artifact_id"]]
        timing_record = json.loads((ROOT / timing_artifact["path"]).read_text(encoding="utf-8"))
        if json_pointer(timing_record, timing["started_utc_pointer"]) != run["started_utc"] or json_pointer(timing_record, timing["training_finished_utc_pointer"]) != run["training_finished_utc"]:
            raise AssertionError(f"run timing strings differ from source record: {run['id']}")
        if not math.isclose(float(json_pointer(timing_record, timing["training_seconds_pointer"])), float(run["training_seconds"]), rel_tol=1e-12, abs_tol=1e-9):
            raise AssertionError(f"run duration differs from source record: {run['id']}")
        started = datetime.fromisoformat(run["started_utc"].replace("Z", "+00:00"))
        finished = datetime.fromisoformat(run["training_finished_utc"].replace("Z", "+00:00"))
        if abs((finished - started).total_seconds() - run["training_seconds"]) > 2.0:
            raise AssertionError(f"run timestamps and duration are inconsistent: {run['id']}")
    passed("run_minimum_fields", "campaign/protocol/arm/seed/block/update/stability/config/timing fields agree with copied controlled-run and record.json sources; missing long-sequence validation is explicit")

    spatial_source_artifact = artifacts["artifact:spatial-domain-sources"]
    spatial_source = json.loads((ROOT / spatial_source_artifact["path"]).read_text(encoding="utf-8"))
    spatial_numeric_fields = ["lower_bound", "last_coordinate", "upper_bound_exclusive", "points", "spacing", "period"]
    for spatial in spatial_domains.values():
        source_value = json_pointer(spatial_source, spatial["source_pointer"])
        if source_value["id"] != spatial["id"] or source_value["array_member"] != spatial["origin_array_member"]:
            raise AssertionError(f"spatial domain source identity mismatch: {spatial['id']}")
        for field in spatial_numeric_fields:
            if not math.isclose(float(source_value[field]), float(spatial[field]), rel_tol=1e-12, abs_tol=1e-14):
                raise AssertionError(f"spatial domain differs from packaged source: {spatial['id']} {field}")
        if any(identifier not in artifacts for identifier in spatial["origin_artifact_ids"]):
            raise AssertionError(f"spatial domain origin artifact missing: {spatial['id']}")
        if args.verify_external:
            origin = artifacts[spatial["origin_artifact_ids"][0]]
            observed = npz_vector_summary((ROOT / origin["path"]).resolve(), spatial["origin_array_member"])
            for field in spatial_numeric_fields:
                if not math.isclose(float(observed[field]), float(spatial[field]), rel_tol=1e-12, abs_tol=1e-14):
                    raise AssertionError(f"spatial domain differs from external NPZ: {spatial['id']} {field}")
    if args.verify_external:
        for run in runs.values():
            full_prediction = next(predictions[item] for item in run["prediction_ids"] if predictions[item]["identity"]["role"] == "direct full-series")
            raw_artifact = artifacts[full_prediction["output_artifact_id"]]
            raw_path = (ROOT / raw_artifact["path"]).resolve()
            direct_domain = spatial_domains[full_prediction["native_spatial_domain_id"]]
            continuation_prediction = next(predictions[item] for item in run["prediction_ids"] if predictions[item]["identity"]["role"] == "numerical-continuation-test")
            continuation_domain = spatial_domains[continuation_prediction["native_spatial_domain_id"]]
            for member, expected in [("x", direct_domain), ("fd_x", continuation_domain)]:
                observed = npz_vector_summary(raw_path, member)
                if any(not math.isclose(float(observed[field]), float(expected[field]), rel_tol=1e-12, abs_tol=1e-14) for field in spatial_numeric_fields):
                    raise AssertionError(f"run-specific {member} grid differs from declared spatial domain: {run['id']}")
    passed("spatial_domains", "four explicit periodic half-open spatial-domain entities resolve to packaged coordinate summaries" + (" and externally verified NPZ members" if args.verify_external else "; external NPZ coordinate verification was not requested"))

    for checkpoint in checkpoints.values():
        if checkpoint["checkpoint_file_sha256"] is not None:
            raise AssertionError("serialized checkpoint hash asserted even though no checkpoint file was preserved")
        source = artifacts[checkpoint["parameter_state_source_artifact_id"]]
        value = json_pointer(json.loads((ROOT / source["path"]).read_text(encoding="utf-8")), checkpoint["parameter_state_source_pointer"])
        if value != checkpoint["parameter_state_sha256"]:
            raise AssertionError(f"checkpoint parameter-state digest differs from source: {checkpoint['id']}")
    passed("checkpoint_digests", "all 57 final-iterate parameter_state_sha256 values are sourced by JSON Pointer; unknown serialized checkpoint hashes remain null")

    generation_activity_ids = {item["id"] for item in generations.values()}
    for prediction in predictions.values():
        if prediction["run_id"] not in runs or prediction["checkpoint_id"] not in checkpoints:
            raise AssertionError(f"prediction run/checkpoint missing: {prediction['id']}")
        if prediction["source_activity_id"] not in generation_activity_ids:
            raise AssertionError(f"prediction activity missing: {prediction['id']}")
        if prediction["status"] == "complete" and prediction["output_artifact_id"] not in artifacts:
            raise AssertionError(f"complete prediction output artifact missing: {prediction['id']}")
        if prediction["status"] == "field_not_persisted" and prediction["output_artifact_id"] is not None:
            raise AssertionError(f"unpersisted prediction incorrectly declares an output artifact: {prediction['id']}")
        if prediction["native_spatial_domain_id"] not in spatial_domains or prediction["evaluation_spatial_domain_id"] not in spatial_domains:
            raise AssertionError(f"prediction spatial domain missing: {prediction['id']}")
        run = runs[prediction["run_id"]]
        if spatial_domains[prediction["native_spatial_domain_id"]]["campaign"] != run["campaign"] or spatial_domains[prediction["evaluation_spatial_domain_id"]]["campaign"] != run["campaign"]:
            raise AssertionError(f"prediction spatial domain campaign mismatch: {prediction['id']}")
        for evaluation_id in prediction["evaluation_ids"]:
            if evaluation_id not in evaluations or evaluations[evaluation_id]["prediction_id"] != prediction["id"]:
                raise AssertionError(f"prediction/evaluation backlink failed: {prediction['id']}")
        if prediction["identity"]["role"] == "numerical-continuation-test":
            native = spatial_domains[prediction["native_spatial_domain_id"]]
            evaluated = spatial_domains[prediction["evaluation_spatial_domain_id"]]
            if run["protocol_id"] == "protocol:confirmatory":
                if native["points"] != 256 or evaluated["points"] != 1024 or native["id"] == evaluated["id"]:
                    raise AssertionError(f"confirmatory continuation 256-to-1024 grid path is not explicit: {prediction['id']}")
                output = artifacts[prediction["output_artifact_id"]]
                if prediction["status"] != "complete" or output["path_scope"] != "external_workspace" or not output["id"].startswith("artifact:continuation-fields:") or prediction["output_field"] != "continued_prediction":
                    raise AssertionError(f"confirmatory continuation field persistence is misrepresented: {prediction['id']}")
            else:
                if native["points"] != 256 or evaluated["id"] != native["id"]:
                    raise AssertionError(f"long-sequence continuation 256-node scoring domain is misrepresented: {prediction['id']}")
                if prediction["status"] != "field_not_persisted" or prediction["output_artifact_id"] is not None or "field not persisted" not in prediction["output_field"]:
                    raise AssertionError(f"long-sequence continuation field-availability status is misrepresented: {prediction['id']}")
        elif prediction["native_spatial_domain_id"] != prediction["evaluation_spatial_domain_id"]:
            raise AssertionError(f"non-continuation prediction unexpectedly changes spatial grid: {prediction['id']}")
    for evaluation in evaluations.values():
        prediction = predictions[evaluation["prediction_id"]]
        if evaluation["run_id"] != prediction["run_id"] or evaluation["id"] not in prediction["evaluation_ids"]:
            raise AssertionError(f"evaluation backlink failed: {evaluation['id']}")
        if evaluation["reference_artifact_id"] not in artifacts or evaluation["source_artifact_id"] not in artifacts:
            raise AssertionError(f"evaluation reference/source missing: {evaluation['id']}")
        source_artifact = artifacts[evaluation["source_artifact_id"]]
        source = json.loads((ROOT / source_artifact["path"]).read_text(encoding="utf-8"))
        source_value = json_pointer(source, evaluation["source_pointer"])
        if not math.isclose(float(source_value), float(evaluation["value"]), rel_tol=1e-12, abs_tol=1e-14):
            raise AssertionError(f"evaluation value differs from source: {evaluation['id']}")
    passed("prediction_evaluations", "315 distinct prediction records carry checkpoint/mechanism/temporal/native-spatial/evaluation-spatial/output/activity fields; confirmatory continuation fields are hash-addressed while long-sequence field_not_persisted status and null output artifacts are explicit; all 258 evaluations match source JSON")

    verified_packaged = 0
    verified_external = 0
    verified_external_predictions = 0
    verified_external_continuation_fields = 0
    for artifact in artifacts.values():
        if Path(artifact["path"]).is_absolute():
            raise AssertionError(f"absolute artifact path: {artifact['id']}")
        path = (ROOT / artifact["path"]).resolve()
        if artifact["path_scope"] == "external_workspace" and not args.verify_external:
            continue
        if not path.is_file():
            raise AssertionError(f"missing artifact: {artifact['id']} -> {path}")
        if path.stat().st_size != artifact["bytes"] or sha256(path) != artifact["sha256"]:
            raise AssertionError(f"artifact hash/size mismatch: {artifact['id']}")
        if artifact["path_scope"] == "packaged":
            verified_packaged += 1
        else:
            verified_external += 1
            if artifact["id"].startswith("artifact:prediction-array:"):
                verified_external_predictions += 1
            if artifact["id"].startswith("artifact:continuation-fields:"):
                verified_external_continuation_fields += 1
    detail = f"verified {verified_packaged} packaged artifacts"
    if args.verify_external:
        detail += f" and {verified_external} external workspace artifacts, including {verified_external_predictions} raw prediction arrays and {verified_external_continuation_fields} persisted confirmatory continuation-field arrays"
    else:
        detail += "; external workspace hashes were not requested"
    passed("artifact_hashes", detail)

    registry = json.loads((ROOT / "evidence/evidence_sources.json").read_text(encoding="utf-8"))
    if len(registry["sources"]) != 17 or registry.get("path_base") != "provenance_root":
        raise AssertionError("portable evidence source registry is incomplete")
    for source in registry["sources"]:
        if "copied_path" in source or Path(source["bundle_path"]).is_absolute():
            raise AssertionError("source registry retains a misleading or absolute copied path")
        path = ROOT / source["bundle_path"]
        if not path.is_file() or sha256(path) != source["sha256"] or path.stat().st_size != source["bytes"]:
            raise AssertionError(f"portable registered source failed hash check: {source['bundle_path']}")
    passed("portable_source_registry", "all 17 registered sources have existing bundle-relative copies and verified hashes")

    for trace in controller_traces.values():
        if trace["run_id"] not in runs or trace["protocol_id"] not in protocols or trace["configuration_id"] not in configurations:
            raise AssertionError(f"controller trace run/protocol/configuration path is broken: {trace['id']}")
        run = runs[trace["run_id"]]
        if run["arm"] != "grouped_rate1" or trace["protocol_id"] != run["protocol_id"] or trace["configuration_id"] != run["config_id"]:
            raise AssertionError(f"controller trace path differs from grouped run: {trace['id']}")
        if any(identifier not in artifacts for identifier in trace["source_artifact_ids"]):
            raise AssertionError(f"controller trace source artifact missing: {trace['id']}")
        source_artifact = artifacts[trace["summary_source_artifact_id"]]
        source = json.loads((ROOT / source_artifact["path"]).read_text(encoding="utf-8"))
        source_value = json_pointer(source, trace["summary_source_pointer"])
        if int(source_value["updates"]) != trace["updates"] or source_value["npz_sha256"] != trace["prediction_array_sha256"] or source_value["diagnostics_sha256"] != trace["diagnostics_sha256"]:
            raise AssertionError(f"controller trace differs from per-seed summary: {trace['id']}")
        raw = next(artifacts[item] for item in trace["source_artifact_ids"] if item.startswith("artifact:prediction-array:"))
        diagnostic = next(artifacts[item] for item in trace["source_artifact_ids"] if item.startswith("artifact:diagnostics:"))
        if raw["sha256"] != trace["prediction_array_sha256"] or diagnostic["sha256"] != trace["diagnostics_sha256"]:
            raise AssertionError(f"controller trace source hashes are inconsistent: {trace['id']}")
    passed("controller_traces", "ten grouped-run controller-trace entities resolve to their run, protocol, exact configuration, raw adaptive-trace NPZ and diagnostic JSON hashes")

    for metric in metrics.values():
        if metric["source_artifact_id"] not in artifacts:
            raise AssertionError(f"metric source missing: {metric['id']}")
        source_artifact = artifacts[metric["source_artifact_id"]]
        source = json.loads((ROOT / source_artifact["path"]).read_text(encoding="utf-8"))
        source_value = json_pointer(source, metric["source_pointer"])
        if isinstance(source_value, dict):
            source_value = source_value["paired_mean_difference"] if metric["kind"] == "paired_mean_difference" else source_value["median"]
        if not math.isclose(float(metric["value"]), float(source_value), rel_tol=1e-12, abs_tol=1e-14):
            raise AssertionError(f"aggregate metric differs from source JSON: {metric['id']}")
        if any(identifier not in predictions for identifier in metric["prediction_ids"]):
            raise AssertionError(f"metric points to missing prediction: {metric['id']}")
        if any(identifier not in evaluations for identifier in metric["evaluation_ids"]):
            raise AssertionError(f"metric points to missing evaluation: {metric['id']}")
        if any(identifier not in controller_traces for identifier in metric["controller_trace_ids"]):
            raise AssertionError(f"metric points to missing controller trace: {metric['id']}")
        if metric["id"].startswith("metric:controller:"):
            if metric["prediction_ids"] or metric["evaluation_ids"] or set(metric["controller_trace_ids"]) != set(controller_traces):
                raise AssertionError(f"controller metric is incorrectly derived from field predictions/evaluations: {metric['id']}")
            if metric["kind"] == "descriptive_summary":
                if metric.get("interval_kind") != "interquartile_range" or "ci95" in metric:
                    raise AssertionError(f"controller IQR is mislabeled as a confidence interval: {metric['id']}")
                source_item = json_pointer(source, metric["source_pointer"])
                expected = [float(source_item["q25"]), float(source_item["q75"])]
                if metric.get("interval_bounds") != expected:
                    raise AssertionError(f"controller IQR differs from source quantiles: {metric['id']}")
        elif metric["controller_trace_ids"]:
            raise AssertionError(f"field-error metric unexpectedly links controller traces: {metric['id']}")
    passed("aggregate_metrics", "all 17 aggregate metric values match JSON Pointer sources; field metrics link predictions/evaluations, while C4 metrics link only the ten controller-trace entities and label IQRs explicitly")

    if set(claims) != {"C1", "C2", "C3", "C4"}:
        raise AssertionError(f"unexpected formal claim set: {sorted(claims)}")
    if any("fixed-zero" in identifier for identifier in list(metrics) + list(claims)):
        raise AssertionError("fixed-versus-zero contextual evidence leaked into formal claim ids")
    for claim in claims.values():
        if not claim["supporting_evidence_ids"] or not claim["limiting_evidence_ids"] or not claim["supporting_artifact_ids"] or not claim["limiting_artifact_ids"]:
            raise AssertionError(f"claim lacks explicit supporting/limiting identifiers: {claim['id']}")
        if any(identifier not in all_application_records for identifier in claim["supporting_evidence_ids"] + claim["limiting_evidence_ids"]):
            raise AssertionError(f"claim evidence identifier does not resolve: {claim['id']}")
        if any(identifier not in artifacts for identifier in claim["supporting_artifact_ids"] + claim["limiting_artifact_ids"]):
            raise AssertionError(f"claim artifact identifier does not resolve: {claim['id']}")
    c1 = claims["C1"]
    expected_c1_training = "metric:confirmatory:grouped-fixed:training"
    expected_heldout = {
        "metric:confirmatory:grouped-fixed:validation",
        "metric:confirmatory:grouped-fixed:test",
        "metric:confirmatory:grouped-fixed:continuation-test",
        "metric:confirmatory:grouped-fixed:terminal",
    }
    if c1["metric_ids"] != [expected_c1_training] or not expected_heldout.issubset(c1["limiting_evidence_ids"]):
        raise AssertionError("C1 does not expose training support and held-out limiting evidence")
    if not any(identifier.startswith("evaluation:confirmatory:") for identifier in c1["limiting_evidence_ids"]):
        raise AssertionError("C1 limiting evidence does not include run-level held-out evaluations")
    c2 = claims["C2"]
    expected_c2_direction = "two-sided 95 percent paired t intervals include zero on all four held-out or terminal criteria"
    if c2["direction"] != expected_c2_direction:
        raise AssertionError("C2 direction must identify the two-sided paired t intervals rather than an unspecified interval family")
    if set(c2["metric_ids"]) != expected_heldout:
        raise AssertionError("C2 does not cover validation, direct-test, continuation-test and terminal-state metrics")
    required_roles = {"validation", "direct-test", "numerical-continuation-test", "terminal-state"}
    c2_evaluations = [evaluations[identifier] for identifier in c2["supporting_evidence_ids"] if identifier in evaluations]
    if {item["identity"]["role"] for item in c2_evaluations} != required_roles:
        raise AssertionError("C2 does not link run-level evaluations for all four required roles")
    for metric_id in expected_heldout:
        interval = metrics[metric_id].get("ci95")
        if interval is None or not (float(interval[0]) <= 0.0 <= float(interval[1])):
            raise AssertionError(f"C2 metric interval does not include zero: {metric_id}")
    if claims["C3"]["evidence_assessment"] != "qualified" or "protocol:long-sequence" not in claims["C3"]["limiting_evidence_ids"]:
        raise AssertionError("C3 does not encode its qualified three-seed development boundary")
    c4 = claims["C4"]
    required_budget_metrics = {
        "metric:controller:physics-budget-max-discrepancy",
        "metric:controller:constraint-budget-max-discrepancy",
        "metric:controller:fixed-anchor-max-discrepancy",
        "metric:controller:guard-trigger-count",
    }
    required_controller_artifacts = {"artifact:controller-diagnostic", "artifact:table:controller", "artifact:figure:controller:pdf", "artifact:figure:controller:png"}
    if not required_budget_metrics.issubset(c4["metric_ids"]) or not required_controller_artifacts.issubset(c4["supporting_artifact_ids"]):
        raise AssertionError("C4 does not link budget invariants and controller artefacts")
    for metric_id in required_budget_metrics:
        if abs(float(metrics[metric_id]["value"])) > 1e-12:
            raise AssertionError(f"controller budget/guard invariant exceeds tolerance: {metric_id}")
    if float(metrics["metric:controller:replay-budget-share"]["value"]) <= 0.5:
        raise AssertionError("later-block replay budget share is not a majority")
    if "protocol:confirmatory" not in c4["limiting_evidence_ids"] or "serially dependent" not in c4["qualification"]:
        raise AssertionError("C4 does not encode serial-dependence/no-randomized-mediator limitations")
    expected_fixed_zero_boundary = "outside the DIM claim graph; the current companion addresses the numerical question with a distinct cohort"
    central_summary = json.loads((ROOT / "evidence/dim_evidence_summary.json").read_text(encoding="utf-8"))
    if central_summary["interpretation"].get("fixed_vs_zero") != expected_fixed_zero_boundary:
        raise AssertionError("central evidence summary assigns fixed-versus-zero to the wrong publication boundary")
    synthesizer_text = (ROOT / "evidence/generators/evidence_synthesizer.py").read_text(encoding="utf-8")
    if expected_fixed_zero_boundary not in synthesizer_text:
        raise AssertionError("evidence synthesizer retains stale fixed-versus-zero companion wording")
    passed("formal_claim_boundary", "all C1--C4 claims resolve supporting and limiting identifiers; C1/C2 held-out boundaries, qualified C3 development status and C4 budget/serial-dependence limits are explicit; fixed-versus-zero remains outside the graph")

    declared_ids = set(all_application_records) | set(runs) | set(generations)
    for item in generations.values():
        if item["generator_id"] not in artifacts or artifacts[item["generator_id"]]["sha256"] != item["generator_digest"]:
            raise AssertionError(f"generation generator digest mismatch: {item['id']}")
        if any(identifier not in declared_ids for identifier in item["input_ids"]):
            unknown = [identifier for identifier in item["input_ids"] if identifier not in declared_ids]
            raise AssertionError(f"generation has unknown inputs: {item['id']} -> {unknown[:3]}")
        if any(identifier not in all_application_records for identifier in item["output_ids"]):
            raise AssertionError(f"generation has unknown output: {item['id']}")
    passed("generation_records", f"all {len(generations)} generation records bind inputs/outputs to a hash-verified generator and package version")

    trace_count = 0
    for claim in claims.values():
        for metric_id in claim["metric_ids"]:
            metric = metrics[metric_id]
            for prediction_id in metric["prediction_ids"]:
                run = runs[predictions[prediction_id]["run_id"]]
                if run["protocol_id"] not in protocols:
                    raise AssertionError("broken claim-to-protocol trace")
                trace_count += 1
            for evaluation_id in metric["evaluation_ids"]:
                run = runs[evaluations[evaluation_id]["run_id"]]
                if run["protocol_id"] not in protocols:
                    raise AssertionError("broken claim-to-evaluation-to-protocol trace")
                trace_count += 1
            for controller_trace_id in metric["controller_trace_ids"]:
                trace = controller_traces[controller_trace_id]
                run = runs[trace["run_id"]]
                if trace["configuration_id"] != run["config_id"] or trace["protocol_id"] != run["protocol_id"]:
                    raise AssertionError("broken claim-to-controller-trace-to-run/configuration/protocol path")
                trace_count += 1
    passed("claim_traceability", f"validated {trace_count} claim-to-metric-to-prediction/evaluation/controller-trace-to-run/configuration/protocol paths")

    prov_agents = unique_index(audit["prov"]["agents"], "PROV agent")
    prov_entities = unique_index(audit["prov"]["entities"], "PROV entity")
    prov_activities = unique_index(audit["prov"]["activities"], "PROV activity")
    for collection_name, collection in [("agent", prov_agents), ("entity", prov_entities), ("activity", prov_activities)]:
        for identifier, item in collection.items():
            if set(item["identity"]) != {"label", "type", "role"} or not all(item["identity"].values()):
                raise AssertionError(f"incomplete PROV {collection_name} identity: {identifier}")
    for agent in prov_agents.values():
        if agent["type"] != "prov:SoftwareAgent" or agent["identity"]["type"] != "prov:SoftwareAgent" or agent["source_artifact_id"] not in artifacts:
            raise AssertionError(f"software agent lacks a resolvable source identifier: {agent['id']}")
        if artifacts[agent["source_artifact_id"]]["sha256"] != agent["source_digest"]:
            raise AssertionError(f"software agent source digest mismatch: {agent['id']}")
        if agent["software_version"] == "source-snapshot-no-semantic-version" and "No semantic version was recorded" not in agent["qualification"]:
            raise AssertionError(f"unversioned software agent lacks explicit qualification: {agent['id']}")
    prov_ids = set(prov_agents) | set(prov_entities) | set(prov_activities)
    relation_tuples = {(item["predicate"], item["subject_id"], item["object_id"]) for item in audit["prov"]["relations"]}
    for relation in audit["prov"]["relations"]:
        if relation["subject_id"] not in prov_ids or relation["object_id"] not in prov_ids:
            raise AssertionError(f"broken W3C PROV relation: {relation}")
    for record_id in all_application_records:
        entity_id = f"entity:{record_id}"
        if entity_id not in prov_entities or prov_entities[entity_id]["record_id"] != record_id:
            raise AssertionError(f"missing PROV entity for application record: {record_id}")
    for run in runs.values():
        checkpoint_entity = f"entity:{run['checkpoint_id']}"
        training_activity = f"activity:{run['id']}"
        if ("prov:wasGeneratedBy", checkpoint_entity, training_activity) not in relation_tuples:
            raise AssertionError(f"checkpoint is not generated by training: {run['id']}")
        if ("prov:used", training_activity, f"entity:{run['config_id']}") not in relation_tuples or ("prov:used", training_activity, f"entity:{run['protocol_id']}") not in relation_tuples:
            raise AssertionError(f"training activity does not use explicit configuration and protocol entities: {run['id']}")
        for prediction_id in run["prediction_ids"]:
            prediction = predictions[prediction_id]
            prediction_activity = generations[prediction["source_activity_id"]]["prov_activity_id"]
            if ("prov:wasGeneratedBy", f"entity:{prediction_id}", prediction_activity) not in relation_tuples:
                raise AssertionError(f"prediction is not generated by its prediction activity: {prediction_id}")
            if ("prov:used", prediction_activity, checkpoint_entity) not in relation_tuples:
                raise AssertionError(f"prediction activity does not use checkpoint: {prediction_id}")
            for spatial_id in {prediction["native_spatial_domain_id"], prediction["evaluation_spatial_domain_id"]}:
                if ("prov:used", prediction_activity, f"entity:{spatial_id}") not in relation_tuples:
                    raise AssertionError(f"prediction activity does not use spatial domain: {prediction_id} -> {spatial_id}")
            if prediction["identity"]["role"] == "numerical-continuation-test":
                terminal_id = prediction_id.rsplit(":", 1)[0] + ":terminal-state"
                if ("prov:used", prediction_activity, f"entity:{terminal_id}") not in relation_tuples:
                    raise AssertionError(f"continuation prediction activity does not use terminal-state prediction: {prediction_id}")
                direct_full_id = prediction_id.rsplit(":", 1)[0] + ":direct-full-series"
                if ("prov:used", prediction_activity, f"entity:{direct_full_id}") in relation_tuples:
                    raise AssertionError(f"continuation prediction incorrectly uses direct full-series as upstream state: {prediction_id}")
        for evaluation_id in run["evaluation_ids"]:
            evaluation = evaluations[evaluation_id]
            evaluation_activity = generations[evaluation["source_activity_id"]]["prov_activity_id"]
            if ("prov:used", evaluation_activity, checkpoint_entity) not in relation_tuples:
                raise AssertionError(f"evaluation activity does not use checkpoint: {evaluation_id}")
            if ("prov:used", evaluation_activity, f"entity:{evaluation['prediction_id']}") not in relation_tuples:
                raise AssertionError(f"evaluation activity does not use its prediction: {evaluation_id}")
            if ("prov:used", evaluation_activity, f"entity:{evaluation['reference_artifact_id']}") not in relation_tuples:
                raise AssertionError(f"evaluation activity does not use its reference: {evaluation_id}")
            if ("prov:wasGeneratedBy", f"entity:{evaluation_id}", evaluation_activity) not in relation_tuples:
                raise AssertionError(f"evaluation entity is not generated by evaluation activity: {evaluation_id}")
            prediction_activity = generations[predictions[evaluation["prediction_id"]]["source_activity_id"]]["prov_activity_id"]
            if prediction_activity == evaluation_activity:
                raise AssertionError(f"prediction and evaluation share an activity: {evaluation_id}")
    for metric in metrics.values():
        activity_id = generations[f"generation:metric:{metric['id']}"]["prov_activity_id"]
        if ("prov:wasGeneratedBy", f"entity:{metric['id']}", activity_id) not in relation_tuples:
            raise AssertionError(f"metric has no generating synthesis activity: {metric['id']}")
        for identifier in metric["prediction_ids"] + metric["evaluation_ids"] + metric["controller_trace_ids"]:
            if ("prov:used", activity_id, f"entity:{identifier}") not in relation_tuples:
                raise AssertionError(f"metric synthesis does not use linked evidence: {metric['id']} -> {identifier}")
    for trace in controller_traces.values():
        trace_generation = generations[f"generation:controller-trace:confirmatory:{runs[trace['run_id']]['seed']}:grouped_rate1"]
        activity_id = trace_generation["prov_activity_id"]
        if ("prov:wasGeneratedBy", f"entity:{trace['id']}", activity_id) not in relation_tuples:
            raise AssertionError(f"controller trace has no generating activity: {trace['id']}")
        for identifier in [trace["protocol_id"], trace["configuration_id"]] + trace["source_artifact_ids"]:
            if ("prov:used", activity_id, f"entity:{identifier}") not in relation_tuples:
                raise AssertionError(f"controller trace activity does not use source entity: {trace['id']} -> {identifier}")
    for generation in generations.values():
        activity_id = generation["prov_activity_id"]
        associated = [object_id for predicate, subject_id, object_id in relation_tuples if predicate == "prov:wasAssociatedWith" and subject_id == activity_id]
        if len(associated) != 1 or prov_agents[associated[0]]["source_artifact_id"] != generation["generator_id"]:
            raise AssertionError(f"activity/software-agent source mismatch: {generation['id']}")
    presentations = {
        "C1": ["artifact:table:confirmatory", "artifact:figure:confirmatory:pdf", "artifact:figure:confirmatory:png"],
        "C2": ["artifact:table:confirmatory", "artifact:figure:confirmatory:pdf", "artifact:figure:confirmatory:png"],
        "C3": ["artifact:table:long", "artifact:figure:long:pdf", "artifact:figure:long:png"],
        "C4": ["artifact:table:controller", "artifact:figure:controller:pdf", "artifact:figure:controller:png"],
    }
    for claim_id, artifact_ids in presentations.items():
        for artifact_id in artifact_ids:
            if ("prov:wasDerivedFrom", f"entity:{artifact_id}", f"entity:{claim_id}") not in relation_tuples:
                raise AssertionError(f"presentation artifact is not derived from claim: {artifact_id} -> {claim_id}")
            for metric_id in claims[claim_id]["metric_ids"]:
                if ("prov:wasDerivedFrom", f"entity:{artifact_id}", f"entity:{metric_id}") not in relation_tuples:
                    raise AssertionError(f"presentation artifact is not derived from metric: {artifact_id} -> {metric_id}")
    passed("w3c_prov_graph", f"validated {len(prov_entities)} entities, {len(prov_activities)} activities, {len(prov_agents)} source-identified software agents and {len(relation_tuples)} relations, including configuration/domain/checkpoint/prediction/evaluation/controller-trace/metric/presentation derivations")

    with (ROOT / "claim_evidence_matrix.csv").open(encoding="utf-8", newline="") as handle:
        matrix = list(csv.DictReader(handle))
    if [row["claim_id"] for row in matrix] != ["C1", "C2", "C3", "C4"]:
        raise AssertionError("claim_evidence_matrix.csv is not aligned to C1--C4")
    for row in matrix:
        claim = claims[row["claim_id"]]
        if row["direction"] != claim["direction"] or row["supporting_evidence_ids"].split(";") != claim["supporting_evidence_ids"]:
            raise AssertionError(f"claim matrix differs from machine claim: {row['claim_id']}")
        expected_limiting = claim["limiting_evidence_ids"]
        observed_limiting = [] if not row["limiting_evidence_ids"] else row["limiting_evidence_ids"].split(";")
        if observed_limiting != expected_limiting:
            raise AssertionError(f"claim matrix limiting evidence differs: {row['claim_id']}")
    passed("claim_matrix", "CSV reproduces C1--C4 directions and supporting/limiting evidence IDs")

    query_module = load_query_module()
    rebuilt_queries = query_module.build_query_results(audit)
    stored_queries = json.loads((ROOT / "competency_query_results.json").read_text(encoding="utf-8"))
    if rebuilt_queries != stored_queries or [item["competency_question_id"] for item in stored_queries["competency_results"]] != [f"CQ{index}" for index in range(1, 7)]:
        raise AssertionError("stored competency-query output cannot be rebuilt from the audit record")
    for query in stored_queries["queries"]:
        for boundary in ("supporting", "limiting"):
            for path in query[boundary]["paths"]:
                if path["run_id"] not in runs or path["protocol_id"] != runs[path["run_id"]]["protocol_id"]:
                    raise AssertionError(f"invalid competency-query path: {path}")
                run = runs[path["run_id"]]
                if path["configuration_id"] != run["config_id"] or path["configuration_digest"] != configurations[run["config_id"]]["digest"]:
                    raise AssertionError(f"competency-query path omits or misstates configuration: {path}")
                if path["prediction_id"]:
                    prediction = predictions[path["prediction_id"]]
                    if path["native_spatial_domain_id"] != prediction["native_spatial_domain_id"] or path["evaluation_spatial_domain_id"] != prediction["evaluation_spatial_domain_id"]:
                        raise AssertionError(f"competency-query spatial domains differ from prediction: {path}")
                if path["controller_trace_id"] and controller_traces[path["controller_trace_id"]]["run_id"] != path["run_id"]:
                    raise AssertionError(f"competency-query controller-trace path is broken: {path}")
            expected_configs = list(dict.fromkeys(path["configuration_id"] for path in query[boundary]["paths"]))
            if query[boundary]["configuration_ids"] != expected_configs:
                raise AssertionError(f"competency-query configuration summary differs from paths: {query['claim_id']} {boundary}")
    if not all(next(item for item in stored_queries["queries"] if item["claim_id"] == claim_id)["supporting"]["configuration_ids"] for claim_id in ["C1", "C2"]):
        raise AssertionError("C1/C2 claim traversals do not return exact configuration entities for CQ1")
    c4_query = next(item for item in stored_queries["queries"] if item["claim_id"] == "C4")
    if set(c4_query["supporting"]["controller_trace_ids"]) != set(controller_traces):
        raise AssertionError("C4 claim traversal does not resolve controller traces for CQ1/CQ4")
    passed("competency_queries", "claim queries C1--C4 are distinct from CQ1--CQ6; rebuilt CQ1--CQ4 answers report origins, mechanisms/domains, evaluations and claim boundaries, while CQ5--CQ6 explicitly delegate unperformed integrity/schema checks")

    canonical_crate_path = ROOT / "ro-crate-metadata.json"
    alias_crate_path = ROOT / "ro-crate-metadata.jsonld"
    crate = json.loads(canonical_crate_path.read_text(encoding="utf-8"))
    if canonical_crate_path.read_bytes() != alias_crate_path.read_bytes():
        raise AssertionError("legacy .jsonld alias is not byte-identical to canonical RO-Crate descriptor")
    if "https://w3id.org/ro/crate/1.3/context" not in json.dumps(crate.get("@context")):
        raise AssertionError("RO-Crate does not declare the 1.3 context")
    graph = {item["@id"]: item for item in crate["@graph"]}
    if "ro-crate-metadata.json" not in graph or "ro-crate-metadata.jsonld" in graph:
        raise AssertionError("RO-Crate descriptor id is not canonical ro-crate-metadata.json")
    descriptor = graph["ro-crate-metadata.json"]
    root_entity = graph.get("./")
    if descriptor.get("about") != {"@id": "./"} or not root_entity:
        raise AssertionError("RO-Crate descriptor/root relationship is invalid")
    if not root_entity.get("datePublished") or root_entity.get("license") != {"@id": "#license-unassigned"}:
        raise AssertionError("RO-Crate root lacks publication date or honest unassigned-licence reference")
    if any(item.get("@id") == "http://www.w3.org/ns/prov#" for item in root_entity.get("conformsTo", [])):
        raise AssertionError("RO-Crate root misleadingly asserts direct PROV conformance")
    license_entity = graph.get("#license-unassigned", {})
    if "grants no reuse rights" not in license_entity.get("description", ""):
        raise AssertionError("unassigned licence metadata could be mistaken for a reuse grant")
    for part in root_entity["hasPart"]:
        relative = part["@id"]
        file_entity = graph.get(relative)
        path = ROOT / relative
        if file_entity is None or not path.is_file() or file_entity.get("identifier") != f"sha256:{sha256(path)}":
            raise AssertionError(f"RO-Crate file digest mismatch: {relative}")
    audit_file_entity = graph.get("allen_cahn_audit_record.json", {})
    if "W3C PROV class/relation projection" not in audit_file_entity.get("description", ""):
        raise AssertionError("RO-Crate does not honestly describe the custom audit record's PROV-aligned projection")
    passed("ro_crate", f"canonical RO-Crate 1.3 descriptor is ro-crate-metadata.json with datePublished, explicit unassigned-licence status, no direct PROV-conformance claim and {len(root_entity['hasPart'])} hash-described packaged files")

    if "/home/" in json.dumps(audit) or "/home/" in json.dumps(crate):
        raise AssertionError("machine-readable audit/RO-Crate contains an absolute /home path")
    passed("portable_machine_paths", "audit record and RO-Crate metadata contain no absolute /home paths; historical absolute commands may remain only inside copied raw records")

    report = {
        "schema_version": 2,
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "validation_layers": {
            "generic_profile": profile_report,
            "frozen_study_regression": {
                "status": "passed", "scope": "Allen--Cahn frozen campaign only; counts, seeds, stored-source values and C1--C4 outcome assertions are not general profile requirements",
                "checks": [check["name"] for check in checks],
            },
        },
        "validation_modes": {
            "packaged": {"status": "passed", "performed": True, "verified_artifacts": verified_packaged},
            "external_workspace": {
                "status": "passed" if args.verify_external else "not_performed",
                "performed": bool(args.verify_external),
                "verified_artifacts": verified_external,
                "verified_raw_prediction_arrays": verified_external_predictions,
                "verified_continuation_field_arrays": verified_external_continuation_fields,
            },
        },
        "counts": {
            "protocols": len(protocols),
            "configurations": len(configurations),
            "spatial_domains": len(spatial_domains),
            "runs": len(runs),
            "checkpoints": len(checkpoints),
            "predictions": len(predictions),
            "evaluations": len(evaluations),
            "controller_traces": len(controller_traces),
            "metrics": len(metrics),
            "claims": len(claims),
            "artifacts": len(artifacts),
            "generations": len(generations),
            "prov_entities": len(prov_entities),
            "prov_activities": len(prov_activities),
            "prov_relations": len(relation_tuples),
            "prov_agents": len(prov_agents),
        },
        "checks": checks,
    }
    if args.write_report:
        (ROOT / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
