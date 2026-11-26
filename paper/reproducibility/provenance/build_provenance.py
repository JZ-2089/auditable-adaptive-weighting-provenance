#!/usr/bin/env python3
"""Build the Allen--Cahn DIM claim-evidence provenance bundle.

This builder does not train a model or recompute predictions. It copies and
hash-checks completed evidence, then creates distinct records for every
prediction mechanism and evaluation domain used by the formal claims.
"""

from __future__ import annotations

import csv
import ast
import hashlib
import json
import mimetypes
import shutil
import struct
import subprocess
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
LATEST = ROOT.parent / "latest_allen_cahn_evidence"
CONFIRM = REPO / "RK4_Publication_Study/allen_cahn_confirmatory_adaptive_20261118"
LONG = REPO / "RK4_Publication_Study/allen_cahn_long_sequence_development_20261119"

PACKAGE_VERSION = "2.0.0"
CREATED_UTC = "2026-11-25T00:00:00+00:00"
DATE_PUBLISHED = "2026-11-25"
CONFIRM_SEEDS = [5101, 5113, 5147, 5167, 5191, 5227, 5261, 5281, 5309, 5333]
LONG_SEEDS = [6101, 6131, 6173]
ARMS = ["zero_rk4", "fixed_rate1", "grouped_rate1"]
SCENARIOS = ["blocks_04", "blocks_08", "blocks_12"]


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def npz_vector_summary(path: Path, member: str = "x") -> dict:
    """Read one little/big-endian float vector from an NPZ using stdlib only."""
    with zipfile.ZipFile(path) as archive:
        payload = archive.read(f"{member}.npy")
    if payload[:6] != b"\x93NUMPY":
        raise RuntimeError(f"invalid NPY member: {path}:{member}")
    major = payload[6]
    if major == 1:
        header_length = struct.unpack("<H", payload[8:10])[0]
        header_start = 10
    elif major in {2, 3}:
        header_length = struct.unpack("<I", payload[8:12])[0]
        header_start = 12
    else:
        raise RuntimeError(f"unsupported NPY version {major}: {path}:{member}")
    header = ast.literal_eval(payload[header_start:header_start + header_length].decode("latin1").strip())
    shape = tuple(header["shape"])
    if len(shape) != 1 or header.get("fortran_order"):
        raise RuntimeError(f"expected one-dimensional C-order vector: {path}:{member}")
    dtype = header["descr"]
    formats = {"<f8": "<d", ">f8": ">d", "<f4": "<f", ">f4": ">f", "|f8": "=d", "|f4": "=f"}
    if dtype not in formats:
        raise RuntimeError(f"unsupported vector dtype {dtype}: {path}:{member}")
    item_size = 8 if dtype.endswith("8") else 4
    raw = payload[header_start + header_length:]
    if len(raw) != shape[0] * item_size:
        raise RuntimeError(f"NPY payload length mismatch: {path}:{member}")
    values = [value[0] for value in struct.iter_unpack(formats[dtype], raw)]
    if len(values) < 2:
        raise RuntimeError(f"spatial vector is too short: {path}:{member}")
    spacing = float(values[1] - values[0])
    tolerance = 1e-11 * max(1.0, abs(spacing))
    if any(abs((values[index] - values[index - 1]) - spacing) > tolerance for index in range(2, len(values))):
        raise RuntimeError(f"spatial vector is not uniform: {path}:{member}")
    return {
        "lower_bound": float(values[0]),
        "last_coordinate": float(values[-1]),
        "upper_bound_exclusive": float(values[0] + spacing * len(values)),
        "points": len(values),
        "spacing": spacing,
        "period": float(spacing * len(values)),
        "endpoint_duplicate_excluded": True,
    }


def media_type(path: Path) -> str:
    if path.suffix == ".json":
        return "application/json"
    if path.suffix == ".jsonld":
        return "application/ld+json"
    if path.suffix == ".md":
        return "text/markdown"
    if path.suffix == ".csv":
        return "text/csv"
    if path.suffix == ".py":
        return "text/x-python"
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def identity(label: str, record_type: str, role: str) -> dict:
    return {"label": label, "type": record_type, "role": role}


def domain(label: str, start: float, end: float, closure: str = "closed") -> dict:
    return {"label": label, "start": float(start), "end": float(end), "closure": closure}


def close(a: float, b: float, tol: float = 1e-12) -> bool:
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)), abs(float(b)))


def copy_file(src: Path, relative: str) -> Path:
    if not src.is_file():
        raise FileNotFoundError(src)
    dst = ROOT / relative
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def add_packaged_artifact(artifacts: list[dict], artifact_id: str, relative: str, role: str, description: str) -> str:
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    artifacts.append(
        {
            "id": artifact_id,
            "identity": identity(path.name, "artifact", role),
            "media_type": media_type(path),
            "path": relative,
            "path_scope": "packaged",
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "description": description,
        }
    )
    return artifact_id


def add_external_artifact(artifacts: list[dict], artifact_id: str, path: Path, role: str, description: str, expected_sha256: str) -> str:
    actual = sha256(path)
    if actual != expected_sha256:
        raise RuntimeError(f"source hash mismatch: {path}: {actual} != {expected_sha256}")
    relative = path.relative_to(REPO).as_posix()
    artifacts.append(
        {
            "id": artifact_id,
            "identity": identity(path.name, "artifact", role),
            "media_type": media_type(path),
            "path": f"../../../{relative}",
            "path_scope": "external_workspace",
            "sha256": actual,
            "bytes": path.stat().st_size,
            "description": description,
        }
    )
    return artifact_id


def generation(generation_id: str, label: str, role: str, inputs: list[str], outputs: list[str], generator_id: str, generator_digest: str, activity_id: str, qualification: str, run_id: str | None = None) -> dict:
    value = {
        "id": generation_id,
        "identity": identity(label, "generation", role),
        "input_ids": list(dict.fromkeys(inputs)),
        "output_ids": list(dict.fromkeys(outputs)),
        "generator_id": generator_id,
        "generator_digest": generator_digest,
        "package_version": PACKAGE_VERSION,
        "prov_activity_id": activity_id,
        "qualification": qualification,
    }
    if run_id is not None:
        value["run_id"] = run_id
    return value


def write_ro_crate(created_utc: str) -> None:
    excluded = {"ro-crate-metadata.json", "ro-crate-metadata.jsonld", "validation_report.json"}
    files = sorted(path for path in ROOT.rglob("*") if path.is_file() and path.relative_to(ROOT).as_posix() not in excluded and "__pycache__" not in path.parts)
    graph = [
        {
            "@id": "ro-crate-metadata.json",
            "@type": "CreativeWork",
            "about": {"@id": "./"},
            "conformsTo": {"@id": "https://w3id.org/ro/crate/1.3"},
        },
        {
            "@id": "./",
            "@type": "Dataset",
            "name": "Allen--Cahn adaptive RK4 evidence provenance bundle",
            "description": "Hash-bound C1--C4 audit over 30 locked confirmatory and 27 long-sequence development runs.",
            "dateCreated": created_utc,
            "datePublished": DATE_PUBLISHED,
            "license": {"@id": "#license-unassigned"},
            "conformsTo": [{"@id": "https://w3id.org/ro/crate/1.3"}],
            "hasPart": [{"@id": path.relative_to(ROOT).as_posix()} for path in files],
        },
        {
            "@id": "#license-unassigned",
            "@type": "CreativeWork",
            "name": "Licence not assigned",
            "description": "No reuse licence has been assigned to this evidence bundle. This metadata statement grants no reuse rights; source components retain their own terms.",
        },
    ]
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        file_entity = {"@id": relative, "@type": "File", "name": path.name, "encodingFormat": media_type(path), "contentSize": str(path.stat().st_size), "identifier": f"sha256:{sha256(path)}"}
        if relative == "allen_cahn_audit_record.json":
            file_entity["description"] = "Custom DIM audit record containing a W3C PROV class/relation projection; this RO-Crate root does not assert PROV conformance."
        graph.append(file_entity)
    crate = {"@context": "https://w3id.org/ro/crate/1.3/context", "@graph": graph}
    content = json.dumps(crate, indent=2) + "\n"
    (ROOT / "ro-crate-metadata.json").write_text(content, encoding="utf-8")
    (ROOT / "ro-crate-metadata.jsonld").write_text(content, encoding="utf-8")


def main() -> None:
    for relative in ("evidence", "evaluations"):
        target = ROOT / relative
        if target.exists():
            shutil.rmtree(target)

    artifacts: list[dict] = []
    configurations: list[dict] = []
    spatial_domains: list[dict] = []
    controller_traces: list[dict] = []
    runs: list[dict] = []
    checkpoints: list[dict] = []
    predictions: list[dict] = []
    evaluations: list[dict] = []
    metrics: list[dict] = []
    claims: list[dict] = []
    generations: list[dict] = []

    core_sources = [
        (LATEST / "inputs/confirmation_PROTOCOL.md", "evidence/confirmation_PROTOCOL.md", "artifact:protocol:confirmatory", "protocol", "Locked ten-seed confirmatory protocol."),
        (LATEST / "inputs/confirmation_manifest.json", "evidence/confirmation_manifest.json", "artifact:manifest:confirmatory", "manifest", "Locked confirmatory run matrix and source hashes."),
        (LATEST / "inputs/confirmation_summary.json", "evidence/confirmation_summary.json", "artifact:summary:confirmatory", "summary", "Complete 30-run confirmatory summary."),
        (LATEST / "inputs/selection_lock.json", "evidence/selection_lock.json", "artifact:selection-lock", "selection lock", "Pre-test method-selection and contrast lock."),
        (LATEST / "inputs/long_sequence_PROTOCOL.md", "evidence/long_sequence_PROTOCOL.md", "artifact:protocol:long", "protocol", "Locked three-seed long-sequence protocol."),
        (LATEST / "inputs/long_sequence_manifest.json", "evidence/long_sequence_manifest.json", "artifact:manifest:long", "manifest", "Locked 27-run long-sequence matrix and source hashes."),
        (LATEST / "inputs/long_sequence_summary.json", "evidence/long_sequence_summary.json", "artifact:summary:long", "summary", "Complete 27-run development summary."),
        (LATEST / "inputs/controller_mechanism_evidence.json", "evidence/controller_mechanism_evidence.json", "artifact:controller-diagnostic", "diagnostic evidence", "Controller trace diagnostic over ten grouped runs."),
        (LATEST / "generated/dim_evidence_summary.json", "evidence/dim_evidence_summary.json", "artifact:dim-summary", "summary", "Compact evidence summary used by the DIM manuscript."),
        (LATEST / "generated/table_confirmatory_grouped_vs_fixed.tex", "evidence/generated/table_confirmatory_grouped_vs_fixed.tex", "artifact:table:confirmatory", "generated table", "Grouped-versus-fixed confirmatory table."),
        (LATEST / "generated/figure_confirmatory_paired.pdf", "evidence/generated/figure_confirmatory_paired.pdf", "artifact:figure:confirmatory:pdf", "generated figure", "Paired confirmatory PDF figure."),
        (LATEST / "generated/figure_confirmatory_paired.png", "evidence/generated/figure_confirmatory_paired.png", "artifact:figure:confirmatory:png", "generated figure", "Paired confirmatory PNG figure."),
        (LATEST / "generated/table_long_sequence.tex", "evidence/generated/table_long_sequence.tex", "artifact:table:long", "generated table", "Long-sequence development table."),
        (LATEST / "generated/figure_long_sequence_effects.pdf", "evidence/generated/figure_long_sequence_effects.pdf", "artifact:figure:long:pdf", "generated figure", "Long-sequence PDF figure."),
        (LATEST / "generated/figure_long_sequence_effects.png", "evidence/generated/figure_long_sequence_effects.png", "artifact:figure:long:png", "generated figure", "Long-sequence PNG figure."),
        (LATEST / "generated/table_controller_mechanism.tex", "evidence/generated/table_controller_mechanism.tex", "artifact:table:controller", "generated table", "Controller diagnostic table."),
        (LATEST / "generated/figure_controller_mechanism.pdf", "evidence/generated/figure_controller_mechanism.pdf", "artifact:figure:controller:pdf", "generated figure", "Controller diagnostic PDF figure."),
        (LATEST / "generated/figure_controller_mechanism.png", "evidence/generated/figure_controller_mechanism.png", "artifact:figure:controller:png", "generated figure", "Controller diagnostic PNG figure."),
    ]
    for src, relative, artifact_id, role, description in core_sources:
        copy_file(src, relative)
        add_packaged_artifact(artifacts, artifact_id, relative, role, description)

    source_registry = load(LATEST / "evidence_sources.json")
    portable_sources = []
    for index, source in enumerate(source_registry["sources"]):
        origin = REPO / source["source_path"]
        relative = f"evidence/source_registry/{index:02d}_{Path(source['copied_path']).name}"
        copied = copy_file(origin, relative)
        if sha256(copied) != source["sha256"] or copied.stat().st_size != source["bytes"]:
            raise RuntimeError(f"source registry mismatch for {origin}")
        artifact_id = f"artifact:source-registry:{index:02d}"
        add_packaged_artifact(artifacts, artifact_id, relative, "registered source", f"Portable copy of registered source {source['source_path']}.")
        portable_sources.append({"source_path": source["source_path"], "source_path_scope": "repository_relative", "bundle_path": relative, "sha256": source["sha256"], "bytes": source["bytes"], "artifact_id": artifact_id})
    portable_registry = {"schema_version": 2, "created_utc": source_registry["created_utc"], "purpose": source_registry["purpose"], "path_base": "provenance_root", "sources": portable_sources}
    dump(ROOT / "evidence/evidence_sources.json", portable_registry)
    add_packaged_artifact(artifacts, "artifact:evidence-register", "evidence/evidence_sources.json", "source register", "Portable 17-file source registry.")

    generator_sources = [
        (CONFIRM / "source_snapshot/GPT-code-invloss.py", "evidence/generators/confirmatory_training.py", "artifact:generator:confirmatory-training", "training generator"),
        (CONFIRM / "evaluate.py", "evidence/generators/confirmatory_evaluator.py", "artifact:generator:confirmatory-evaluator", "evaluation generator"),
        (LONG / "source_snapshot/GPT-code-invloss.py", "evidence/generators/long_training.py", "artifact:generator:long-training", "training generator"),
        (LONG / "evaluate_run.py", "evidence/generators/long_evaluator.py", "artifact:generator:long-evaluator", "evaluation generator"),
        (LATEST / "generate_and_audit.py", "evidence/generators/evidence_synthesizer.py", "artifact:generator:evidence-synthesis", "evidence generator"),
    ]
    for src, relative, artifact_id, role in generator_sources:
        copy_file(src, relative)
        add_packaged_artifact(artifacts, artifact_id, relative, role, f"Hash-bound {role} source.")
    add_packaged_artifact(artifacts, "artifact:generator:provenance-builder", "build_provenance.py", "provenance generator", "Builder for this audit package.")
    for artifact_id, relative, role in [
        ("artifact:profile-schema", "audit_record.schema.json", "application-profile schema"),
        ("artifact:validator:profile", "profile_validation.py", "generic profile validator"),
        ("artifact:validator:study", "validate_provenance.py", "study bundle validator"),
        ("artifact:query:claims", "query_claim_evidence.py", "claim and competency query program"),
        ("artifact:audit-comparison:program", "audit_comparison.py", "audit comparison program"),
        ("artifact:audit-comparison:specification", "audit_comparison_spec.json", "frozen audit comparison specification"),
    ]:
        add_packaged_artifact(artifacts, artifact_id, relative, role, f"Hash-bound {role}; validation does not establish scientific truth.")


    confirm_summary = load(LATEST / "inputs/confirmation_summary.json")
    confirm_manifest = load(LATEST / "inputs/confirmation_manifest.json")
    long_summary = load(LATEST / "inputs/long_sequence_summary.json")
    long_manifest = load(LATEST / "inputs/long_sequence_manifest.json")
    controller = load(LATEST / "inputs/controller_mechanism_evidence.json")
    confirm_rows = {(int(row["seed"]), row["arm"]): row for row in confirm_summary["rows"]}
    long_rows = {(row["scenario"], int(row["seed"]), row["arm"]): row for row in long_summary["rows"]}

    confirm_references = {}
    for split, filename in [("training", "reference_training.npz"), ("validation", "reference_validation_only.npz"), ("test", "reference_test_only.npz")]:
        artifact_id = f"artifact:reference:confirmatory:{split}"
        add_external_artifact(artifacts, artifact_id, CONFIRM / f"reference/{filename}", "reference field", f"Locked confirmatory {split} reference field.", confirm_manifest["source_hashes"][f"reference/{filename}"])
        confirm_references[split] = artifact_id
    long_reference_id = "artifact:reference:long:truth"
    add_external_artifact(artifacts, long_reference_id, LONG / "reference/reference_truth.npz", "reference field", "Locked long-sequence reference field.", long_manifest["source_hashes"]["reference/reference_truth.npz"])

    confirm_direct_domain_id = "spatial-domain:confirmatory:direct"
    confirm_continuation_domain_id = "spatial-domain:confirmatory:continuation"
    long_direct_domain_id = "spatial-domain:long-sequence:direct"
    long_continuation_domain_id = "spatial-domain:long-sequence:continuation"

    def add_role_records(*, base: str, run_id: str, checkpoint_id: str, full_prediction_id: str, role: str, label: str, mechanism: str, interval: dict, native_spatial_domain_id: str, evaluation_spatial_domain_id: str, output_field: str, output_artifact_id: str | None, reference_artifact_id: str, value: float, source_artifact_id: str, source_pointer: str, evaluator_id: str, evaluator_digest: str, qualification: str, aggregation: str = "space-time aggregate norm ratio", upstream_prediction_id: str | None = None, prediction_status: str = "complete") -> tuple[str, str]:
        prediction_id = f"prediction:{base}:{role}"
        evaluation_id = f"evaluation:{base}:{role}"
        prediction_generation_id = f"generation:prediction:{base}:{role}"
        evaluation_generation_id = f"generation:evaluation:{base}:{role}"
        predictions.append(
            {
                "id": prediction_id,
                "identity": identity(label, "prediction", role),
                "run_id": run_id,
                "checkpoint_id": checkpoint_id,
                "mechanism": mechanism,
                "temporal_domain": interval,
                "native_spatial_domain_id": native_spatial_domain_id,
                "evaluation_spatial_domain_id": evaluation_spatial_domain_id,
                "output_field": output_field,
                "output_artifact_id": output_artifact_id,
                "source_activity_id": prediction_generation_id,
                "evaluation_ids": [evaluation_id],
                "status": prediction_status,
                "qualification": qualification,
            }
        )
        evaluations.append(
            {
                "id": evaluation_id,
                "identity": identity(f"Relative L2 evaluation: {label}", "evaluation", role),
                "run_id": run_id,
                "prediction_id": prediction_id,
                "reference_artifact_id": reference_artifact_id,
                "interval": interval,
                "metric": "relative_l2",
                "aggregation": aggregation,
                "value": float(value),
                "source_artifact_id": source_artifact_id,
                "source_pointer": source_pointer,
                "source_activity_id": evaluation_generation_id,
                "qualification": qualification,
            }
        )
        generations.append(
            generation(
                prediction_generation_id,
                f"Materialize {label}",
                "prediction",
                [checkpoint_id, upstream_prediction_id or full_prediction_id, native_spatial_domain_id, evaluation_spatial_domain_id] + ([output_artifact_id] if output_artifact_id else []),
                [prediction_id],
                evaluator_id,
                evaluator_digest,
                f"activity:{prediction_generation_id}",
                qualification,
                run_id,
            )
        )
        generations.append(
            generation(
                evaluation_generation_id,
                f"Evaluate {label}",
                "evaluation",
                [checkpoint_id, prediction_id, reference_artifact_id, source_artifact_id],
                [evaluation_id],
                evaluator_id,
                evaluator_digest,
                f"activity:{evaluation_generation_id}",
                qualification,
                run_id,
            )
        )
        return prediction_id, evaluation_id

    confirm_metric_names = ["direct_training_relative_l2", "direct_validation_relative_l2", "direct_test_relative_l2", "continuation_test_relative_l2", "terminal_relative_l2"]
    for seed in CONFIRM_SEEDS:
        for arm in ARMS:
            source_dir = CONFIRM / f"runs/seed_{seed}/{arm}"
            relative_dir = f"evaluations/confirmatory/seed_{seed}/{arm}"
            source_files = {
                "metrics": source_dir / "metrics.json",
                "record": source_dir / "record.json",
                "completion": source_dir / "output/completion.json",
                "controlled_run": source_dir / "output/controlled_run.json",
            }
            if arm == "grouped_rate1":
                source_files["diagnostics"] = source_dir / "output/diagnostics.json"
            values = {name: load(path) for name, path in source_files.items()}
            row = confirm_rows[(seed, arm)]
            for name in confirm_metric_names:
                if not close(values["metrics"][name], row[name]):
                    raise RuntimeError(f"confirmatory metric mismatch: {seed} {arm} {name}")
            copied_ids = {}
            for name, src in source_files.items():
                relative = f"{relative_dir}/{name}.json"
                copy_file(src, relative)
                artifact_id = f"artifact:{name}:confirmatory:{seed}:{arm}"
                add_packaged_artifact(artifacts, artifact_id, relative, f"confirmatory {name}", f"{name.replace('_', ' ').title()} for confirmatory seed {seed}, arm {arm}.")
                copied_ids[name] = artifact_id

            prediction_path = source_dir / "output/predictions.npz"
            prediction_sha = values["metrics"]["provenance"]["prediction_sha256"]
            raw_prediction_id = f"artifact:prediction-array:confirmatory:{seed}:{arm}"
            add_external_artifact(artifacts, raw_prediction_id, prediction_path, "prediction array", f"Raw direct prediction arrays for confirmatory seed {seed}, arm {arm}.", prediction_sha)
            continuation_fields_id = f"artifact:continuation-fields:confirmatory:{seed}:{arm}"
            add_external_artifact(
                artifacts,
                continuation_fields_id,
                source_dir / "evaluation/fields.npz",
                "evaluated continuation field",
                f"Direct and DOP853-continuation fields interpolated to the locked 1024-node evaluation grid for confirmatory seed {seed}, arm {arm}.",
                values["metrics"]["provenance"]["fields_sha256"],
            )
            controlled = values["controlled_run"]
            config_digest = canonical_digest(controlled["config"])
            config_id = f"config:{config_digest}"
            base = f"confirmatory:{seed}:{arm}"
            run_id = f"run:{base}"
            checkpoint_id = f"checkpoint:{base}:final"
            full_prediction_id = f"prediction:{base}:direct-full-series"
            training_generation_id = f"generation:training:{base}"
            full_prediction_generation_id = f"generation:prediction:{base}:direct-full-series"

            configurations.append(
                {
                    "id": config_id,
                    "identity": identity(f"Configuration: confirmatory seed {seed}, {arm}", "configuration", "training configuration"),
                    "campaign": "allen_cahn_confirmatory_adaptive_20261118",
                    "run_id": run_id,
                    "digest": config_digest,
                    "source_artifact_id": copied_ids["controlled_run"],
                    "source_pointer": "/config",
                    "qualification": "Canonical SHA-256 of the exact controlled-run configuration object; machine-specific paths were replaced deterministically by repository-relative placeholders in the public record.",
                }
            )
            config_generation_id = f"generation:configuration:{base}"
            generations.append(
                generation(
                    config_generation_id,
                    f"Register configuration for confirmatory seed {seed}, {arm}",
                    "configuration extraction",
                    [copied_ids["controlled_run"]],
                    [config_id],
                    "artifact:generator:provenance-builder",
                    sha256(ROOT / "build_provenance.py"),
                    f"activity:{config_generation_id}",
                    "Exact configuration object extracted by JSON Pointer and identified by canonical SHA-256.",
                    run_id,
                )
            )

            checkpoints.append(
                {
                    "id": checkpoint_id,
                    "identity": identity(f"Final iterate: confirmatory seed {seed}, {arm}", "checkpoint", "final iterate"),
                    "run_id": run_id,
                    "selection": "final_iterate",
                    "checkpoint_file_sha256": None,
                    "parameter_state_sha256": controlled["final_parameters_sha256"],
                    "parameter_state_source_artifact_id": copied_ids["controlled_run"],
                    "parameter_state_source_pointer": "/final_parameters_sha256",
                    "qualification": "The controlled-run record supplies the canonical parameter-state SHA-256. No serialized checkpoint-file hash is available and none is asserted.",
                }
            )
            predictions.append(
                {
                    "id": full_prediction_id,
                    "identity": identity(f"Direct full series: confirmatory seed {seed}, {arm}", "prediction", "direct full-series"),
                    "run_id": run_id,
                    "checkpoint_id": checkpoint_id,
                    "mechanism": values["metrics"]["prediction_mechanism"],
                    "temporal_domain": domain("full evaluated series", 0.0, 1.2),
                    "native_spatial_domain_id": confirm_direct_domain_id,
                    "evaluation_spatial_domain_id": confirm_direct_domain_id,
                    "output_field": "direct_prediction",
                    "output_artifact_id": raw_prediction_id,
                    "source_activity_id": full_prediction_generation_id,
                    "evaluation_ids": [],
                    "status": "complete",
                    "qualification": "Full direct final-network field stored in predictions.npz; role-specific evaluations are represented separately.",
                }
            )
            generations.append(
                generation(
                    training_generation_id,
                    f"Train confirmatory seed {seed}, {arm}",
                    "training",
                    ["protocol:confirmatory", config_id, "artifact:generator:confirmatory-training"],
                    [checkpoint_id],
                    "artifact:generator:confirmatory-training",
                    confirm_manifest["source_hashes"]["source_snapshot/GPT-code-invloss.py"],
                    f"activity:{run_id}",
                    "Training generation bound to the frozen protocol, canonical configuration digest, final parameter-state digest and raw prediction-array digest.",
                    run_id,
                )
            )
            generations.append(
                generation(
                    full_prediction_generation_id,
                    f"Materialize direct full series for confirmatory seed {seed}, {arm}",
                    "prediction",
                    [checkpoint_id, confirm_direct_domain_id, raw_prediction_id],
                    [full_prediction_id],
                    "artifact:generator:confirmatory-training",
                    confirm_manifest["source_hashes"]["source_snapshot/GPT-code-invloss.py"],
                    f"activity:{full_prediction_generation_id}",
                    "The full direct field is bound to the final checkpoint and hash-verified raw prediction array.",
                    run_id,
                )
            )

            prediction_ids = [full_prediction_id]
            evaluation_ids = []
            role_specs = [
                ("training", "Direct training interval", values["metrics"]["prediction_mechanism"], domain("training", 0.0, 0.8), confirm_direct_domain_id, confirm_direct_domain_id, "direct_prediction[:, training interval]", raw_prediction_id, confirm_references["training"], row["direct_training_relative_l2"], "/direct_training_relative_l2", "Direct final-network field restricted to the locked training interval.", "space-time aggregate norm ratio"),
                ("validation", "Direct validation interval", values["metrics"]["prediction_mechanism"], domain("validation", 0.8, 1.0, "left-open-right-closed"), confirm_direct_domain_id, confirm_direct_domain_id, "direct_prediction[:, validation interval]", raw_prediction_id, confirm_references["validation"], row["direct_validation_relative_l2"], "/direct_validation_relative_l2", "Held-out validation interval evaluated after all training.", "space-time aggregate norm ratio"),
                ("direct-test", "Direct test interval", values["metrics"]["prediction_mechanism"], domain("test", 1.0, 1.2, "left-open-right-closed"), confirm_direct_domain_id, confirm_direct_domain_id, "direct_prediction[:, test interval]", raw_prediction_id, confirm_references["test"], row["direct_test_relative_l2"], "/direct_test_relative_l2", "Held-out direct final-network test interval evaluated after all training.", "space-time aggregate norm ratio"),
                ("numerical-continuation-test", "Numerical continuation test interval", "DOP853 continuation from the native final-network FD-grid state at t=0.8, periodically interpolated to the reference grid for scoring", domain("test", 1.0, 1.2, "left-open-right-closed"), confirm_continuation_domain_id, confirm_direct_domain_id, "continued_prediction", continuation_fields_id, confirm_references["test"], row["continuation_test_relative_l2"], "/continuation_test_relative_l2", "The evaluator integrated on the native 256-node periodic FD grid, interpolated periodically to the 1024-node reference grid, and persisted the evaluated continuation field in evaluation/fields.npz.", "space-time aggregate norm ratio"),
                ("terminal-state", "Terminal state", values["metrics"]["prediction_mechanism"], domain("training terminal", 0.8, 0.8, "point"), confirm_direct_domain_id, confirm_direct_domain_id, "terminal_state", raw_prediction_id, confirm_references["training"], row["terminal_relative_l2"], "/terminal_relative_l2", "Final-network state at the terminal point of the training interval.", "spatial norm ratio"),
            ]
            for spec in role_specs:
                prediction_id, evaluation_id = add_role_records(
                    base=base,
                    run_id=run_id,
                    checkpoint_id=checkpoint_id,
                    full_prediction_id=full_prediction_id,
                    role=spec[0],
                    label=f"{spec[1]}: seed {seed}, {arm}",
                    mechanism=spec[2],
                    interval=spec[3],
                    native_spatial_domain_id=spec[4],
                    evaluation_spatial_domain_id=spec[5],
                    output_field=spec[6],
                    output_artifact_id=spec[7],
                    reference_artifact_id=spec[8],
                    value=spec[9],
                    source_artifact_id=copied_ids["metrics"],
                    source_pointer=spec[10],
                    evaluator_id="artifact:generator:confirmatory-evaluator",
                    evaluator_digest=confirm_manifest["source_hashes"]["evaluate.py"],
                    qualification=spec[11],
                    aggregation=spec[12],
                    upstream_prediction_id=f"prediction:{base}:terminal-state" if spec[0] == "numerical-continuation-test" else None,
                )
                prediction_ids.append(prediction_id)
                evaluation_ids.append(evaluation_id)
            runs.append(
                {
                    "id": run_id,
                    "identity": identity(f"Confirmatory seed {seed}, {arm}", "run", "training run"),
                    "campaign": "allen_cahn_confirmatory_adaptive_20261118",
                    "protocol_id": "protocol:confirmatory",
                    "arm": arm,
                    "seed": seed,
                    "block_count": len(controlled["block_records"]),
                    "update_count": int(controlled["completed_updates"]),
                    "nonfinite_updates": int(controlled["nonfinite_updates"]),
                    "clipped_gradient_updates": int(controlled["clipped_gradient_updates"]),
                    "config_id": config_id,
                    "config_digest": config_digest,
                    "started_utc": values["record"]["started_utc"],
                    "training_finished_utc": values["record"]["training_finished_utc"],
                    "training_seconds": float(values["record"]["training_seconds"]),
                    "timing_source": {
                        "artifact_id": copied_ids["record"],
                        "started_utc_pointer": "/started_utc",
                        "training_finished_utc_pointer": "/training_finished_utc",
                        "training_seconds_pointer": "/training_seconds",
                    },
                    "checkpoint_id": checkpoint_id,
                    "prediction_ids": prediction_ids,
                    "evaluation_ids": evaluation_ids,
                    "status": "complete",
                    "unavailable_prediction_roles": [],
                    "qualification": "All required confirmatory prediction roles are represented; the native-grid continuation and its periodically interpolated evaluation field are distinguished explicitly.",
                }
            )

    long_metric_names = ["direct_training_relative_l2", "direct_terminal_relative_l2", "direct_future_0.8_relative_l2", "continuation_future_0.8_relative_l2"]
    for scenario in SCENARIOS:
        for seed in LONG_SEEDS:
            for arm in ARMS:
                source_dir = LONG / f"runs/{scenario}/seed_{seed}/{arm}"
                relative_dir = f"evaluations/long_sequence/{scenario}/seed_{seed}/{arm}"
                source_files = {
                    "metrics": source_dir / "metrics.json",
                    "record": source_dir / "record.json",
                    "completion": source_dir / "output/completion.json",
                    "controlled_run": source_dir / "output/controlled_run.json",
                }
                values = {name: load(path) for name, path in source_files.items()}
                row = long_rows[(scenario, seed, arm)]
                for name in long_metric_names:
                    if not close(values["metrics"][name], row[name]):
                        raise RuntimeError(f"long-sequence metric mismatch: {scenario} {seed} {arm} {name}")
                copied_ids = {}
                for name, src in source_files.items():
                    relative = f"{relative_dir}/{name}.json"
                    copy_file(src, relative)
                    artifact_id = f"artifact:{name}:long:{scenario}:{seed}:{arm}"
                    add_packaged_artifact(artifacts, artifact_id, relative, f"long-sequence {name}", f"{name.replace('_', ' ').title()} for {scenario}, seed {seed}, arm {arm}.")
                    copied_ids[name] = artifact_id

                controlled = values["controlled_run"]
                raw_prediction_id = f"artifact:prediction-array:long:{scenario}:{seed}:{arm}"
                add_external_artifact(artifacts, raw_prediction_id, source_dir / "output/predictions.npz", "prediction array", f"Raw direct prediction arrays for {scenario}, seed {seed}, arm {arm}.", values["completion"]["prediction_sha256"])
                config_digest = canonical_digest(controlled["config"])
                config_id = f"config:{config_digest}"
                base = f"long:{scenario}:{seed}:{arm}"
                run_id = f"run:{base}"
                checkpoint_id = f"checkpoint:{base}:final"
                full_prediction_id = f"prediction:{base}:direct-full-series"
                training_generation_id = f"generation:training:{base}"
                full_prediction_generation_id = f"generation:prediction:{base}:direct-full-series"
                train_end = float(values["metrics"]["train_end"])
                forecast_end = float(values["metrics"]["forecast_end"])
                configurations.append(
                    {
                        "id": config_id,
                        "identity": identity(f"Configuration: {scenario}, seed {seed}, {arm}", "configuration", "training configuration"),
                        "campaign": "allen_cahn_long_sequence_development_20261119",
                        "run_id": run_id,
                        "digest": config_digest,
                        "source_artifact_id": copied_ids["controlled_run"],
                        "source_pointer": "/config",
                        "qualification": "Canonical SHA-256 of the exact controlled-run configuration object; machine-specific paths were replaced deterministically by repository-relative placeholders in the public record.",
                    }
                )
                config_generation_id = f"generation:configuration:{base}"
                generations.append(
                    generation(
                        config_generation_id,
                        f"Register configuration for {scenario}, seed {seed}, {arm}",
                        "configuration extraction",
                        [copied_ids["controlled_run"]],
                        [config_id],
                        "artifact:generator:provenance-builder",
                        sha256(ROOT / "build_provenance.py"),
                        f"activity:{config_generation_id}",
                        "Exact configuration object extracted by JSON Pointer and identified by canonical SHA-256.",
                        run_id,
                    )
                )
                checkpoints.append(
                    {
                        "id": checkpoint_id,
                        "identity": identity(f"Final iterate: {scenario}, seed {seed}, {arm}", "checkpoint", "final iterate"),
                        "run_id": run_id,
                        "selection": "final_iterate",
                        "checkpoint_file_sha256": None,
                        "parameter_state_sha256": controlled["final_parameters_sha256"],
                        "parameter_state_source_artifact_id": copied_ids["controlled_run"],
                        "parameter_state_source_pointer": "/final_parameters_sha256",
                        "qualification": "The controlled-run record supplies the canonical parameter-state SHA-256. No serialized checkpoint-file hash is available and none is asserted.",
                    }
                )
                predictions.append(
                    {
                        "id": full_prediction_id,
                        "identity": identity(f"Direct full series: {scenario}, seed {seed}, {arm}", "prediction", "direct full-series"),
                        "run_id": run_id,
                        "checkpoint_id": checkpoint_id,
                        "mechanism": "direct final-network coordinate queries",
                        "temporal_domain": domain("full evaluated series", 0.0, forecast_end),
                        "native_spatial_domain_id": long_direct_domain_id,
                        "evaluation_spatial_domain_id": long_direct_domain_id,
                        "output_field": "direct_prediction",
                        "output_artifact_id": raw_prediction_id,
                        "source_activity_id": full_prediction_generation_id,
                        "evaluation_ids": [],
                        "status": "complete",
                        "qualification": "Full direct final-network field stored in predictions.npz; role-specific evaluations are represented separately.",
                    }
                )
                generations.append(
                    generation(training_generation_id, f"Train {scenario}, seed {seed}, {arm}", "training", ["protocol:long-sequence", config_id, "artifact:generator:long-training"], [checkpoint_id], "artifact:generator:long-training", long_manifest["source_hashes"]["source_snapshot/GPT-code-invloss.py"], f"activity:{run_id}", "Training generation bound to the locked development protocol, canonical configuration digest and final parameter-state digest.", run_id)
                )
                generations.append(
                    generation(full_prediction_generation_id, f"Materialize direct full series for {scenario}, seed {seed}, {arm}", "prediction", [checkpoint_id, long_direct_domain_id, raw_prediction_id], [full_prediction_id], "artifact:generator:long-training", long_manifest["source_hashes"]["source_snapshot/GPT-code-invloss.py"], f"activity:{full_prediction_generation_id}", "The full direct field is bound to the final checkpoint and hash-verified raw prediction array.", run_id)
                )

                prediction_ids = [full_prediction_id]
                evaluation_ids = []
                role_specs = [
                    ("training", "Direct training interval", "direct final-network coordinate queries", domain("training", 0.0, train_end), long_direct_domain_id, long_direct_domain_id, "direct_prediction[:, training interval]", raw_prediction_id, row["direct_training_relative_l2"], "/direct_training_relative_l2", "Direct final-network field restricted to the sequential training interval.", "space-time aggregate norm ratio"),
                    ("direct-test", "Direct future interval", "direct final-network coordinate queries", domain("future horizon 0.8", train_end, forecast_end, "left-open-right-closed"), long_direct_domain_id, long_direct_domain_id, "direct_prediction[:, future 0.8 interval]", raw_prediction_id, row["direct_future_0.8_relative_l2"], "/direct_future_0.8_relative_l2", "Direct final-network evaluation over the full locked 0.8 future horizon.", "space-time aggregate norm ratio"),
                    ("numerical-continuation-test", "Numerical continuation future interval", "DOP853 continuation on the native final-network 256-node FD grid; the 1024-node reference is periodically interpolated to 256 nodes for scoring", domain("future horizon 0.8", train_end, forecast_end, "left-open-right-closed"), long_continuation_domain_id, long_continuation_domain_id, "continued_prediction (field not persisted)", None, row["continuation_future_0.8_relative_l2"], "/continuation_future_0.8_relative_l2", "The development evaluator integrated and scored on the native 256-node periodic FD grid after periodically interpolating the 1024-node reference field down to 256 nodes; only the aggregate continuation error was persisted.", "space-time aggregate norm ratio"),
                    ("terminal-state", "Terminal state", "direct final-network coordinate query", domain("training terminal", train_end, train_end, "point"), long_direct_domain_id, long_direct_domain_id, "terminal_state", raw_prediction_id, row["direct_terminal_relative_l2"], "/direct_terminal_relative_l2", "Final-network state at the terminal point of the sequential training interval.", "spatial norm ratio"),
                ]
                for spec in role_specs:
                    prediction_id, evaluation_id = add_role_records(
                        base=base,
                        run_id=run_id,
                        checkpoint_id=checkpoint_id,
                        full_prediction_id=full_prediction_id,
                        role=spec[0],
                        label=f"{spec[1]}: {scenario}, seed {seed}, {arm}",
                        mechanism=spec[2],
                        interval=spec[3],
                        native_spatial_domain_id=spec[4],
                        evaluation_spatial_domain_id=spec[5],
                        output_field=spec[6],
                        output_artifact_id=spec[7],
                        reference_artifact_id=long_reference_id,
                        value=spec[8],
                        source_artifact_id=copied_ids["metrics"],
                        source_pointer=spec[9],
                        evaluator_id="artifact:generator:long-evaluator",
                        evaluator_digest=long_manifest["source_hashes"]["evaluate_run.py"],
                        qualification=spec[10],
                        aggregation=spec[11],
                        upstream_prediction_id=f"prediction:{base}:terminal-state" if spec[0] == "numerical-continuation-test" else None,
                        prediction_status="field_not_persisted" if spec[0] == "numerical-continuation-test" else "complete",
                    )
                    prediction_ids.append(prediction_id)
                    evaluation_ids.append(evaluation_id)
                runs.append(
                    {
                        "id": run_id,
                        "identity": identity(f"{scenario}, seed {seed}, {arm}", "run", "training run"),
                        "campaign": "allen_cahn_long_sequence_development_20261119",
                        "protocol_id": "protocol:long-sequence",
                        "arm": arm,
                        "seed": seed,
                        "scenario": scenario,
                        "block_count": len(controlled["block_records"]),
                        "update_count": int(controlled["completed_updates"]),
                        "nonfinite_updates": int(controlled["nonfinite_updates"]),
                        "clipped_gradient_updates": int(controlled["clipped_gradient_updates"]),
                        "config_id": config_id,
                        "config_digest": config_digest,
                        "started_utc": values["record"]["started_utc"],
                        "training_finished_utc": values["record"]["training_finished_utc"],
                        "training_seconds": float(values["record"]["training_seconds"]),
                        "timing_source": {
                            "artifact_id": copied_ids["record"],
                            "started_utc_pointer": "/started_utc",
                            "training_finished_utc_pointer": "/training_finished_utc",
                            "training_seconds_pointer": "/training_seconds",
                        },
                        "checkpoint_id": checkpoint_id,
                        "prediction_ids": prediction_ids,
                        "evaluation_ids": evaluation_ids,
                        "status": "complete",
                        "unavailable_prediction_roles": ["validation"],
                        "qualification": "The development protocol has no separate validation split; validation is explicitly unavailable rather than imputed. Continuation is retained only as a scalar evaluation because its field was not persisted.",
                    }
                )

    if len(configurations) != 57 or len({item["id"] for item in configurations}) != 57:
        raise RuntimeError("expected one distinct canonical configuration entity for each of 57 controlled runs")

    # Spatial domains are explicit application and PROV entities.  Direct
    # predictions use the released reference grid; continuation evaluations
    # use the native half-open finite-difference grid from the terminal state.
    spatial_specs = [
        (
            confirm_direct_domain_id,
            "Confirmatory direct-query spatial grid",
            "allen_cahn_confirmatory_adaptive_20261118",
            npz_vector_summary(CONFIRM / "runs/seed_5101/grouped_rate1/output/predictions.npz", "x"),
            "artifact:prediction-array:confirmatory:5101:grouped_rate1",
            "x",
            "released direct-query and reference grid",
        ),
        (
            confirm_continuation_domain_id,
            "Confirmatory continuation finite-difference grid",
            "allen_cahn_confirmatory_adaptive_20261118",
            npz_vector_summary(CONFIRM / "runs/seed_5101/grouped_rate1/output/predictions.npz", "fd_x"),
            "artifact:prediction-array:confirmatory:5101:grouped_rate1",
            "fd_x",
            "native terminal-state finite-difference grid",
        ),
        (
            long_direct_domain_id,
            "Long-sequence direct-query spatial grid",
            "allen_cahn_long_sequence_development_20261119",
            npz_vector_summary(LONG / "runs/blocks_04/seed_6101/grouped_rate1/output/predictions.npz", "x"),
            "artifact:prediction-array:long:blocks_04:6101:grouped_rate1",
            "x",
            "released direct-query and reference grid",
        ),
        (
            long_continuation_domain_id,
            "Long-sequence continuation finite-difference grid",
            "allen_cahn_long_sequence_development_20261119",
            npz_vector_summary(LONG / "runs/blocks_04/seed_6101/grouped_rate1/output/predictions.npz", "fd_x"),
            "artifact:prediction-array:long:blocks_04:6101:grouped_rate1",
            "fd_x",
            "native terminal-state finite-difference grid",
        ),
    ]
    spatial_source = {
        "schema_version": 1,
        "qualification": "Domain coordinates were read directly from hash-verified prediction NPZ members; every run is checked against these campaign/mechanism domains by validate_provenance.py.",
        "domains": [],
    }
    for domain_id, label, campaign, summary, origin_id, member, role in spatial_specs:
        spatial_source["domains"].append(
            {
                "id": domain_id,
                "campaign": campaign,
                "array_member": member,
                "origin_artifact_id": origin_id,
                **summary,
            }
        )
    dump(ROOT / "evidence/spatial_domain_sources.json", spatial_source)
    add_packaged_artifact(
        artifacts,
        "artifact:spatial-domain-sources",
        "evidence/spatial_domain_sources.json",
        "spatial-domain source",
        "Derived, hash-bound coordinate summaries for direct and numerical-continuation mechanisms.",
    )
    provenance_builder_digest = sha256(ROOT / "build_provenance.py")
    for index, (domain_id, label, campaign, summary, origin_id, member, role) in enumerate(spatial_specs):
        spatial_domains.append(
            {
                "id": domain_id,
                "identity": identity(label, "spatial domain", role),
                "campaign": campaign,
                "coordinate": "x",
                "topology": "periodic_half_open",
                **summary,
                "source_artifact_id": "artifact:spatial-domain-sources",
                "source_pointer": f"/domains/{index}",
                "origin_artifact_ids": [origin_id],
                "origin_array_member": member,
                "qualification": "The right endpoint is the excluded periodic duplicate; the recorded last coordinate is the largest stored coordinate.",
            }
        )
        generation_id = f"generation:spatial-domain:{domain_id}"
        generations.append(
            generation(
                generation_id,
                f"Extract {label}",
                "domain extraction",
                ["artifact:spatial-domain-sources", origin_id],
                [domain_id],
                "artifact:generator:provenance-builder",
                provenance_builder_digest,
                f"activity:{generation_id}",
                "Coordinate metadata extracted from the declared hash-verified NPZ array member.",
            )
        )

    # Controller traces are distinct from predicted fields.  Each record binds
    # one grouped run to its raw adaptive trace, diagnostic JSON and exact
    # configuration before aggregate controller metrics are synthesized.
    run_index = {item["id"]: item for item in runs}
    for seed in CONFIRM_SEEDS:
        run_id = f"run:confirmatory:{seed}:grouped_rate1"
        run = run_index[run_id]
        seed_summary = controller["by_seed"][str(seed)]
        raw_prediction_id = f"artifact:prediction-array:confirmatory:{seed}:grouped_rate1"
        diagnostics_id = f"artifact:diagnostics:confirmatory:{seed}:grouped_rate1"
        artifact_index = {item["id"]: item for item in artifacts}
        if artifact_index[raw_prediction_id]["sha256"] != seed_summary["npz_sha256"]:
            raise RuntimeError(f"controller NPZ hash mismatch for seed {seed}")
        if artifact_index[diagnostics_id]["sha256"] != seed_summary["diagnostics_sha256"]:
            raise RuntimeError(f"controller diagnostics hash mismatch for seed {seed}")
        trace_id = f"controller-trace:confirmatory:{seed}:grouped_rate1"
        controller_traces.append(
            {
                "id": trace_id,
                "identity": identity(f"Grouped controller trace: seed {seed}", "controller trace", "adaptive-weight trajectory"),
                "run_id": run_id,
                "protocol_id": run["protocol_id"],
                "configuration_id": run["config_id"],
                "trace_field": "adaptive_trace",
                "updates": int(seed_summary["updates"]),
                "prediction_array_sha256": seed_summary["npz_sha256"],
                "diagnostics_sha256": seed_summary["diagnostics_sha256"],
                "source_artifact_ids": [raw_prediction_id, diagnostics_id, "artifact:controller-diagnostic"],
                "summary_source_artifact_id": "artifact:controller-diagnostic",
                "summary_source_pointer": f"/by_seed/{seed}",
                "qualification": "Controller-state trajectory and gradient diagnostics from one grouped run; serial updates are descriptive observations, not independent replicates.",
            }
        )
        generation_id = f"generation:controller-trace:confirmatory:{seed}:grouped_rate1"
        generations.append(
            generation(
                generation_id,
                f"Register grouped controller trace for seed {seed}",
                "controller trace extraction",
                [run["protocol_id"], run["config_id"], raw_prediction_id, diagnostics_id, "artifact:controller-diagnostic"],
                [trace_id],
                "artifact:generator:evidence-synthesis",
                sha256(ROOT / "evidence/generators/evidence_synthesizer.py"),
                f"activity:{generation_id}",
                "Trace entity is hash-bound to the raw NPZ adaptive_trace, diagnostic JSON and pooled evidence record.",
                run_id,
            )
        )

    protocols = [
        {
            "id": "protocol:confirmatory",
            "identity": identity("Locked ten-seed Allen--Cahn RK4 weighting comparison", "protocol", "confirmatory protocol"),
            "campaign": "allen_cahn_confirmatory_adaptive_20261118",
            "inferential_status": "locked_confirmatory_primary",
            "protocol_artifact_id": "artifact:protocol:confirmatory",
            "manifest_artifact_id": "artifact:manifest:confirmatory",
            "summary_artifact_id": "artifact:summary:confirmatory",
            "selection_lock_artifact_id": "artifact:selection-lock",
            "expected_runs": 30,
            "primary_outcome": confirm_manifest["primary_outcome"],
            "design": {
                "seeds": CONFIRM_SEEDS,
                "arms": confirm_manifest["arms"],
                "blocks": 4,
                "updates_per_block": 5000,
                "matched_initialization_and_pair_schedule": True,
                "training_blind_to_reference": True,
                "checkpoint_selection": "final_iterate",
            },
            "qualification": "Grouped-versus-fixed was a frozen secondary contrast; the formal DIM graph does not include fixed-versus-zero as a claim.",
        },
        {
            "id": "protocol:long-sequence",
            "identity": identity("Locked three-seed Allen--Cahn long-sequence development study", "protocol", "development protocol"),
            "campaign": "allen_cahn_long_sequence_development_20261119",
            "inferential_status": "development_only",
            "protocol_artifact_id": "artifact:protocol:long",
            "manifest_artifact_id": "artifact:manifest:long",
            "summary_artifact_id": "artifact:summary:long",
            "expected_runs": 27,
            "primary_outcome": "direct_training_relative_l2",
            "design": {
                "seeds": LONG_SEEDS,
                "arms": long_manifest["arms"],
                "scenarios": long_manifest["scenarios"],
                "updates_per_block": 5000,
                "forecast_horizon": 0.8,
                "explicit_causal_loss_weighting": False,
                "checkpoint_selection": "final_iterate",
            },
            "qualification": "Three-seed development evidence only; no confirmatory long-sequence inference is claimed and no validation split exists.",
        },
    ]

    def record_ids(campaign: str, role: str, seeds: list[int], arms: list[str], scenario: str | None = None, prefix: str = "prediction") -> list[str]:
        result = []
        for seed in seeds:
            for arm in arms:
                base = f"{campaign}:{scenario}:{seed}:{arm}" if scenario else f"{campaign}:{seed}:{arm}"
                result.append(f"{prefix}:{base}:{role}")
        return result

    def add_contrast(metric_id: str, metric_name: str, item: dict, status: str, role: str, prediction_ids: list[str], evaluation_ids: list[str], source_artifact_id: str, source_pointer: str, qualification: str) -> None:
        value = {
            "id": metric_id,
            "identity": identity(metric_name, "metric", role),
            "metric_name": metric_name,
            "kind": "paired_mean_difference",
            "value": float(item["paired_mean_difference"]),
            "unit": "relative L2 error",
            "treatment": "grouped_rate1",
            "control": "fixed_rate1",
            "n": int(item.get("n", len(item["paired_differences"]))),
            "inferential_status": status,
            "prediction_ids": prediction_ids,
            "evaluation_ids": evaluation_ids,
            "controller_trace_ids": [],
            "source_artifact_id": source_artifact_id,
            "source_pointer": source_pointer,
            "qualification": qualification,
        }
        if "mean_difference_95pct_t_ci" in item:
            value["ci95"] = [float(number) for number in item["mean_difference_95pct_t_ci"]]
        if "wins_lower_error" in item:
            value["wins_lower_error"] = int(item["wins_lower_error"])
        if "one_sided_exact_sign_flip_p" in item:
            value["one_sided_exact_sign_flip_p"] = float(item["one_sided_exact_sign_flip_p"])
        if item.get("holm_adjusted_one_sided_p") is not None:
            value["holm_adjusted_one_sided_p"] = float(item["holm_adjusted_one_sided_p"])
        metrics.append(value)

    confirm_roles = [
        ("training", "direct_training_relative_l2", "training"),
        ("validation", "direct_validation_relative_l2", "validation"),
        ("direct-test", "direct_test_relative_l2", "test"),
        ("numerical-continuation-test", "continuation_test_relative_l2", "continuation-test"),
        ("terminal-state", "terminal_relative_l2", "terminal"),
    ]
    confirm_metric_ids = {}
    for role, metric_name, short in confirm_roles:
        item = confirm_summary["paired_contrasts"][metric_name]["grouped_rate1_minus_fixed_rate1"]
        metric_id = f"metric:confirmatory:grouped-fixed:{short}"
        confirm_metric_ids[role] = metric_id
        add_contrast(
            metric_id,
            metric_name,
            item,
            "locked_confirmatory_secondary",
            role,
            record_ids("confirmatory", role, CONFIRM_SEEDS, ["fixed_rate1", "grouped_rate1"]),
            record_ids("confirmatory", role, CONFIRM_SEEDS, ["fixed_rate1", "grouped_rate1"], prefix="evaluation"),
            "artifact:summary:confirmatory",
            f"/paired_contrasts/{metric_name}/grouped_rate1_minus_fixed_rate1",
            "Frozen secondary matched contrast; negative values favour grouped weighting. Held-out non-improvement is not an equivalence result." if role != "training" else "Frozen secondary matched contrast; negative values favour grouped weighting.",
        )

    long_metric_ids = []
    for scenario in SCENARIOS:
        item = long_summary["paired_contrasts"][scenario]["grouped_rate1_minus_fixed_rate1"]["direct_training_relative_l2"]
        metric_id = f"metric:long:{scenario}:grouped-fixed:training"
        long_metric_ids.append(metric_id)
        add_contrast(
            metric_id,
            "direct_training_relative_l2",
            item,
            "development_only",
            "training",
            record_ids("long", "training", LONG_SEEDS, ["fixed_rate1", "grouped_rate1"], scenario),
            record_ids("long", "training", LONG_SEEDS, ["fixed_rate1", "grouped_rate1"], scenario, "evaluation"),
            "artifact:summary:long",
            f"/paired_contrasts/{scenario}/grouped_rate1_minus_fixed_rate1/direct_training_relative_l2",
            "Three paired development seeds; direction is descriptive and not confirmatory.",
        )
    trend = long_summary["adaptive_minus_fixed_training_error_trend"]
    all_long_training_predictions = sum((record_ids("long", "training", LONG_SEEDS, ["fixed_rate1", "grouped_rate1"], scenario) for scenario in SCENARIOS), [])
    all_long_training_evaluations = sum((record_ids("long", "training", LONG_SEEDS, ["fixed_rate1", "grouped_rate1"], scenario, "evaluation") for scenario in SCENARIOS), [])
    metrics.append(
        {
            "id": "metric:long:grouped-fixed:training-trend",
            "identity": identity("Training-error difference trend with block count", "metric", "long-sequence trend"),
            "metric_name": "linear_slope_per_block",
            "kind": "linear_trend",
            "value": float(trend["linear_slope_per_block"]),
            "unit": "relative L2 error difference per temporal block",
            "n": 3,
            "inferential_status": "development_only",
            "prediction_ids": all_long_training_predictions,
            "evaluation_ids": all_long_training_evaluations,
            "controller_trace_ids": [],
            "source_artifact_id": "artifact:summary:long",
            "source_pointer": "/adaptive_minus_fixed_training_error_trend/linear_slope_per_block",
            "qualification": trend["qualification"],
        }
    )

    grouped_controller_traces = [f"controller-trace:confirmatory:{seed}:grouped_rate1" for seed in CONFIRM_SEEDS]
    for short, key, unit in [
        ("gradient-cosine", "temporal_pde_gradient_cosine", "cosine similarity"),
        ("weighted-gradient-ratio", "weighted_temporal_to_pde_gradient_ratio", "ratio"),
        ("rk4-budget-share", "physics_rk4_budget_share", "share"),
        ("replay-budget-share", "constraint_previous_budget_share_active_blocks_2_to_4", "share"),
    ]:
        item = controller["pooled_summary"][key]
        metrics.append(
            {
                "id": f"metric:controller:{short}",
                "identity": identity(key, "metric", "controller diagnostic"),
                "metric_name": key,
                "kind": "descriptive_summary",
                "value": float(item["median"]),
                "unit": unit,
                "n": int(item["n"]),
                "interval_kind": "interquartile_range",
                "interval_bounds": [float(item["q25"]), float(item["q75"])],
                "inferential_status": "descriptive_diagnostic",
                "prediction_ids": [],
                "evaluation_ids": [],
                "controller_trace_ids": grouped_controller_traces,
                "source_artifact_id": "artifact:controller-diagnostic",
                "source_pointer": f"/pooled_summary/{key}",
                "qualification": controller["qualification"] + " The stored interval is the interquartile range, not a confidence interval.",
            }
        )
    for short, key, unit in [
        ("physics-budget-max-discrepancy", "maximum_physics_budget_error", "maximum absolute discrepancy"),
        ("constraint-budget-max-discrepancy", "maximum_constraint_budget_error", "maximum absolute discrepancy"),
        ("fixed-anchor-max-discrepancy", "maximum_fixed_anchor_error", "maximum absolute discrepancy"),
        ("guard-trigger-count", "guard_trigger_count", "count"),
    ]:
        metrics.append(
            {
                "id": f"metric:controller:{short}",
                "identity": identity(key, "metric", "budget invariant"),
                "metric_name": key,
                "kind": "budget_invariant",
                "value": float(controller["trace_verification"][key]),
                "unit": unit,
                "n": int(controller["trace_verification"]["total_updates_verified"]),
                "inferential_status": "descriptive_diagnostic",
                "prediction_ids": [],
                "evaluation_ids": [],
                "controller_trace_ids": grouped_controller_traces,
                "source_artifact_id": "artifact:controller-diagnostic",
                "source_pointer": f"/trace_verification/{key}",
                "qualification": "Descriptive audit of the controller trace; a near-zero discrepancy or zero trigger count records the configured budget/safeguard behaviour and is not a predictive-performance claim.",
            }
        )

    heldout_metric_ids = [confirm_metric_ids[role] for role in ["validation", "direct-test", "numerical-continuation-test", "terminal-state"]]
    training_eval_ids = record_ids("confirmatory", "training", CONFIRM_SEEDS, ["fixed_rate1", "grouped_rate1"], prefix="evaluation")
    heldout_eval_ids = sum((record_ids("confirmatory", role, CONFIRM_SEEDS, ["fixed_rate1", "grouped_rate1"], prefix="evaluation") for role in ["validation", "direct-test", "numerical-continuation-test", "terminal-state"]), [])
    controller_metric_ids = [metric["id"] for metric in metrics if metric["id"].startswith("metric:controller:")]
    claims.extend(
        [
            {
                "id": "C1",
                "identity": identity("Grouped adaptive loss weighting lowers final-network fitted-interval retention error", "claim", "confirmatory fitted-interval retention claim"),
                "statement": "Grouped adaptive loss weighting reduced final-network relative L2 retention error over the fitted interval compared with fixed coefficients in all ten paired Allen--Cahn seeds; both arms retained the same positive rate-normalised RK4 term.",
                "direction": "grouped_rate1 minus fixed_rate1 is negative for final-network fitted-interval retention error; both arms use positive rate-normalised RK4",
                "evidence_assessment": "supported",
                "inferential_status": "locked_confirmatory_secondary",
                "supporting_evidence_ids": [confirm_metric_ids["training"]] + training_eval_ids,
                "limiting_evidence_ids": heldout_metric_ids + heldout_eval_ids,
                "supporting_artifact_ids": ["artifact:summary:confirmatory", "artifact:table:confirmatory", "artifact:figure:confirmatory:pdf", "artifact:figure:confirmatory:png"],
                "limiting_artifact_ids": ["artifact:summary:confirmatory", "artifact:table:confirmatory", "artifact:figure:confirmatory:pdf", "artifact:figure:confirmatory:png"],
                "metric_ids": [confirm_metric_ids["training"]],
                "intended_use": "Formal DIM adaptive-controller training result.",
                "qualification": "This is a frozen secondary grouped-versus-fixed contrast. Held-out direct, continuation and terminal evidence limits generalisation of the training result.",
            },
            {
                "id": "C2",
                "identity": identity("No observed held-out reduction", "claim", "held-out boundary claim"),
                "statement": "The confirmatory data did not demonstrate a grouped-over-fixed error reduction on direct validation, direct test, numerical-continuation test or terminal-state evaluation.",
                "direction": "two-sided 95 percent paired t intervals include zero on all four held-out or terminal criteria",
                "evidence_assessment": "qualified",
                "inferential_status": "locked_confirmatory_secondary",
                "supporting_evidence_ids": heldout_metric_ids + heldout_eval_ids,
                "limiting_evidence_ids": [confirm_metric_ids["training"]] + training_eval_ids,
                "supporting_artifact_ids": ["artifact:summary:confirmatory", "artifact:table:confirmatory", "artifact:figure:confirmatory:pdf", "artifact:figure:confirmatory:png"],
                "limiting_artifact_ids": ["artifact:summary:confirmatory"],
                "metric_ids": heldout_metric_ids,
                "intended_use": "Formal scope boundary for the adaptive-controller result.",
                "qualification": "Failure to observe a reduction is not evidence of equivalence or absence of an effect.",
            },
            {
                "id": "C3",
                "identity": identity("No growing long-sequence advantage", "claim", "development boundary claim"),
                "statement": "The three-seed development study did not show growing support for a grouped-over-fixed final-network fitted-interval retention advantage as the number of temporal blocks increased.",
                "direction": "fitted-interval retention-error difference slope is non-negative across 4, 8 and 12 blocks",
                "evidence_assessment": "qualified",
                "inferential_status": "development_only",
                "supporting_evidence_ids": long_metric_ids + ["metric:long:grouped-fixed:training-trend"] + all_long_training_evaluations,
                "limiting_evidence_ids": ["protocol:long-sequence"],
                "supporting_artifact_ids": ["artifact:summary:long", "artifact:table:long", "artifact:figure:long:pdf", "artifact:figure:long:png"],
                "limiting_artifact_ids": ["artifact:protocol:long"],
                "metric_ids": long_metric_ids + ["metric:long:grouped-fixed:training-trend"],
                "intended_use": "Development-only boundary on the proposed long-sequence mechanism.",
                "qualification": "The linked locked protocol records only three paired seeds per block count; this is development evidence rather than confirmatory inference.",
            },
            {
                "id": "C4",
                "identity": identity("Controller budget diagnostics", "claim", "descriptive mechanism claim"),
                "statement": "The grouped controller maintained its configured within-group coefficient budgets while allocating a majority of the active later-block constraint budget to replay, with no recorded guard triggers.",
                "direction": "budget discrepancies are at floating-point scale, guard count is zero, and median later-block replay share exceeds one half",
                "evidence_assessment": "supported",
                "inferential_status": "descriptive_diagnostic",
                "supporting_evidence_ids": controller_metric_ids,
                "limiting_evidence_ids": ["protocol:confirmatory"],
                "supporting_artifact_ids": ["artifact:controller-diagnostic", "artifact:table:controller", "artifact:figure:controller:pdf", "artifact:figure:controller:png"],
                "limiting_artifact_ids": ["artifact:controller-diagnostic"],
                "metric_ids": controller_metric_ids,
                "intended_use": "Descriptive controller-mechanism evidence.",
                "qualification": "Updates within each trace are serially dependent and no mediator was randomized. These diagnostics establish controller behaviour and budget invariants, not improved held-out prediction or a causal mediator effect.",
            },
        ]
    )

    generated_files = sorted(path for path in (ROOT / "evidence/generated").iterdir() if path.is_file())
    hash_lines = [f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}" for path in generated_files]
    (ROOT / "evidence/generated_outputs.sha256").write_text("\n".join(hash_lines) + "\n", encoding="utf-8")
    add_packaged_artifact(artifacts, "artifact:generated-hashes", "evidence/generated_outputs.sha256", "hash manifest", "Portable hashes of the generated manuscript tables and figures.")

    synthesis_digest = sha256(ROOT / "evidence/generators/evidence_synthesizer.py")
    builder_digest = sha256(ROOT / "build_provenance.py")
    for metric in metrics:
        generation_id = f"generation:metric:{metric['id']}"
        generations.append(
            generation(
                generation_id,
                f"Synthesize {metric['identity']['label']}",
                "metric synthesis",
                [metric["source_artifact_id"]] + metric["prediction_ids"] + metric["evaluation_ids"] + metric["controller_trace_ids"],
                [metric["id"]],
                "artifact:generator:evidence-synthesis",
                synthesis_digest,
                f"activity:{generation_id}",
                "Aggregate metric reproduced from the declared JSON Pointer source and linked run-level evidence.",
            )
        )
    for claim in claims:
        generation_id = f"generation:claim:{claim['id']}"
        generations.append(
            generation(
                generation_id,
                f"Assemble claim {claim['id']}",
                "claim synthesis",
                claim["supporting_evidence_ids"] + claim["limiting_evidence_ids"] + claim["supporting_artifact_ids"] + claim["limiting_artifact_ids"],
                [claim["id"]],
                "artifact:generator:provenance-builder",
                builder_digest,
                f"activity:{generation_id}",
                claim["qualification"],
            )
        )
    presentation_groups = [
        (["C1", "C2"], ["artifact:table:confirmatory", "artifact:figure:confirmatory:pdf", "artifact:figure:confirmatory:png"]),
        (["C3"], ["artifact:table:long", "artifact:figure:long:pdf", "artifact:figure:long:png"]),
        (["C4"], ["artifact:table:controller", "artifact:figure:controller:pdf", "artifact:figure:controller:png"]),
    ]
    claim_index = {claim["id"]: claim for claim in claims}
    for claim_ids, artifact_ids in presentation_groups:
        metric_ids = list(dict.fromkeys(metric_id for claim_id in claim_ids for metric_id in claim_index[claim_id]["metric_ids"]))
        for artifact_id in artifact_ids:
            generation_id = f"generation:presentation:{artifact_id}"
            generations.append(
                generation(
                    generation_id,
                    f"Generate presentation artefact {artifact_id}",
                    "presentation",
                    claim_ids + metric_ids,
                    [artifact_id],
                    "artifact:generator:evidence-synthesis",
                    synthesis_digest,
                    f"activity:{generation_id}",
                    "Generated table or figure is explicitly derived from the formal claim and aggregate metric entities listed as inputs.",
                )
            )

    # W3C PROV mapping: application records remain primary, with a parallel
    # Entity/Activity/Agent graph for a PROV-aligned class/relation projection.
    entity_records: list[tuple[str, dict]] = []
    for record_type, collection in [
        ("protocol", protocols),
        ("configuration", configurations),
        ("spatial_domain", spatial_domains),
        ("checkpoint", checkpoints),
        ("prediction", predictions),
        ("evaluation", evaluations),
        ("controller_trace", controller_traces),
        ("metric", metrics),
        ("claim", claims),
        ("artifact", artifacts),
    ]:
        entity_records.extend((record_type, item) for item in collection)
    entities = [
        {
            "id": f"entity:{item['id']}",
            "identity": identity(item["identity"]["label"], "prov:Entity", item["identity"]["role"]),
            "type": "prov:Entity",
            "record_type": record_type,
            "record_id": item["id"],
        }
        for record_type, item in entity_records
    ]
    entity_by_record = {entity["record_id"]: entity["id"] for entity in entities}
    def software_agent(agent_id: str, label: str, role: str, source_artifact_id: str, software_version: str, qualification: str) -> dict:
        source = next(item for item in artifacts if item["id"] == source_artifact_id)
        return {
            "id": agent_id,
            "identity": identity(label, "prov:SoftwareAgent", role),
            "type": "prov:SoftwareAgent",
            "software_version": software_version,
            "source_artifact_id": source_artifact_id,
            "source_digest": source["sha256"],
            "qualification": qualification,
        }

    agents = [
        software_agent("agent:confirmatory-training", "Locked confirmatory Allen--Cahn training source", "training", "artifact:generator:confirmatory-training", "source-snapshot-no-semantic-version", "No semantic version was recorded; the source SHA-256 is the operative identifier."),
        software_agent("agent:long-training", "Locked long-sequence Allen--Cahn training source", "training", "artifact:generator:long-training", "source-snapshot-no-semantic-version", "No semantic version was recorded; the source SHA-256 is the operative identifier."),
        software_agent("agent:confirmatory-evaluator", "Locked confirmatory Allen--Cahn evaluator", "evaluation", "artifact:generator:confirmatory-evaluator", "source-snapshot-no-semantic-version", "No semantic version was recorded; the source SHA-256 is the operative identifier."),
        software_agent("agent:long-evaluator", "Locked long-sequence Allen--Cahn evaluator", "evaluation", "artifact:generator:long-evaluator", "source-snapshot-no-semantic-version", "No semantic version was recorded; the source SHA-256 is the operative identifier."),
        software_agent("agent:evidence-synthesis", "DIM evidence synthesizer", "evidence synthesis", "artifact:generator:evidence-synthesis", "source-snapshot-no-semantic-version", "No semantic version was recorded; the source SHA-256 is the operative identifier."),
        software_agent("agent:provenance-builder", "DIM provenance builder", "provenance assembly", "artifact:generator:provenance-builder", PACKAGE_VERSION, "Package version and source SHA-256 jointly identify this builder."),
    ]
    agent_by_generator = {
        "artifact:generator:confirmatory-training": "agent:confirmatory-training",
        "artifact:generator:long-training": "agent:long-training",
        "artifact:generator:confirmatory-evaluator": "agent:confirmatory-evaluator",
        "artifact:generator:long-evaluator": "agent:long-evaluator",
        "artifact:generator:evidence-synthesis": "agent:evidence-synthesis",
        "artifact:generator:provenance-builder": "agent:provenance-builder",
    }
    activities = []
    relations = []
    activity_ids = set()
    for item in generations:
        activity_id = item["prov_activity_id"]
        if activity_id in activity_ids:
            raise RuntimeError(f"duplicate PROV activity id: {activity_id}")
        activity_ids.add(activity_id)
        role = item["identity"]["role"]
        if role == "training":
            activity_type = "training"
        elif role == "prediction":
            activity_type = "prediction"
        elif role == "evaluation":
            activity_type = "evaluation"
        elif role == "presentation":
            activity_type = "presentation"
        else:
            activity_type = "synthesis"
        agent_id = agent_by_generator[item["generator_id"]]
        activities.append(
            {
                "id": activity_id,
                "identity": identity(item["identity"]["label"], "prov:Activity", role),
                "type": "prov:Activity",
                "activity_type": activity_type,
                "record_id": item["run_id"] if activity_type == "training" else item["id"],
            }
        )
        relations.append({"predicate": "prov:wasAssociatedWith", "subject_id": activity_id, "object_id": agent_id})
        generator_entity = entity_by_record.get(item["generator_id"])
        if generator_entity is not None:
            relations.append({"predicate": "prov:used", "subject_id": activity_id, "object_id": generator_entity})
        for input_id in item["input_ids"]:
            input_entity = entity_by_record.get(input_id)
            if input_entity is not None:
                relations.append({"predicate": "prov:used", "subject_id": activity_id, "object_id": input_entity})
        for output_id in item["output_ids"]:
            output_entity = entity_by_record.get(output_id)
            if output_entity is None:
                raise RuntimeError(f"generation output has no PROV entity: {item['id']} -> {output_id}")
            relations.append({"predicate": "prov:wasGeneratedBy", "subject_id": output_entity, "object_id": activity_id})
            if activity_type in {"prediction", "evaluation", "synthesis", "presentation"}:
                for input_id in item["input_ids"]:
                    input_entity = entity_by_record.get(input_id)
                    if input_entity is not None:
                        relations.append({"predicate": "prov:wasDerivedFrom", "subject_id": output_entity, "object_id": input_entity})
        if activity_type in {"prediction", "evaluation"} and item.get("run_id"):
            relations.append({"predicate": "prov:wasInformedBy", "subject_id": activity_id, "object_id": f"activity:{item['run_id']}"})
    # Deduplicate relations while preserving stable order.
    deduplicated_relations = []
    seen_relations = set()
    for relation in relations:
        key = (relation["predicate"], relation["subject_id"], relation["object_id"])
        if key not in seen_relations:
            seen_relations.add(key)
            deduplicated_relations.append(relation)

    audit = {
        "$schema": "audit_record.schema.json",
        "schema_version": PACKAGE_VERSION,
        "record_id": "allen-cahn-dim-provenance-20261125",
        "identity": identity("Allen--Cahn adaptive RK4 claim-to-run provenance record", "audit record", "claim-evidence audit"),
        "created_utc": CREATED_UTC,
        "scope": "Completed Allen--Cahn confirmatory and long-sequence development evidence used by the DIM adaptive-weighting manuscript. No new training is represented; fixed-versus-zero remains contextual and outside the formal claim graph.",
        "inferential_status_vocabulary": ["locked_confirmatory_primary", "locked_confirmatory_secondary", "contextual_component_control", "development_only", "descriptive_diagnostic"],
        "protocols": protocols,
        "configurations": configurations,
        "spatial_domains": spatial_domains,
        "runs": runs,
        "checkpoints": checkpoints,
        "predictions": predictions,
        "evaluations": evaluations,
        "controller_traces": controller_traces,
        "metrics": metrics,
        "claims": claims,
        "artifacts": artifacts,
        "generations": generations,
        "prov": {
            "namespaces": {"prov": "http://www.w3.org/ns/prov#", "schema": "https://schema.org/"},
            "record_class_mapping": {
                "protocols": "prov:Entity",
                "configurations": "prov:Entity",
                "spatial_domains": "prov:Entity",
                "runs": "prov:Activity",
                "checkpoints": "prov:Entity",
                "predictions": "prov:Entity",
                "evaluations": "prov:Entity",
                "controller_traces": "prov:Entity",
                "metrics": "prov:Entity",
                "claims": "prov:Entity",
                "artifacts": "prov:Entity",
                "generations": "prov:Activity",
            },
            "agents": agents,
            "entities": entities,
            "activities": activities,
            "relations": deduplicated_relations,
        },
    }
    dump(ROOT / "allen_cahn_audit_record.json", audit)

    with (ROOT / "claim_evidence_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["claim_id", "inferential_status", "evidence_assessment", "direction", "metric_ids", "supporting_evidence_ids", "limiting_evidence_ids", "supporting_artifact_ids", "limiting_artifact_ids", "statement", "qualification"])
        for claim in claims:
            writer.writerow(
                [
                    claim["id"],
                    claim["inferential_status"],
                    claim["evidence_assessment"],
                    claim["direction"],
                    ";".join(claim["metric_ids"]),
                    ";".join(claim["supporting_evidence_ids"]),
                    ";".join(claim["limiting_evidence_ids"]),
                    ";".join(claim["supporting_artifact_ids"]),
                    ";".join(claim["limiting_artifact_ids"]),
                    claim["statement"],
                    claim["qualification"],
                ]
            )

    subprocess.run([sys.executable, str(ROOT / "query_claim_evidence.py"), "--write"], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(ROOT / "audit_comparison.py"), "--write"], cwd=ROOT, check=True)
    write_ro_crate(audit["created_utc"])
    subprocess.run([sys.executable, str(ROOT / "validate_provenance.py"), "--verify-external", "--write-report"], cwd=ROOT, check=True)
    print(
        "wrote "
        f"{len(runs)} runs, {len(checkpoints)} checkpoints, {len(predictions)} predictions, "
        f"{len(evaluations)} evaluations, {len(metrics)} metrics, {len(claims)} claims, "
        f"{len(artifacts)} artifacts and {len(generations)} generations"
    )


if __name__ == "__main__":
    main()
