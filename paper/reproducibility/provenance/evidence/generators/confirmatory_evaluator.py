#!/usr/bin/env python3
"""Independent, reference-separated evaluation for controlled Allen--Cahn runs.

The trainer's ``predictions.npz`` supplies x, t and the direct final-network
prediction (space, time). References are separate NPZ files with x, t and
reference/usol/u. Development reads training and validation-only references;
confirmation requires an existing selection-lock file before opening truth.

All continuation arms use float64 DOP853 and the same 256-node periodic,
centred finite-difference operator, independently of the training integrator.
Reference-state discrepancies are diagnostics, not universal error floors.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
from typing import Callable

import numpy as np
import scipy
from scipy.integrate import solve_ivp

TRAIN_END = 0.8
VALIDATION_END = 1.0
TEST_END = 1.2
TIME_TOLERANCE = 1e-7
PERIOD = 2.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(dir=path.parent, delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def relative_l2(prediction: np.ndarray, reference: np.ndarray) -> float:
    """Aggregate field error; never average the per-time relative errors."""
    prediction, reference = np.asarray(prediction), np.asarray(reference)
    if prediction.shape != reference.shape or not prediction.size:
        raise ValueError("relative L2 needs matching nonempty fields")
    if not (np.isfinite(prediction).all() and np.isfinite(reference).all()):
        raise FloatingPointError("non-finite field in relative L2")
    denominator = max(float(np.linalg.norm(reference.ravel())), np.finfo(float).eps)
    return float(np.linalg.norm((prediction - reference).ravel()) / denominator)


def interval_masks(times: np.ndarray, phase: str) -> dict[str, np.ndarray]:
    if phase not in {"development", "confirmation"}:
        raise ValueError("phase must be development or confirmation")
    times = np.asarray(times, dtype=float)
    masks = {
        "training": (times >= -TIME_TOLERANCE) & (times <= TRAIN_END + TIME_TOLERANCE),
        "validation": (times > TRAIN_END + TIME_TOLERANCE)
        & (times <= VALIDATION_END + TIME_TOLERANCE),
    }
    if phase == "confirmation":
        masks["test"] = (times > VALIDATION_END + TIME_TOLERANCE) & (
            times <= TEST_END + TIME_TOLERANCE
        )
    if any(not mask.any() for mask in masks.values()):
        raise ValueError("every requested evaluation interval must contain snapshots")
    return masks


def spatial_field(values: np.ndarray, nx: int, nt: int) -> np.ndarray:
    field = np.asarray(values, dtype=np.float64)
    if field.shape == (nt, nx) and field.shape != (nx, nt):
        field = field.T
    if field.shape != (nx, nt) or not np.isfinite(field).all():
        raise ValueError(f"expected a finite ({nx}, {nt}) field, got {field.shape}")
    return field


def unique_periodic_grid(x: np.ndarray, field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size < 3 or np.any(np.diff(x) <= 0) or not np.isfinite(x).all():
        raise ValueError("spatial coordinates must be finite and increasing")
    # Permit a stored duplicate endpoint, but integrate and score unique nodes.
    if np.isclose(x[-1] - x[0], PERIOD, rtol=0, atol=2e-6):
        x, field = x[:-1], field[:-1]
    spacing = PERIOD / x.size
    if not np.allclose(np.diff(x), spacing, rtol=2e-4, atol=2e-7):
        raise ValueError("expected a uniform, unique periodic grid of period two")
    return x, field


def periodic_interpolate(values: np.ndarray, source_x: np.ndarray,
                         target_x: np.ndarray) -> np.ndarray:
    """Periodic linear interpolation along the spatial (first) axis."""
    values = np.asarray(values, dtype=np.float64)
    vector = values.ndim == 1
    matrix = values[:, None] if vector else values
    if matrix.ndim != 2 or matrix.shape[0] != np.asarray(source_x).size:
        raise ValueError("periodic interpolation expects space or space-by-time values")
    source_x, matrix = unique_periodic_grid(source_x, matrix)
    target_x = np.asarray(target_x, dtype=np.float64)
    wrapped = (target_x - source_x[0]) % PERIOD + source_x[0]
    extended_x = np.concatenate((source_x, [source_x[0] + PERIOD]))
    extended = np.concatenate((matrix, matrix[:1]), axis=0)
    output = np.column_stack([
        np.interp(wrapped, extended_x, extended[:, column])
        for column in range(matrix.shape[1])
    ])
    return output[:, 0] if vector else output


def allen_cahn_rhs(state: np.ndarray, *, alpha: float = 5.0,
                   beta: float = 5.0, nu: float = 1e-4) -> np.ndarray:
    state = np.asarray(state, dtype=np.float64)
    dx = PERIOD / state.size
    laplacian = (np.roll(state, -1) - 2.0 * state + np.roll(state, 1)) / dx**2
    return -alpha * state**3 + beta * state + nu * laplacian


def temporal_step(state: np.ndarray, step: float, rhs: Callable,
                  method: str = "rk4") -> np.ndarray:
    """Independent full-state maps used to test numerical order."""
    k1 = rhs(state)
    if method == "heun":
        return state + step * (k1 + rhs(state + step * k1)) / 2.0
    if method != "rk4":
        raise ValueError("method must be rk4 or heun")
    k2 = rhs(state + step * k1 / 2.0)
    k3 = rhs(state + step * k2 / 2.0)
    k4 = rhs(state + step * k3)
    return state + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def continue_field(initial: np.ndarray, start: float, times: np.ndarray,
                   coefficients: dict, max_step: float = 0.005) -> tuple[np.ndarray, dict]:
    times = np.asarray(times, dtype=np.float64)
    if times.size == 0 or np.any(np.diff(times) <= 0) or times[0] < start - TIME_TOLERANCE:
        raise ValueError("continuation times must increase from its initial time")
    result = solve_ivp(
        lambda _time, state: allen_cahn_rhs(state, **coefficients),
        (float(start), float(times[-1])), np.asarray(initial, dtype=np.float64),
        method="DOP853", t_eval=times, rtol=1e-9, atol=1e-11, max_step=max_step,
    )
    if not result.success or not np.isfinite(result.y).all():
        raise FloatingPointError(f"DOP853 continuation failed: {result.message}")
    return result.y, {
        "method": "DOP853", "dtype": "float64", "rtol": 1e-9,
        "atol": 1e-11, "max_step": max_step, "nfev": int(result.nfev),
        "success": True, "spatial_points": int(np.asarray(initial).size),
        "operator": "periodic centred second-order finite-difference Laplacian",
    }


def _time_indices(available: np.ndarray, requested: np.ndarray) -> np.ndarray:
    available = np.asarray(available, dtype=float)
    if available.ndim != 1 or np.any(np.diff(available) <= 0):
        raise ValueError("stored time coordinates must be increasing")
    indices = np.asarray([int(np.argmin(np.abs(available - time))) for time in requested])
    if not np.allclose(available[indices], requested, rtol=0, atol=TIME_TOLERANCE):
        raise ValueError("prediction and reference snapshots must have matching times")
    return indices


def _reference_segment(path: Path, role: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    lower, upper = {"training": (0.0, TRAIN_END),
                    "validation": (TRAIN_END, VALIDATION_END),
                    "test": (VALIDATION_END, TEST_END)}[role]
    with np.load(path, allow_pickle=False) as record:
        times = np.asarray(record["t"], dtype=np.float64).reshape(-1)
        # Inspect times before reading the truth array from the lazy NPZ archive.
        first = lower if role == "training" else lower + 0.005
        if (not times.size or np.any(np.diff(times) <= 0)
                or abs(times[0] - first) > TIME_TOLERANCE
                or abs(times[-1] - upper) > TIME_TOLERANCE
                or not np.allclose(np.diff(times), 0.005, rtol=0, atol=TIME_TOLERANCE)):
            raise ValueError(f"{role} reference must contain only its locked 0.005-spaced snapshots")
        x = np.asarray(record["x"], dtype=np.float64).reshape(-1)
        key = next((key for key in ("reference", "usol", "u") if key in record), None)
        if key is None:
            raise ValueError("reference NPZ needs reference, usol or u")
        field = spatial_field(record[key], x.size, times.size)
        coefficients = {
            name: float(np.asarray(record[name]).item()) if name in record else default
            for name, default in (("alpha", 5.0), ("beta", 5.0), ("nu", 1e-4))
        }
    x, field = unique_periodic_grid(x, field)
    if x.size != 1024:
        raise ValueError("the frozen evaluation reference must have 1024 unique periodic nodes")
    return x, times, field, coefficients


def evaluate(run_dir: str | Path, *, phase: str, reference_path: str | Path,
             selection_lock: str | Path | None = None, output_dir: str | Path | None = None,
             continuation_nodes: int = 256,
             training_reference_path: str | Path | None = None,
             validation_reference_path: str | Path | None = None,
             metrics_path: str | Path | None = None) -> dict:
    """Evaluate one run and atomically write metrics.json and fields.npz."""
    phase = {"validation": "development", "all": "confirmation"}.get(phase, phase)
    if phase not in {"development", "confirmation"}:
        raise ValueError("unknown phase")
    if continuation_nodes < 8:
        raise ValueError("continuation needs at least eight periodic nodes")
    lock = Path(selection_lock).resolve() if selection_lock is not None else None
    if phase == "confirmation":
        # Perform this check before loading either the prediction or test truth.
        if lock is None or not lock.is_file():
            raise ValueError("confirmation requires an existing frozen selection-lock file")
        if not isinstance(json.loads(lock.read_text()), dict):
            raise ValueError("selection lock must be a JSON object")
    run_dir = Path(run_dir).resolve()
    reference_path = Path(reference_path).resolve()
    prediction_path = run_dir / "predictions.npz"
    output_dir = Path(output_dir).resolve() if output_dir else run_dir / "evaluation" / phase
    metrics_path = Path(metrics_path).resolve() if metrics_path else output_dir / "metrics.json"
    diagnostics_path = run_dir / "diagnostics.json"
    if not diagnostics_path.exists():
        diagnostics_path = run_dir / "controlled_run.json"
    diagnostics = json.loads(diagnostics_path.read_text()) if diagnostics_path.exists() else {}
    training_path = (Path(training_reference_path).resolve() if training_reference_path
                     else reference_path.parent / "reference_training.npz")
    validation_path = (reference_path if phase == "development" else
                       Path(validation_reference_path).resolve() if validation_reference_path else
                       reference_path.parent / "reference_validation_only.npz")
    split_paths = {"training": training_path, "validation": validation_path}
    if phase == "confirmation":
        split_paths["test"] = reference_path
    segments = {role: _reference_segment(path, role) for role, path in split_paths.items()}
    x, _, _, coefficients = segments["training"]
    for role, (segment_x, _, _, segment_coefficients) in segments.items():
        if not np.array_equal(segment_x, x) or segment_coefficients != coefficients:
            raise ValueError(f"{role} reference grid or coefficients differ from training reference")
    times = np.concatenate([segment[1] for segment in segments.values()])
    reference = np.concatenate([segment[2] for segment in segments.values()], axis=1)
    masks = interval_masks(times, phase)
    terminal_index = int(_time_indices(times, np.asarray([TRAIN_END]))[0])
    with np.load(prediction_path, allow_pickle=False) as record:
        prediction_x = np.asarray(record["x"], dtype=np.float64).reshape(-1)
        prediction_times = np.asarray(record["t"], dtype=np.float64).reshape(-1)
        all_prediction = spatial_field(record["prediction"], prediction_x.size, prediction_times.size)
        direct = periodic_interpolate(
            all_prediction[:, _time_indices(prediction_times, times)], prediction_x, x
        )
        native_terminal = None
        if "fd_x" in record and "fd_terminal_state" in record:
            if "terminal_time" not in record or abs(float(record["terminal_time"]) - TRAIN_END) > TIME_TOLERANCE:
                raise ValueError("native FD terminal state must be recorded at t=0.8")
            native_terminal = (
                np.asarray(record["fd_x"], dtype=np.float64).reshape(-1),
                np.asarray(record["fd_terminal_state"], dtype=np.float64).reshape(-1),
            )
        accepted = None
        if "accepted_prediction" in record and "accepted_t" in record:
            accepted_times = np.asarray(record["accepted_t"], dtype=np.float64).reshape(-1)
            raw_accepted = spatial_field(record["accepted_prediction"], prediction_x.size,
                                         accepted_times.size)
            accepted = periodic_interpolate(
                raw_accepted[:, _time_indices(accepted_times, times[masks["training"]])],
                prediction_x, x,
            )
    grid = np.linspace(x[0], x[0] + PERIOD, continuation_nodes, endpoint=False)
    future_mask = times > TRAIN_END + TIME_TOLERANCE
    future_times = times[future_mask]
    learned_terminal = (
        periodic_interpolate(native_terminal[1], native_terminal[0], grid)
        if native_terminal is not None else
        periodic_interpolate(direct[:, terminal_index], x, grid)
    )
    reference_terminal = periodic_interpolate(reference[:, terminal_index], x, grid)
    reference_initial = periodic_interpolate(reference[:, 0], x, grid)
    continued, continuation_info = continue_field(learned_terminal, TRAIN_END, future_times, coefficients)
    reference_started, reference_info = continue_field(reference_terminal, TRAIN_END, future_times, coefficients)
    baseline, baseline_info = continue_field(reference_initial, float(times[0]), times, coefficients)
    continued = periodic_interpolate(continued, grid, x)
    reference_started = periodic_interpolate(reference_started, grid, x)
    baseline = periodic_interpolate(baseline, grid, x)
    fields = {
        "x": x, "t": times, "direct_prediction": direct,
        "reference": reference, "future_t": future_times,
        "continued_prediction": continued, "reference_terminal_continuation": reference_started,
        "numerical_baseline": baseline,
        "direct_time_relative_l2": np.asarray([
            relative_l2(direct[:, index], reference[:, index]) for index in range(times.size)
        ]),
        "continuation_time_relative_l2": np.asarray([
            relative_l2(continued[:, index], reference[:, np.flatnonzero(future_mask)[index]])
            for index in range(future_times.size)
        ]),
    }
    metrics = {
        "schema_version": 1, "status": "complete", "phase": phase,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_mechanism": "direct final-network coordinate queries",
        "relative_l2_convention": "norm of aggregate space-time error divided by norm of reference",
        "training_retention_definition": "final-network error on the entire training interval",
        "intervals": {"training": [0.0, TRAIN_END], "validation": [TRAIN_END, VALIDATION_END]},
        "evaluation_spatial_nodes": int(x.size), "continuation_spatial_nodes": continuation_nodes,
        "coefficients": coefficients,
        "terminal_relative_l2": relative_l2(direct[:, terminal_index], reference[:, terminal_index]),
        "continuation_solver": continuation_info,
        "continuation_initial_state_source": (
            "native final-network FD-grid state at t=0.8" if native_terminal is not None else
            "final-network reference-grid state at t=0.8, periodically interpolated"
        ),
        "reference_terminal_solver": reference_info,
        "numerical_baseline_solver": baseline_info,
        "discrepancy_qualification": "Reference-started finite-difference discrepancies are not universal error lower bounds.",
        "periodic_interpolation": "linear interpolation with period 2 onto the common 1024-node reference grid",
        "block_records": diagnostics.get("block_records", []),
        "direct": {}, "continuation": {}, "numerical_baseline": {},
        "reference_terminal_continuation": {},
        "provenance": {
            "reference_path": str(reference_path), "reference_sha256": sha256(reference_path),
            "prediction_path": str(prediction_path), "prediction_sha256": sha256(prediction_path),
            "evaluator_sha256": sha256(Path(__file__)),
            "reference_splits": {role: {"path": str(path), "sha256": sha256(path)}
                                 for role, path in split_paths.items()},
            "selection_lock_path": str(lock) if lock else None,
            "selection_lock_sha256": sha256(lock) if lock else None,
            "diagnostics_sha256": sha256(diagnostics_path) if diagnostics_path.exists() else None,
            "initial_parameters_sha256": diagnostics.get("initial_parameters_sha256"),
            "pair_schedule_sha256": diagnostics.get("pair_schedule_sha256"),
            "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        },
    }
    if phase == "confirmation":
        metrics["intervals"]["test"] = [VALIDATION_END, TEST_END]
    for name, mask in masks.items():
        metrics[f"direct_{name}_relative_l2"] = relative_l2(direct[:, mask], reference[:, mask])
        metrics[f"numerical_baseline_{name}_relative_l2"] = relative_l2(baseline[:, mask], reference[:, mask])
        metrics["direct"][name] = {"relative_l2": metrics[f"direct_{name}_relative_l2"]}
        metrics["numerical_baseline"][name] = {"relative_l2": metrics[f"numerical_baseline_{name}_relative_l2"]}
        if name != "training":
            subset = mask[future_mask]
            metrics[f"continuation_{name}_relative_l2"] = relative_l2(continued[:, subset], reference[:, mask])
            metrics[f"reference_terminal_continuation_{name}_relative_l2"] = relative_l2(
                reference_started[:, subset], reference[:, mask]
            )
            metrics["continuation"][name] = {"relative_l2": metrics[f"continuation_{name}_relative_l2"]}
            metrics["reference_terminal_continuation"][name] = {
                "relative_l2": metrics[f"reference_terminal_continuation_{name}_relative_l2"]}
    if accepted is not None:
        metrics["accepted_history_training_relative_l2"] = relative_l2(accepted, reference[:, masks["training"]])
        metrics["final_vs_accepted_history_relative_l2"] = relative_l2(direct[:, masks["training"]], accepted)
        fields["accepted_prediction"] = accepted
    # Interface-time final-network errors complement the trainer's handoff mismatch.
    interface_times = np.asarray([0.2, 0.4, 0.6, 0.8])
    interface_indices = _time_indices(times, interface_times)
    metrics["final_network_interface_errors"] = [
        {"time": float(time), "relative_l2": relative_l2(direct[:, index], reference[:, index])}
        for time, index in zip(interface_times, interface_indices)
    ]
    metrics["finite"] = all(np.isfinite(value).all() for value in fields.values())
    atomic_npz(output_dir / "fields.npz", **fields)
    metrics["provenance"]["fields_sha256"] = sha256(output_dir / "fields.npz")
    atomic_json(metrics_path, metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--phase", choices=("development", "confirmation", "validation", "all"), required=True)
    parser.add_argument("--reference")
    parser.add_argument("--training-reference")
    parser.add_argument("--validation-reference")
    parser.add_argument("--selection-lock")
    parser.add_argument("--output-dir")
    parser.add_argument("--output", help="Explicit metrics JSON destination; arrays remain in output-dir")
    parser.add_argument("--continuation-nodes", type=int, default=256)
    args = parser.parse_args()
    phase = {"validation": "development", "all": "confirmation"}.get(args.phase, args.phase)
    reference = args.reference or Path(__file__).parent / "reference" / (
        "reference_validation_only.npz" if phase == "development" else "reference_test_only.npz")
    metrics = evaluate(args.run_dir, phase=phase, reference_path=reference,
                       selection_lock=args.selection_lock, output_dir=args.output_dir,
                       continuation_nodes=args.continuation_nodes,
                       training_reference_path=args.training_reference,
                       validation_reference_path=args.validation_reference,
                       metrics_path=args.output)
    print(json.dumps({key: value for key, value in metrics.items()
                      if key.startswith("direct_") or key in {"status", "phase", "finite"}}, indent=2))


if __name__ == "__main__":
    main()
