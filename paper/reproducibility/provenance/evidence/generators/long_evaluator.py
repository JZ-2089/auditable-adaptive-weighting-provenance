#!/usr/bin/env python3
"""Evaluate one long-sequence run on short, medium and long forecast windows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from run_study import ROOT, SCENARIOS, atomic_json


def relative_l2(prediction: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm((prediction - reference).ravel()) /
                 max(np.linalg.norm(reference.ravel()), np.finfo(float).eps))


def field(values: np.ndarray, nx: int, nt: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.shape == (nt, nx):
        values = values.T
    if values.shape != (nx, nt) or not np.isfinite(values).all():
        raise ValueError(f"Expected finite ({nx},{nt}) field, got {values.shape}")
    return values


def time_indices(available: np.ndarray, requested: np.ndarray) -> np.ndarray:
    indices = np.asarray([np.argmin(np.abs(available - value)) for value in requested])
    if not np.allclose(available[indices], requested, atol=1.0e-7, rtol=0):
        raise ValueError("Prediction and reference times do not match")
    return indices


def periodic_interpolate(values: np.ndarray, source_x: np.ndarray,
                         target_x: np.ndarray) -> np.ndarray:
    source_x = np.asarray(source_x, dtype=float)
    values = np.asarray(values, dtype=float)
    if source_x.size > 2 and np.isclose(source_x[-1] - source_x[0], 2.0, atol=2e-6):
        source_x, values = source_x[:-1], values[:-1]
    vector = values.ndim == 1
    values = values[:, None] if vector else values
    wrapped = (target_x - source_x[0]) % 2.0 + source_x[0]
    extended_x = np.r_[source_x, source_x[0] + 2.0]
    extended = np.r_[values, values[:1]]
    result = np.column_stack([
        np.interp(wrapped, extended_x, extended[:, index])
        for index in range(values.shape[1])
    ])
    return result[:, 0] if vector else result


def continue_from(state: np.ndarray, start: float, times: np.ndarray) -> np.ndarray:
    dx = 2.0 / state.size

    def rhs(_time: float, u: np.ndarray) -> np.ndarray:
        laplacian = (np.roll(u, -1) - 2.0 * u + np.roll(u, 1)) / dx**2
        return 1.0e-4 * laplacian + 5.0 * u - 5.0 * u**3

    result = solve_ivp(rhs, (start, float(times[-1])), state, method="DOP853",
                       t_eval=times, rtol=1e-9, atol=1e-11, max_step=0.005)
    if not result.success or not np.isfinite(result.y).all():
        raise RuntimeError(result.message)
    return result.y


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    scenario = SCENARIOS[args.scenario]
    train_end, forecast_end = scenario["train_end"], scenario["forecast_end"]
    with np.load(ROOT / "reference/reference_truth.npz") as archive:
        reference_x = np.asarray(archive["x"], dtype=float)
        reference_t = np.asarray(archive["t"], dtype=float)
        reference = np.asarray(archive["usol"], dtype=float).T
    keep = reference_t <= forecast_end + 1e-7
    reference_t, reference = reference_t[keep], reference[:, keep]
    with np.load(run_dir / "output/predictions.npz") as archive:
        prediction_x = np.asarray(archive["x"], dtype=float)
        prediction_t = np.asarray(archive["t"], dtype=float)
        key = "direct_prediction" if "direct_prediction" in archive else "prediction"
        prediction = field(archive[key], prediction_x.size, prediction_t.size)
        native_x = np.asarray(archive["fd_x"], dtype=float)
        native_terminal = np.asarray(archive["fd_terminal_state"], dtype=float)
    direct = periodic_interpolate(
        prediction[:, time_indices(prediction_t, reference_t)], prediction_x, reference_x
    )
    train_mask = reference_t <= train_end + 1e-7
    terminal_index = int(time_indices(reference_t, np.asarray([train_end]))[0])
    metrics = {
        "scenario": args.scenario, "blocks": scenario["blocks"],
        "train_end": train_end, "forecast_end": forecast_end,
        "direct_training_relative_l2": relative_l2(direct[:, train_mask], reference[:, train_mask]),
        "direct_terminal_relative_l2": relative_l2(
            direct[:, terminal_index], reference[:, terminal_index]
        ),
    }
    for horizon in (0.2, 0.4, 0.8):
        mask = (reference_t > train_end + 1e-7) & (
            reference_t <= train_end + horizon + 1e-7
        )
        metrics[f"direct_future_{horizon:.1f}_relative_l2"] = relative_l2(
            direct[:, mask], reference[:, mask]
        )
    interface_times = np.linspace(0.0, train_end, scenario["blocks"] + 1)[1:]
    interface_indices = time_indices(reference_t, interface_times)
    metrics["interface_times"] = interface_times.tolist()
    metrics["interface_relative_l2"] = [
        relative_l2(direct[:, index], reference[:, index]) for index in interface_indices
    ]
    metrics["interface_mean_relative_l2"] = float(np.mean(metrics["interface_relative_l2"]))
    metrics["interface_max_relative_l2"] = float(np.max(metrics["interface_relative_l2"]))

    continuation_grid = np.linspace(-1.0, 1.0, 256, endpoint=False)
    terminal = periodic_interpolate(native_terminal, native_x, continuation_grid)
    future_mask = reference_t > train_end + 1e-7
    future_t = reference_t[future_mask]
    continued = continue_from(terminal, train_end, future_t)
    continued_reference = periodic_interpolate(
        reference[:, future_mask], reference_x, continuation_grid
    )
    for horizon in (0.2, 0.4, 0.8):
        mask = future_t <= train_end + horizon + 1e-7
        metrics[f"continuation_future_{horizon:.1f}_relative_l2"] = relative_l2(
            continued[:, mask], continued_reference[:, mask]
        )
    atomic_json(run_dir / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
