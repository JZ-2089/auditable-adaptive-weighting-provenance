#!/usr/bin/env python3
"""Controlled fixed-update sequential FD temporal-consistency PINN for Allen--Cahn.

Publication-study entry point: fixed weights, final iterate, immutable block
checkpoints, initial-only input, paired sampling and numerical diagnostics.
The historical implementation below is retained for source provenance; the
entry point uses controlled_runner.run and never invokes legacy train/save.

Historical source description follows:

Implements the paper's method: u_t + alpha*u^3 - beta*u - nu*u_xx = 0,
with a periodic centred finite-difference Laplacian and explicit RK4 temporal
consistency.  In inverse mode, alpha, beta, and nu are positive trainable
scalars inferred from sparse field observations.

Conservative fixed forecast (train on [0, 1], predict (1, 3]):
  python GPT-code-adaptive.py --config forward
Grouped adaptive experiment:
  python GPT-code-adaptive.py --config forward --adaptive-weights --rk4-weight 0.03
Inverse parameter estimation:
  python GPT-code-adaptive.py --config inverse --infer alpha beta nu

The original ``GPT-code.py`` is intentionally left unchanged.  This copy can
also reproduce its static, unnormalised objective with
``--no-adaptive-weights --no-normalize-rk4-defect --no-deduplicate-replay
--rk4-weight 0.1``.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import scipy.io
from flax import linen as nn
from jax import grad, jit, lax, random, value_and_grad, vmap
from tqdm.auto import trange

# Shared, PDE-agnostic controllers -- the inverse-loss option lives there so
# this solver runs the identical code to ns_unsteady_cylinder / RD / Burgers.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import adaptive_weight_controllers as controllers

def to_defaults(config):
    """Adapt a nested JAXPI-style config to the internal Config dataclass."""
    hidden = tuple(config.arch.hidden_dims) if config.arch.hidden_dims else (config.arch.hidden_dim,) * config.arch.num_layers
    return {
        "data_path": config.data.path, "output_dir": config.data.output_dir,
        "architecture": config.arch.arch_name, "activation": config.arch.activation,
        "hidden": hidden, "method": config.physics.method,
        "normalize_inputs": config.arch.normalize_inputs,
        "gate_activation": config.arch.gate_activation,
        "use_residual_gates": config.arch.use_residual_gates,
        "residual_gate_init": config.arch.residual_gate_init,
        "weight_factorization": config.arch.reparam is not None,
        "weight_fact_mean": config.arch.reparam.mean if config.arch.reparam else 0.0,
        "weight_fact_stddev": config.arch.reparam.stddev if config.arch.reparam else 0.0,
        # Hard initial condition, as in the RD/Burgers/NS solvers of this family.
        # Without it the residual's stable equilibria u = +-1 are exact solutions
        # the optimiser can collapse onto while ignoring a softly weighted IC.
        "lr_decay_spans_run": getattr(config.optim, "lr_decay_spans_run", True),
        "hard_initial_condition": getattr(config.arch, "hard_initial_condition", False),
        "hard_initial_transition": getattr(config.arch, "hard_initial_transition", 0.05),
        "alpha": config.physics.alpha, "beta": config.physics.beta, "nu": config.physics.nu,
        "slabs": config.training.time_slabs, "epochs": config.training.max_epochs,
        "pairs_per_step": config.training.rk_pairs_per_step, "spatial_points": config.training.spatial_points,
        "rk4_substeps": config.training.rk4_substeps,
        "early_stopping": config.training.early_stopping.enabled,
        "early_stopping_warmup": config.training.early_stopping.warmup_epochs,
        "early_stopping_patience": config.training.early_stopping.patience,
        "early_stopping_min_delta": config.training.early_stopping.min_delta,
        "early_stopping_ema_decay": config.training.early_stopping.ema_decay,
        "learning_rate": config.optim.learning_rate, "parameter_learning_rate": config.optim.parameter_learning_rate,
        "rk4_weight": config.weighting.rk4, "residual_weight": config.weighting.residual,
        "initial_weight": config.weighting.initial,
        "boundary_weight": config.weighting.boundary, "previous_weight": config.weighting.previous,
        "data_weight": config.weighting.data, "mode": config.mode,
        "deduplicate_replay": config.weighting.deduplicate_replay,
        "normalize_rk4_defect": config.weighting.normalize_rk4_defect,
        "adaptive_weights": config.weighting.adaptive.enabled,
        "adaptive_update_every": config.weighting.adaptive.update_every,
        "adaptive_first_slab_warmup": config.weighting.adaptive.first_slab_warmup_epochs,
        "adaptive_later_slab_warmup": config.weighting.adaptive.later_slab_warmup_epochs,
        "adaptive_ema_decay": config.weighting.adaptive.ema_decay,
        "adaptive_rate": config.weighting.adaptive.rate,
        "adaptive_progress_power": config.weighting.adaptive.progress_power,
        "adaptive_min_multiplier": config.weighting.adaptive.min_multiplier,
        "adaptive_max_multiplier": config.weighting.adaptive.max_multiplier,
        "adaptive_gradient_epsilon": config.weighting.adaptive.gradient_epsilon,
        "rk4_previous_controller": getattr(config.weighting, "rk4_previous_controller", "grouped"),
        "inverse_loss_normalize": getattr(getattr(config.weighting, "inverse_loss", None), "enabled", True),
        "inverse_loss_ema_decay": getattr(getattr(config.weighting, "inverse_loss", None), "ema_decay", 0.99),
        "inverse_loss_power": getattr(getattr(config.weighting, "inverse_loss", None), "power", 1.0),
        "inverse_loss_min": getattr(getattr(config.weighting, "inverse_loss", None), "min_coefficient", 1.0e-2),
        "inverse_loss_max": getattr(getattr(config.weighting, "inverse_loss", None), "max_coefficient", 1.0e2),
        "inverse_loss_warmup_epochs": getattr(getattr(config.weighting, "inverse_loss", None), "warmup_epochs", 200),
        "inverse_loss_update_every": getattr(getattr(config.weighting, "inverse_loss", None), "update_every", 1),
        "inverse_loss_epsilon": getattr(getattr(config.weighting, "inverse_loss", None), "epsilon", 1.0e-12),
        "observations": config.inverse.observations_per_slab, "observation_path": config.data.observation_path,
        "initial_alpha": config.inverse.initial_alpha, "initial_beta": config.inverse.initial_beta,
        "initial_nu": config.inverse.initial_nu, "infer_parameters": tuple(config.inverse.parameters),
        "seed": config.seed, "use_wandb": config.logging.use_wandb,
        "wandb_project": config.logging.wandb_project, "wandb_entity": config.logging.wandb_entity,
        "wandb_run_name": config.logging.wandb_run_name, "log_every": config.logging.log_every_steps,
        "checkpoint_dir": config.saving.checkpoint_dir, "checkpoint_every": config.saving.checkpoint_every_slabs,
        "resume_checkpoint": config.saving.resume_checkpoint,
        "train_end_time": config.forecast.train_end_time,
        "forecast_end_time": config.forecast.end_time,
        "forecast_strategy": config.forecast.strategy,
    }


def load_selected_config(argv):
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--config", choices=("forward", "inverse"), default="forward")
    selection, _ = selector.parse_known_args(argv)
    module = importlib.import_module(f"{selection.config}_invloss_config")
    return selection.config, module.get_config()


SELECTED_CONFIG, RAW_CONFIG = load_selected_config(sys.argv[1:])
DEFAULTS = to_defaults(RAW_CONFIG)

PARAMETER_NAMES = ("alpha", "beta", "nu")
ACTIVATIONS = {"tanh": nn.tanh, "silu": nn.silu, "gelu": nn.gelu}
# This canonical order is used by optimisation, monitoring, checkpoints, and
# saved diagnostics.  Keeping one order prevents silently applying a weight to
# the wrong loss term.
LOSS_NAMES = ("residual", "rk4", "initial", "boundary", "previous", "data")
# Preserve weight mass within semantically related tasks.  A single global
# budget lets an already-solved replay loss steal nearly all weight from the
# Allen--Cahn physics losses because its raw gradient is tiny.
ADAPTIVE_GROUP_NAMES = (("residual", "rk4"),
                        ("boundary", "previous", "data"))
ADAPTIVE_POLICY = "grouped_priority_gradnorm_v2"
MODEL_VERSION = 11
ADAPTIVE_GROUPS = tuple(
    tuple(LOSS_NAMES.index(name) for name in group)
    for group in ADAPTIVE_GROUP_NAMES
)
# --rk4-previous-controller inverse_loss pulls these two out of the grouped
# controller and gives each its own inverse-loss coefficient (no gradient
# signal, no budget); "grouped" is this file's native controller.
_INVERSE_LOSS_NAMES = ("rk4", "previous")
_INVERSE_LOSS_INDICES = tuple(LOSS_NAMES.index(name) for name in _INVERSE_LOSS_NAMES)


@dataclass(frozen=True)
class Config:
    data_path: str = DEFAULTS["data_path"]
    output_dir: str = DEFAULTS["output_dir"]
    alpha: float = DEFAULTS["alpha"]
    beta: float = DEFAULTS["beta"]
    nu: float = DEFAULTS["nu"]
    slabs: int = DEFAULTS["slabs"]
    epochs: int = DEFAULTS["epochs"]
    pairs_per_step: int = DEFAULTS["pairs_per_step"]
    spatial_points: int = DEFAULTS["spatial_points"]
    rk4_substeps: int = DEFAULTS["rk4_substeps"]
    early_stopping: bool = DEFAULTS["early_stopping"]
    early_stopping_warmup: int = DEFAULTS["early_stopping_warmup"]
    early_stopping_patience: int = DEFAULTS["early_stopping_patience"]
    early_stopping_min_delta: float = DEFAULTS["early_stopping_min_delta"]
    early_stopping_ema_decay: float = DEFAULTS["early_stopping_ema_decay"]
    hidden: Sequence[int] = DEFAULTS["hidden"]
    architecture: str = DEFAULTS["architecture"]
    activation: str = DEFAULTS["activation"]
    normalize_inputs: bool = DEFAULTS["normalize_inputs"]
    gate_activation: str = DEFAULTS["gate_activation"]
    use_residual_gates: bool = DEFAULTS["use_residual_gates"]
    residual_gate_init: float = DEFAULTS["residual_gate_init"]
    weight_factorization: bool = DEFAULTS["weight_factorization"]
    weight_fact_mean: float = DEFAULTS["weight_fact_mean"]
    weight_fact_stddev: float = DEFAULTS["weight_fact_stddev"]
    lr_decay_spans_run: bool = DEFAULTS["lr_decay_spans_run"]
    hard_initial_condition: bool = DEFAULTS["hard_initial_condition"]
    hard_initial_transition: float = DEFAULTS["hard_initial_transition"]
    method: str = DEFAULTS["method"]
    learning_rate: float = DEFAULTS["learning_rate"]
    rk4_weight: float = DEFAULTS["rk4_weight"]
    residual_weight: float = DEFAULTS["residual_weight"]
    initial_weight: float = DEFAULTS["initial_weight"]
    boundary_weight: float = DEFAULTS["boundary_weight"]
    previous_weight: float = DEFAULTS["previous_weight"]
    data_weight: float = DEFAULTS["data_weight"]
    deduplicate_replay: bool = DEFAULTS["deduplicate_replay"]
    normalize_rk4_defect: bool = DEFAULTS["normalize_rk4_defect"]
    adaptive_weights: bool = DEFAULTS["adaptive_weights"]
    adaptive_update_every: int = DEFAULTS["adaptive_update_every"]
    adaptive_first_slab_warmup: int = DEFAULTS["adaptive_first_slab_warmup"]
    adaptive_later_slab_warmup: int = DEFAULTS["adaptive_later_slab_warmup"]
    adaptive_ema_decay: float = DEFAULTS["adaptive_ema_decay"]
    adaptive_rate: float = DEFAULTS["adaptive_rate"]
    adaptive_progress_power: float = DEFAULTS["adaptive_progress_power"]
    adaptive_min_multiplier: float = DEFAULTS["adaptive_min_multiplier"]
    adaptive_max_multiplier: float = DEFAULTS["adaptive_max_multiplier"]
    adaptive_gradient_epsilon: float = DEFAULTS["adaptive_gradient_epsilon"]
    adaptive_max_temporal_gradient_ratio: float = 1.0e30
    rk4_previous_controller: str = DEFAULTS["rk4_previous_controller"]
    inverse_loss_normalize: bool = DEFAULTS["inverse_loss_normalize"]
    inverse_loss_ema_decay: float = DEFAULTS["inverse_loss_ema_decay"]
    inverse_loss_power: float = DEFAULTS["inverse_loss_power"]
    inverse_loss_min: float = DEFAULTS["inverse_loss_min"]
    inverse_loss_max: float = DEFAULTS["inverse_loss_max"]
    inverse_loss_warmup_epochs: int = DEFAULTS["inverse_loss_warmup_epochs"]
    inverse_loss_update_every: int = DEFAULTS["inverse_loss_update_every"]
    inverse_loss_epsilon: float = DEFAULTS["inverse_loss_epsilon"]
    mode: str = DEFAULTS["mode"]
    observations: int = DEFAULTS["observations"]
    observation_path: str | None = DEFAULTS["observation_path"]
    initial_alpha: float = DEFAULTS["initial_alpha"]
    initial_beta: float = DEFAULTS["initial_beta"]
    initial_nu: float = DEFAULTS["initial_nu"]
    parameter_learning_rate: float = DEFAULTS["parameter_learning_rate"]
    infer_parameters: tuple[str, ...] = DEFAULTS["infer_parameters"]
    seed: int = DEFAULTS["seed"]
    use_wandb: bool = DEFAULTS["use_wandb"]
    wandb_project: str = DEFAULTS["wandb_project"]
    wandb_entity: str | None = DEFAULTS["wandb_entity"]
    wandb_run_name: str | None = DEFAULTS["wandb_run_name"]
    log_every: int = DEFAULTS["log_every"]
    checkpoint_dir: str = DEFAULTS["checkpoint_dir"]
    checkpoint_every: int = DEFAULTS["checkpoint_every"]
    resume_checkpoint: str | None = DEFAULTS["resume_checkpoint"]
    train_end_time: float = DEFAULTS["train_end_time"]
    forecast_end_time: float = DEFAULTS["forecast_end_time"]
    forecast_strategy: str = DEFAULTS["forecast_strategy"]
    temporal_integrator: str = "rk4"
    diagnostic_every: int = 500
    gradient_clip_norm: float = 0.0


class FactorizedDense(nn.Module):
    """Dense layer with the same random weight factorization used by JAXPI."""
    features: int
    enabled: bool = True
    mean: float = 0.5
    stddev: float = 0.1

    @nn.compact
    def __call__(self, inputs):
        shape = (inputs.shape[-1], self.features)
        kernel_init = nn.initializers.glorot_normal()
        if self.enabled:
            def factorized_init(key, kernel_shape):
                weight_key, scale_key = random.split(key)
                weight = kernel_init(weight_key, kernel_shape)
                scale = jnp.exp(self.mean + self.stddev * random.normal(scale_key, (kernel_shape[-1],)))
                return scale, weight / scale
            scale, direction = self.param("kernel", factorized_init, shape)
            kernel = scale * direction
        else:
            kernel = self.param("kernel", kernel_init, shape)
        bias = self.param("bias", nn.initializers.zeros_init(), (self.features,))
        return jnp.dot(inputs, kernel) + bias


def normalize_network_inputs(inputs, lower, upper, enabled):
    if not enabled:
        return inputs
    lower, upper = jnp.asarray(lower), jnp.asarray(upper)
    return 2.0 * (inputs - lower) / jnp.maximum(upper - lower, 1e-12) - 1.0


class ModifiedMLP(nn.Module):
    """mMLP enhanced with normalized inputs, residual gates, and weight factorization."""
    widths: Sequence[int]
    activation: str = "tanh"
    input_lower: Sequence[float] = (0.0, -1.0)
    input_upper: Sequence[float] = (1.0, 1.0)
    normalize_inputs: bool = True
    gate_activation: str = "sigmoid"
    use_residual_gates: bool = True
    residual_gate_init: float = 0.1
    weight_factorization: bool = True
    weight_fact_mean: float = 0.5
    weight_fact_stddev: float = 0.1

    @nn.compact
    def __call__(self, inputs):
        if not self.widths:
            raise ValueError("ModifiedMLP needs at least one hidden width")
        width = self.widths[0]
        if any(layer_width != width for layer_width in self.widths):
            raise ValueError("ModifiedMLP requires equal hidden widths")
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"Unknown activation {self.activation!r}; choose from {tuple(ACTIVATIONS)}")
        if self.gate_activation not in ("sigmoid", "tanh"):
            raise ValueError("gate_activation must be 'sigmoid' or 'tanh'")
        if not 0.0 < self.residual_gate_init < 1.0:
            raise ValueError("residual_gate_init must lie strictly between 0 and 1")
        act = ACTIVATIONS[self.activation]
        inputs = normalize_network_inputs(inputs, self.input_lower, self.input_upper, self.normalize_inputs)

        def dense(features, name):
            return FactorizedDense(features, self.weight_factorization, self.weight_fact_mean,
                                   self.weight_fact_stddev, name=name)

        u = act(dense(width, "encoder_u")(inputs))
        v = act(dense(width, "encoder_v")(inputs))
        h = act(dense(width, "hidden_input")(inputs))
        residual_logit = np.log(self.residual_gate_init / (1.0 - self.residual_gate_init))
        for layer_index in range(len(self.widths)):
            gate_pre = dense(width, f"gate_{layer_index}")(h)
            gate = nn.sigmoid(gate_pre) if self.gate_activation == "sigmoid" else act(gate_pre)
            mixed = (1.0 - gate) * u + gate * v
            if self.use_residual_gates:
                blend_logit = self.param(f"residual_logit_{layer_index}",
                                         nn.initializers.constant(residual_logit), ())
                blend = nn.sigmoid(blend_logit)
                h = (1.0 - blend) * h + blend * mixed
            else:
                h = mixed
        return dense(1, "output")(h).squeeze(-1)


class VanillaMLP(nn.Module):
    """Standard fully connected MLP baseline for architecture ablations."""
    widths: Sequence[int]
    activation: str = "tanh"
    input_lower: Sequence[float] = (0.0, -1.0)
    input_upper: Sequence[float] = (1.0, 1.0)
    normalize_inputs: bool = True
    weight_factorization: bool = True
    weight_fact_mean: float = 0.5
    weight_fact_stddev: float = 0.1

    @nn.compact
    def __call__(self, inputs):
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"Unknown activation {self.activation!r}; choose from {tuple(ACTIVATIONS)}")
        act = ACTIVATIONS[self.activation]
        h = normalize_network_inputs(inputs, self.input_lower, self.input_upper, self.normalize_inputs)
        for layer_index, width in enumerate(self.widths):
            h = act(FactorizedDense(width, self.weight_factorization, self.weight_fact_mean,
                                    self.weight_fact_stddev, name=f"hidden_{layer_index}")(h))
        return FactorizedDense(1, self.weight_factorization, self.weight_fact_mean,
                               self.weight_fact_stddev, name="output")(h).squeeze(-1)


def build_model(cfg, input_lower, input_upper):
    if cfg.architecture == "modified":
        return ModifiedMLP(cfg.hidden, cfg.activation, input_lower, input_upper,
                           cfg.normalize_inputs, cfg.gate_activation, cfg.use_residual_gates,
                           cfg.residual_gate_init, cfg.weight_factorization,
                           cfg.weight_fact_mean, cfg.weight_fact_stddev)
    if cfg.architecture == "vanilla":
        return VanillaMLP(cfg.hidden, cfg.activation, input_lower, input_upper,
                          cfg.normalize_inputs, cfg.weight_factorization,
                          cfg.weight_fact_mean, cfg.weight_fact_stddev)
    raise ValueError("architecture must be 'modified' or 'vanilla'")


def mse(a, b):
    return jnp.mean((a - b) ** 2)


def normalized_rk4_mse(predicted, stepped, dt):
    """Return the squared RK4 defect rate, removing its artificial dt^2 scale."""
    floor = 10.0 * jnp.finfo(predicted.dtype).eps
    safe_dt = jnp.maximum(jnp.abs(dt), floor)
    return jnp.mean(((predicted - stepped) / safe_dt) ** 2)


def base_loss_weights(cfg, slab):
    """Build weights in ``LOSS_NAMES`` order and mask structurally inactive terms."""
    # Under the recommended policy, slab-one replay contains only the exact
    # initial condition and is therefore identical to L_initial.  Consolidate
    # that duplicate without changing the numerical objective.  Crucially,
    # this policy is independent of whether adaptive weighting is enabled.
    consolidate = cfg.deduplicate_replay and slab == 0
    previous = 0.0 if consolidate else cfg.previous_weight
    initial = cfg.initial_weight + (cfg.previous_weight if consolidate else 0.0)
    return np.asarray((
        cfg.residual_weight,
        cfg.rk4_weight if cfg.method == "fd_rk4" else 0.0,
        initial,  # The initial-condition coefficient remains a fixed anchor.
        cfg.boundary_weight,
        previous,
        cfg.data_weight if cfg.mode == "inverse" else 0.0,
    ), dtype=np.float64)


def project_adaptive_weights(base, log_multipliers, active, minimum, maximum):
    """Project bounded multipliers while preserving each semantic-group budget."""
    base = np.asarray(base, dtype=np.float64)
    log_multipliers = np.asarray(log_multipliers, dtype=np.float64)
    active = np.asarray(active, dtype=bool) & (base > 0.0)
    result = base.copy()
    lower_log, upper_log = np.log(minimum), np.log(maximum)
    for group in ADAPTIVE_GROUPS:
        indices = np.asarray(
            [index for index in group if active[index]], dtype=np.int64
        )
        if indices.size <= 1:
            continue
        budget = float(np.sum(base[indices]))

        def candidate(shift):
            multiplier = np.exp(np.clip(
                log_multipliers[indices] + shift, lower_log, upper_log
            ))
            return base[indices] * multiplier

        # The target budget is feasible because minimum <= 1 <= maximum.
        lower_shift, upper_shift = -128.0, 128.0
        for _ in range(80):
            midpoint = 0.5 * (lower_shift + upper_shift)
            if float(np.sum(candidate(midpoint))) < budget:
                lower_shift = midpoint
            else:
                upper_shift = midpoint
        result[indices] = candidate(0.5 * (lower_shift + upper_shift))
    return result


def initialize_adaptive_state(base):
    """Create host-side controller state; the state is reset at each slab."""
    base = np.asarray(base, dtype=np.float64)
    return {
        "weights": base.copy(),
        "log_multipliers": np.zeros_like(base),
        "loss_ema": np.full_like(base, np.nan),
        "gradient_ema": np.full_like(base, np.nan),
        "initial_loss": np.full_like(base, np.nan),
    }


def update_adaptive_state(state, base, losses, gradient_rms, cfg):
    """One bounded GradNorm-style controller update on the host."""
    base = np.asarray(base, dtype=np.float64)
    losses = np.asarray(losses, dtype=np.float64)
    gradient_rms = np.asarray(gradient_rms, dtype=np.float64)
    active = base > 0.0
    active[LOSS_NAMES.index("initial")] = False
    if getattr(cfg, "rk4_previous_controller", "grouped") == "inverse_loss":
        # These two are owned by the inverse-loss controller; the physics
        # group becomes the {residual} singleton (no-op) and the constraint
        # group balances only {boundary, data}.
        for name in _INVERSE_LOSS_NAMES:
            active[LOSS_NAMES.index(name)] = False
    observed = (active & np.isfinite(losses) & (losses >= 0.0)
                & np.isfinite(gradient_rms)
                & (gradient_rms > cfg.adaptive_gradient_epsilon))

    loss_ema = state["loss_ema"].copy()
    gradient_ema = state["gradient_ema"].copy()
    initial_loss = state["initial_loss"].copy()
    decay = cfg.adaptive_ema_decay
    for index in np.flatnonzero(observed):
        if np.isfinite(loss_ema[index]):
            loss_ema[index] = decay * loss_ema[index] + (1.0 - decay) * losses[index]
            gradient_ema[index] = (decay * gradient_ema[index]
                                   + (1.0 - decay) * gradient_rms[index])
        else:
            loss_ema[index] = losses[index]
            gradient_ema[index] = gradient_rms[index]
        if not np.isfinite(initial_loss[index]):
            initial_loss[index] = max(loss_ema[index], cfg.adaptive_gradient_epsilon)

    valid = (observed & np.isfinite(loss_ema) & np.isfinite(gradient_ema)
             & (gradient_ema > cfg.adaptive_gradient_epsilon)
             & np.isfinite(initial_loss) & (initial_loss > 0.0))
    log_multipliers = state["log_multipliers"].copy()
    weights = state["weights"].copy()
    for group in ADAPTIVE_GROUPS:
        indices = np.asarray(
            [index for index in group if active[index]], dtype=np.int64
        )
        if indices.size < 2:
            continue
        # Do not leave a stale extreme allocation in place when one member's
        # statistic becomes invalid.  Ease the complete semantic group back
        # toward its configured priorities until all members are observable.
        if not np.all(valid[indices]):
            correction = np.clip(
                -log_multipliers[indices], -np.log(2.0), np.log(2.0)
            )
            log_multipliers[indices] += cfg.adaptive_rate * correction
            group_mask = np.zeros_like(active)
            group_mask[indices] = True
            projected = project_adaptive_weights(
                base, log_multipliers, group_mask,
                cfg.adaptive_min_multiplier, cfg.adaptive_max_multiplier,
            )
            weights[indices] = projected[indices]
            log_multipliers[indices] = np.log(
                weights[indices] / base[indices]
            )
            continue
        group_mask = np.zeros_like(active)
        group_mask[indices] = True
        progress = np.clip(
            loss_ema[indices] / initial_loss[indices], 1e-8, 1e8
        )
        # Absolute target: w_i*g_i is proportional to the base priority b_i,
        # with a mild boost for terms learning more slowly.  Recomputing the
        # target from current statistics avoids hysteresis after a bound is hit.
        desired_log_multipliers = log_multipliers.copy()
        desired_log_multipliers[indices] = (
            cfg.adaptive_progress_power * np.log(progress)
            - np.log(gradient_ema[indices])
        )
        target_weights = project_adaptive_weights(
            base, desired_log_multipliers, group_mask,
            cfg.adaptive_min_multiplier, cfg.adaptive_max_multiplier,
        )
        target_log_multipliers = np.log(
            target_weights[indices] / base[indices]
        )
        correction = np.clip(
            target_log_multipliers - log_multipliers[indices],
            -np.log(2.0), np.log(2.0),
        )
        log_multipliers[indices] += cfg.adaptive_rate * correction
        projected = project_adaptive_weights(
            base, log_multipliers, group_mask,
            cfg.adaptive_min_multiplier, cfg.adaptive_max_multiplier,
        )
        # Store the projected logits.  Keeping the pre-projection values would
        # let them drift far beyond their bounds and prevent later recovery.
        weights[indices] = projected[indices]
        log_multipliers[indices] = np.log(
            weights[indices] / base[indices]
        )
    log_multipliers[~active] = 0.0

    return {
        "weights": weights,
        "log_multipliers": log_multipliers,
        "loss_ema": loss_ema,
        "gradient_ema": gradient_ema,
        "initial_loss": initial_loss,
    }


def select_replay_history(past_t, past_values, deduplicate):
    """Return strict-prior replay, or the original interface-inclusive history."""
    if deduplicate:
        return past_t[:-1], past_values[:-1]
    return past_t, past_values


def periodic_linear_resample(values, source_x, target_x):
    """Periodically interpolate state vectors without using reference values."""
    values = np.atleast_2d(np.asarray(values))
    source_x, target_x = np.asarray(source_x), np.asarray(target_x)
    period = float(np.median(np.diff(source_x)) * source_x.size)
    wrapped_target = (target_x - source_x[0]) % period + source_x[0]
    extended_x = np.concatenate((source_x, [source_x[0] + period]))
    return np.stack([
        np.interp(wrapped_target, extended_x, np.concatenate((state, state[:1])))
        for state in values
    ]).astype(np.float32)


def load_data(path):
    mat = scipy.io.loadmat(path)
    x, t, u = (np.asarray(mat[k]).squeeze() for k in ("x", "t", "usol"))
    if u.shape == (t.size, x.size):
        u = u.T
    if u.shape != (x.size, t.size):
        raise ValueError(f"usol has shape {u.shape}; expected {(x.size, t.size)}")
    return x.astype(np.float32), t.astype(np.float32), u.astype(np.float32)


def load_observations(path):
    """Load experimental sensors from ``.npz`` with t_obs, x_obs, u_obs arrays."""
    observations = np.load(path)
    required = {"t_obs", "x_obs", "u_obs"}
    missing = required.difference(observations.files)
    if missing:
        raise ValueError(f"{path} must contain {sorted(required)}; missing {sorted(missing)}")
    t_obs = np.asarray(observations["t_obs"], dtype=np.float32).ravel()
    x_obs = np.asarray(observations["x_obs"], dtype=np.float32).ravel()
    u_obs = np.asarray(observations["u_obs"], dtype=np.float32).ravel()
    if not (t_obs.size == x_obs.size == u_obs.size) or t_obs.size == 0:
        raise ValueError("t_obs, x_obs, and u_obs must be non-empty arrays of equal length")
    return t_obs, x_obs, u_obs


def checkpoint_signature(cfg):
    """Return settings that must match for a scientifically valid resume."""
    return {
        "mode": cfg.mode,
        "controlled_protocol": "fixed_updates_dynamic_weights_final_iterate_v2",
        "seed": cfg.seed,
        "temporal_integrator": cfg.temporal_integrator,
        "gradient_clip_norm": cfg.gradient_clip_norm,
        "lr_decay_spans_run": cfg.lr_decay_spans_run,
        "physics": (cfg.alpha, cfg.beta, cfg.nu),
        "initial_physics": (cfg.initial_alpha, cfg.initial_beta, cfg.initial_nu),
        "infer_parameters": tuple(cfg.infer_parameters),
        "spatial_points": cfg.spatial_points,
        "pairs_per_step": cfg.pairs_per_step,
        "epochs_per_slab": cfg.epochs,
        "optimizer": (cfg.learning_rate, cfg.parameter_learning_rate),
        "loss_weights": (LOSS_NAMES, (cfg.residual_weight, cfg.rk4_weight,
                                      cfg.initial_weight,
                                      cfg.boundary_weight, cfg.previous_weight,
                                      cfg.data_weight)),
        "normalize_rk4_defect": cfg.normalize_rk4_defect,
        "deduplicate_replay": cfg.deduplicate_replay,
        "adaptive_controller": (
            cfg.adaptive_weights, cfg.adaptive_update_every,
            cfg.adaptive_first_slab_warmup, cfg.adaptive_later_slab_warmup,
            cfg.adaptive_ema_decay, cfg.adaptive_rate,
            cfg.adaptive_progress_power, cfg.adaptive_min_multiplier,
            cfg.adaptive_max_multiplier, cfg.adaptive_gradient_epsilon,
            cfg.adaptive_max_temporal_gradient_ratio,
            ADAPTIVE_POLICY, ADAPTIVE_GROUP_NAMES,
        ),
        "observations": cfg.observations,
        "observation_path": cfg.observation_path,
    }


def save_checkpoint(path, *, params, opt_state, next_slab, past_t, past_values,
                    history, parameter_history, weight_history, adaptive_trace,
                    rng_state, cfg, controlled_state=None):
    """Atomically save all state needed to resume at the next time slab."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "controlled_state": controlled_state,
        "params": jax.device_get(params), "opt_state": jax.device_get(opt_state),
        "next_slab": next_slab, "past_t": np.asarray(past_t), "past_values": np.asarray(past_values),
        "history": np.asarray(history), "parameter_history": np.asarray(parameter_history),
        "weight_history": np.asarray(weight_history), "loss_weight_names": LOSS_NAMES,
        "adaptive_trace": np.asarray(adaptive_trace),
        "architecture": cfg.architecture, "method": cfg.method, "hidden": tuple(cfg.hidden), "slabs": cfg.slabs,
        "forecast_window": (cfg.train_end_time, cfg.forecast_end_time),
        "forecast_strategy": cfg.forecast_strategy,
        "rk4_substeps": cfg.rk4_substeps,
        "training_signature": checkpoint_signature(cfg),
        "rng_state": rng_state,
        "model_version": MODEL_VERSION,
        "model_options": (cfg.activation, cfg.normalize_inputs, cfg.gate_activation,
                          cfg.use_residual_gates, cfg.residual_gate_init,
                          cfg.weight_factorization, cfg.weight_fact_mean, cfg.weight_fact_stddev,
                          cfg.hard_initial_condition, cfg.hard_initial_transition),
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(destination)


def load_checkpoint(path, cfg):
    """Restore a compatible slab checkpoint created by :func:`save_checkpoint`."""
    with Path(path).open("rb") as stream:
        payload = pickle.load(stream)
    for key, expected in (("architecture", cfg.architecture), ("method", cfg.method),
                          ("hidden", tuple(cfg.hidden)), ("slabs", cfg.slabs),
                          ("model_version", MODEL_VERSION),
                          ("forecast_window", (cfg.train_end_time, cfg.forecast_end_time)),
                          ("forecast_strategy", cfg.forecast_strategy),
                          ("rk4_substeps", cfg.rk4_substeps),
                          ("training_signature", checkpoint_signature(cfg)),
                          ("model_options", (cfg.activation, cfg.normalize_inputs, cfg.gate_activation,
                                             cfg.use_residual_gates, cfg.residual_gate_init,
                                             cfg.weight_factorization, cfg.weight_fact_mean,
                                             cfg.weight_fact_stddev, cfg.hard_initial_condition,
                                             cfg.hard_initial_transition))):
        if payload.get(key) != expected:
            raise ValueError(f"Checkpoint {key}={payload.get(key)!r} is incompatible with current {key}={expected!r}")
    return payload


def make_update(model, cfg, x_grid, initial_profile, initial_time, parameter_template):
    """Construct the hybrid FD-in-space / RK4-in-time losses and optimiser."""
    dx = x_grid[1] - x_grid[0]

    # Smooth periodic interpolation of the initial state, so the hard initial
    # condition stays differentiable in x for the spatial stencils.
    wave_numbers = 2.0 * jnp.pi * jnp.fft.fftfreq(x_grid.size, d=dx)
    initial_modes = jnp.fft.fft(initial_profile) / x_grid.size

    def periodic_initial(x):
        phase = jnp.exp(1j * (x[..., None] - x_grid[0]) * wave_numbers)
        return jnp.real(jnp.sum(initial_modes * phase, axis=-1))

    def net(params, t, x):
        raw = model.apply(params["network"], jnp.stack((t, x), axis=-1))
        if cfg.hard_initial_condition:
            # Exponential blending is exact at t = t0 without forcing the raw
            # network to grow like 1/(t - t0) through the initial transient.
            blend = -jnp.expm1(-(t - initial_time) / cfg.hard_initial_transition)
            return (1.0 - blend) * periodic_initial(x) + blend * raw
        return raw

    def coefficients(params):
        """Return (alpha, beta, nu), training only the requested subset."""
        if cfg.mode == "inverse":
            estimated = jnp.exp(params["physics_raw"])
            fixed = jnp.asarray((cfg.alpha, cfg.beta, cfg.nu), dtype=jnp.float32)
            learned_mask = jnp.asarray(tuple(name in cfg.infer_parameters for name in PARAMETER_NAMES))
            return jnp.where(learned_mask, estimated, fixed)
        return jnp.asarray((cfg.alpha, cfg.beta, cfg.nu), dtype=jnp.float32)

    def grid_prediction(params, t):
        return net(params, jnp.full_like(x_grid, t), x_grid)

    def laplacian(u):
        return (jnp.roll(u, -1) - 2 * u + jnp.roll(u, 1)) / dx**2

    def rhs(u, physics):
        alpha, beta, nu = physics
        return -alpha * u**3 + beta * u + nu * laplacian(u)

    def integrate_state(u0, physics, dt, integrator):
        """Advance the whole periodic grid without clipping or value replacement."""
        step = dt / cfg.rk4_substeps

        def substep(_, state):
            k1 = rhs(state, physics)
            if integrator == "heun":
                k2 = rhs(state + step * k1, physics)
                return state + step * (k1 + k2) / 2
            k2 = rhs(state + step * k1 / 2, physics)
            k3 = rhs(state + step * k2 / 2, physics)
            k4 = rhs(state + step * k3, physics)
            return state + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6

        return lax.fori_loop(0, cfg.rk4_substeps, substep, u0)

    def rk4_state(u0, physics, dt):
        # Common continuation remains RK4 independently of the training penalty.
        return integrate_state(u0, physics, dt, "rk4")

    def rk4(params, t, dt):
        # Each stage is a whole spatial state: FD is therefore applied to all
        # staged states, matching the manuscript's semi-discrete RK4 scheme.
        u0 = grid_prediction(params, t)
        return integrate_state(u0, coefficients(params), dt, cfg.temporal_integrator)

    def pde_residual(params, t):
        u = grid_prediction(params, t)
        u_t = vmap(lambda xi: grad(net, argnums=1)(params, t, xi))(x_grid)
        if cfg.method == "vanilla":
            u_xx = vmap(lambda xi: grad(grad(lambda z: net(params, t, z)))(xi))(x_grid)
            alpha, beta, nu = coefficients(params)
            return u_t + alpha * u**3 - beta * u - nu * u_xx
        return u_t - rhs(u, coefficients(params))

    def periodic_bc(params, times):
        left, right = jnp.full_like(times, x_grid[0]), jnp.full_like(times, x_grid[0] + 2.0)
        value_loss = mse(net(params, times, left), net(params, times, right))
        u_x = grad(net, argnums=2)
        derivative_loss = mse(vmap(lambda ti: u_x(params, ti, left[0]))(times),
                              vmap(lambda ti: u_x(params, ti, right[0]))(times))
        return value_loss + derivative_loss

    # The optimiser state is created once and reused across slabs, so the
    # schedule's step counter spans the whole run. Decaying over cfg.epochs
    # therefore halves the rate once per slab: at 60 slabs the rate reaches
    # 1e-8 by slab 14 and later slabs cannot correct their hand-off at all.
    # Spanning the decay over the whole run matches ns_unsteady_cylinder.
    schedule_steps = max(cfg.epochs * cfg.slabs, 1) if cfg.lr_decay_spans_run else max(cfg.epochs, 1)
    network_schedule = optax.exponential_decay(cfg.learning_rate, schedule_steps, 0.5)
    labels = {
        "network": jax.tree_util.tree_map(lambda _: "network", parameter_template["network"]),
        "physics_raw": "physics",
    }
    optimizer = optax.multi_transform(
        {"network": optax.adam(network_schedule), "physics": optax.adam(cfg.parameter_learning_rate)}, labels
    )

    def loss_terms(p, pairs, initial_t, initial_values, past_t, past_values,
                   observation_t, observation_x, observation_values):
        """Return raw terms in the single canonical ``LOSS_NAMES`` order."""
        times, dts = pairs[:, 0], pairs[:, 1]
        zero = jnp.array(0.0, dtype=x_grid.dtype)
        # A zero base coefficient is permanently inactive: the adaptive
        # controller only redistributes positive base weights.  Avoid both the
        # work and the IEEE 0*NaN failure mode by not evaluating disabled terms.
        if cfg.residual_weight > 0.0:
            residual = jnp.mean(vmap(
                lambda ti: jnp.mean(pde_residual(p, ti) ** 2)
            )(times))
        else:
            residual = zero
        if cfg.method == "fd_rk4" and cfg.rk4_weight > 0.0:
            def pair_loss(ti, dt):
                predicted = grid_prediction(p, ti + dt)
                stepped = rk4(p, ti, dt)
                if cfg.normalize_rk4_defect:
                    return normalized_rk4_mse(predicted, stepped, dt)
                return mse(predicted, stepped)
            rk_loss = jnp.mean(vmap(pair_loss)(times, dts))
        else:
            rk_loss = zero
        initial = mse(grid_prediction(p, initial_t), initial_values)
        boundary = periodic_bc(p, times) if cfg.boundary_weight > 0.0 else zero
        if cfg.previous_weight == 0.0 or past_t.shape[0] == 0:
            previous = zero
        else:
            previous = jnp.mean(
                (vmap(lambda ti: grid_prediction(p, ti))(past_t) - past_values) ** 2
            )
        if cfg.mode == "inverse" and cfg.data_weight > 0.0:
            data = mse(net(p, observation_t, observation_x), observation_values)
        else:
            data = zero
        return jnp.stack((residual, rk_loss, initial, boundary, previous, data))

    @jit
    def update(params, opt_state, pairs, initial_t, initial_values, past_t, past_values,
               observation_t, observation_x, observation_values, loss_weights):
        def loss_fn(p):
            terms = loss_terms(
                p, pairs, initial_t, initial_values, past_t, past_values,
                observation_t, observation_x, observation_values,
            )
            total = jnp.vdot(lax.stop_gradient(loss_weights), terms)
            return total, (terms, coefficients(p))
        (loss, (terms, physics)), grads = value_and_grad(loss_fn, has_aux=True)(params)
        gradient_norm = optax.global_norm(grads)
        clipped = jnp.asarray(False)
        if cfg.gradient_clip_norm > 0:
            clipped = gradient_norm > cfg.gradient_clip_norm
            scale = jnp.minimum(1.0, cfg.gradient_clip_norm / jnp.maximum(gradient_norm, 1e-30))
            grads = jax.tree_util.tree_map(lambda g: g * scale, grads)
        updates, proposed_opt_state = optimizer.update(grads, opt_state, params)
        proposed_params = optax.apply_updates(params, updates)
        finite = jnp.isfinite(loss) & jnp.all(jnp.isfinite(terms)) & jnp.isfinite(gradient_norm)
        for leaf in jax.tree_util.tree_leaves((grads, proposed_params, proposed_opt_state)):
            finite = finite & jnp.all(jnp.isfinite(leaf))
        # The caller synchronises this flag and aborts before accepting a bad update.
        return proposed_params, proposed_opt_state, loss, terms, physics, (gradient_norm, clipped, finite)

    @jit
    def gradient_stats(params, pairs, initial_t, initial_values,
                       past_t, past_values, observation_t,
                       observation_x, observation_values):
        """Return loss and blockwise gradient diagnostics.

        The controller uses network-gradient RMS, the shared-parameter GradNorm
        statistic.  In inverse mode the separate ``physics_raw`` RMS is saved
        as a coefficient-identifiability diagnostic; mixing the two blocks
        blindly would be biased by their very different sizes and learning
        rates.
        """
        def from_parameters(candidate_params):
            terms = loss_terms(
                candidate_params, pairs, initial_t, initial_values,
                past_t, past_values,
                observation_t, observation_x, observation_values,
            )
            return terms, terms

        jacobian, terms = jax.jacrev(from_parameters, has_aux=True)(params)
        network_leaves = jax.tree_util.tree_leaves(jacobian["network"])
        network_squared_norms = sum(
            jnp.sum(leaf * leaf, axis=tuple(range(1, leaf.ndim)))
            for leaf in network_leaves
        )
        network_parameter_count = sum(
            int(np.prod(leaf.shape[1:])) for leaf in network_leaves
        )
        network_rms = jnp.sqrt(
            network_squared_norms / max(network_parameter_count, 1)
        )

        physics_jacobian = jacobian["physics_raw"]
        inferred_mask = jnp.asarray(
            tuple(cfg.mode == "inverse" and name in cfg.infer_parameters
                  for name in PARAMETER_NAMES),
            dtype=physics_jacobian.dtype,
        )
        physics_count = max(sum(
            cfg.mode == "inverse" and name in cfg.infer_parameters
            for name in PARAMETER_NAMES
        ), 1)
        physics_rms = jnp.sqrt(
            jnp.sum((physics_jacobian * inferred_mask) ** 2, axis=1)
            / physics_count
        )

        residual_index = LOSS_NAMES.index("residual")
        rk4_index = LOSS_NAMES.index("rk4")
        residual_rk4_dot = sum(
            jnp.sum(leaf[residual_index] * leaf[rk4_index])
            for leaf in network_leaves
        )
        cosine_floor = jnp.finfo(network_rms.dtype).tiny
        residual_rk4_cosine = jnp.clip(
            residual_rk4_dot / jnp.maximum(
                jnp.sqrt(network_squared_norms[residual_index]
                         * network_squared_norms[rk4_index]),
                cosine_floor,
            ),
            -1.0, 1.0,
        )
        return terms, network_rms, physics_rms, residual_rk4_cosine

    @jit
    def evaluate(params, pairs, initial_t, initial_values, past_t, past_values,
                 observation_t, observation_x, observation_values):
        terms = loss_terms(
            params, pairs, initial_t, initial_values, past_t, past_values,
            observation_t, observation_x, observation_values,
        )
        return terms, coefficients(params)

    return (update, evaluate, gradient_stats, grid_prediction, net,
            coefficients, optimizer, jit(rk4_state))


def train(cfg):
    """Train through ``train_end_time`` and forecast without future labels."""
    if cfg.adaptive_weights:
        physics_active = int(cfg.residual_weight > 0.0) + int(
            cfg.method == "fd_rk4" and cfg.rk4_weight > 0.0
        )
        if physics_active < 2:
            print(
                "Adaptive-weight note: the residual/RK4 group has fewer than "
                "two active terms, so its configured physics weights stay fixed."
            )
    x, t_full, reference_full = load_data(cfg.data_path)
    if np.any(np.diff(t_full) <= 0):
        raise ValueError("The time coordinates must be strictly increasing")
    tolerance = 1e-7
    if not (float(t_full.min()) <= cfg.train_end_time
            < cfg.forecast_end_time <= float(t_full.max()) + tolerance):
        raise ValueError(
            f"Need data covering train_end_time={cfg.train_end_time} and "
            f"forecast_end_time={cfg.forecast_end_time}; available range is "
            f"[{t_full.min()}, {t_full.max()}]"
        )

    evaluation_keep = t_full <= cfg.forecast_end_time + tolerance
    t = t_full[evaluation_keep]
    reference = reference_full[:, evaluation_keep]
    train_stop = int(np.flatnonzero(t <= cfg.train_end_time + tolerance)[-1])
    if train_stop < 1 or train_stop >= t.size - 1:
        raise ValueError("The selected training and forecast intervals need at least two snapshots")
    t_train = t[:train_stop + 1]
    # This is the only reference view available to the optimization code.
    reference_train = reference[:, :train_stop + 1]

    # Exclude x=1 because it duplicates x=-1 on the periodic finite-difference grid.
    nx = min(cfg.spatial_points, x.size - 1)
    x_grid_np = np.linspace(x.min(), x.max(), nx + 1, dtype=np.float32)[:-1]
    x_grid = jnp.asarray(x_grid_np)
    initial_exact = jnp.asarray(np.interp(x_grid_np, x, reference_train[:, 0]))
    model = build_model(
        cfg,
        (float(t_train.min()), float(x.min())),
        (float(t_train.max()), float(x.max())),
    )
    params = {
        "network": model.init(random.PRNGKey(cfg.seed), jnp.zeros((1, 2), dtype=jnp.float32)),
        "physics_raw": jnp.log(jnp.asarray(
            (cfg.initial_alpha, cfg.initial_beta, cfg.initial_nu), dtype=jnp.float32
        )),
    }
    (update, evaluate, gradient_stats, grid_prediction, net,
     coefficients, optimizer, rk4_state) = make_update(
        model, cfg, x_grid, initial_exact, jnp.asarray(t_train[0]), params)
    opt_state = optimizer.init(params)

    external_observations = None
    if cfg.mode == "inverse" and cfg.observation_path:
        observations = load_observations(cfg.observation_path)
        visible = observations[0] <= cfg.train_end_time + tolerance
        if not np.any(visible):
            raise ValueError("No supplied observations lie in the visible interval [0, 1]")
        external_observations = tuple(values[visible] for values in observations)

    wandb_run = None
    if cfg.use_wandb:
        try:
            import wandb
        except ImportError as error:
            raise ImportError("W&B monitoring needs the optional dependency: pip install wandb") from error
        wandb_run = wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=cfg.wandb_run_name,
            config={
                "equation": "u_t + alpha*u^3 - beta*u - nu*u_xx = 0",
                "forecast_protocol": "losses use t<=1; held-out state is advanced without labels",
                **cfg.__dict__,
            },
        )

    edges = np.linspace(0, train_stop, cfg.slabs + 1, dtype=int)
    if np.any(np.diff(edges) == 0):
        raise ValueError("Too many training slabs for the visible time snapshots")
    past_t, past_values = jnp.asarray([t_train[0]]), initial_exact[None, :]
    rng = np.random.default_rng(cfg.seed)
    history, parameter_history, weight_history, adaptive_trace = [], [], [], []
    start_slab = 0

    if cfg.resume_checkpoint:
        restored = load_checkpoint(cfg.resume_checkpoint, cfg)
        params, opt_state = restored["params"], restored["opt_state"]
        past_t = jnp.asarray(restored["past_t"])
        past_values = jnp.asarray(restored["past_values"])
        history = list(np.asarray(restored["history"]))
        parameter_history = list(np.asarray(restored["parameter_history"]))
        weight_history = list(np.asarray(restored["weight_history"]))
        adaptive_trace = list(np.asarray(restored["adaptive_trace"]))
        start_slab = int(restored["next_slab"])
        rng.bit_generator.state = restored["rng_state"]
        if not 0 <= start_slab <= cfg.slabs:
            raise ValueError(f"Invalid next_slab={start_slab} in checkpoint")
        if start_slab == cfg.slabs:
            print(f"Checkpoint {cfg.resume_checkpoint} already contains all training slabs")
        else:
            print(f"Resuming training at slab {start_slab + 1}/{cfg.slabs}")

    checkpoint_directory = Path(cfg.checkpoint_dir)
    for slab in range(start_slab, cfg.slabs):
        start, stop = edges[slab], edges[slab + 1]
        slab_t = t_train[start:stop + 1]
        dt = np.diff(slab_t)
        monitor_indices = np.linspace(
            0, dt.size - 1, min(cfg.pairs_per_step, dt.size), dtype=int
        )
        monitor_pairs = jnp.asarray(np.column_stack((
            slab_t[:-1][monitor_indices], dt[monitor_indices]
        )), dtype=jnp.float32)
        initial_values = initial_exact if slab == 0 else past_values[-1]
        # L_initial owns the current interface under the recommended policy.
        # In slab one, the strict-prior replay set is empty.
        replay_t, replay_values = select_replay_history(
            past_t, past_values, cfg.deduplicate_replay
        )
        base_weights = base_loss_weights(cfg, slab)
        adaptive_state = initialize_adaptive_state(base_weights)
        current_weights = adaptive_state["weights"].copy()
        warmup = (cfg.adaptive_first_slab_warmup if slab == 0
                  else cfg.adaptive_later_slab_warmup)
        # Inverse-loss controller state (reset per slab, like the grouped one).
        # The reference is frozen once the GradNorm warm-up has settled.
        use_inverse_loss = (cfg.adaptive_weights
                            and cfg.rk4_previous_controller == "inverse_loss"
                            and cfg.inverse_loss_normalize)
        inverse_loss_state = controllers.initialize_inverse_loss_state(len(_INVERSE_LOSS_INDICES))
        inverse_loss_calibrate_epoch = warmup + cfg.inverse_loss_warmup_epochs
        effective_weights = current_weights.copy()

        if cfg.mode == "forward":
            # Shape-stable dummy arrays; data_weight is exactly zero in forward mode.
            observation_t = jnp.full((cfg.observations,), slab_t[0])
            observation_x = jnp.full((cfg.observations,), x[0])
            observation_values = jnp.zeros((cfg.observations,), dtype=jnp.float32)
        elif external_observations is None:
            observation_count = min(cfg.observations, slab_t.size * x.size)
            obs_t_idx = rng.integers(start, stop + 1, size=observation_count)
            obs_x_idx = rng.integers(0, x.size, size=observation_count)
            observation_t = jnp.asarray(t_train[obs_t_idx])
            observation_x = jnp.asarray(x[obs_x_idx])
            observation_values = jnp.asarray(reference_train[obs_x_idx, obs_t_idx])
        else:
            all_t, all_x, all_u = external_observations
            in_slab = (all_t >= slab_t[0]) & (all_t <= slab_t[-1])
            available = np.flatnonzero(in_slab)
            if available.size == 0:
                raise ValueError(f"No supplied observations fall in [{slab_t[0]}, {slab_t[-1]}]")
            chosen = rng.choice(
                available, size=cfg.observations, replace=available.size < cfg.observations
            )
            observation_t = jnp.asarray(all_t[chosen])
            observation_x = jnp.asarray(all_x[chosen])
            observation_values = jnp.asarray(all_u[chosen])

        best_loss, best_params, best_opt_state = np.inf, params, opt_state
        best_weights = effective_weights.copy()
        ema_loss, best_ema_loss = None, np.inf
        epochs_without_improvement = 0
        epochs_completed = cfg.epochs
        progress = trange(cfg.epochs, desc=f"train slab {slab + 1}/{cfg.slabs}", leave=False)
        for epoch in progress:
            pick = rng.integers(0, dt.size, size=cfg.pairs_per_step)
            pairs = jnp.asarray(
                np.column_stack((slab_t[:-1][pick], dt[pick])), dtype=jnp.float32
            )
            if (cfg.adaptive_weights and epoch >= warmup
                    and (epoch - warmup) % cfg.adaptive_update_every == 0):
                (sampled_losses, gradient_rms, physics_gradient_rms,
                 residual_rk4_gradient_cosine) = gradient_stats(
                    params, monitor_pairs, jnp.asarray(slab_t[0]), initial_values,
                    replay_t, replay_values, observation_t, observation_x,
                    observation_values,
                )
                adaptive_state = update_adaptive_state(
                    adaptive_state, base_weights, np.asarray(sampled_losses),
                    np.asarray(gradient_rms), cfg,
                )
                current_weights = adaptive_state["weights"].copy()
                adaptive_trace.append(np.concatenate((
                    np.asarray((slab + 1, epoch + 1), dtype=np.float64),
                    current_weights, np.asarray(sampled_losses, dtype=np.float64),
                    np.asarray(gradient_rms, dtype=np.float64),
                    np.asarray(physics_gradient_rms, dtype=np.float64),
                    np.atleast_1d(np.asarray(
                        residual_rk4_gradient_cosine, dtype=np.float64
                    )),
                )))

            # Grouped GradNorm sets current_weights; in inverse_loss mode the
            # rk4/previous entries are additionally scaled by their own
            # coefficient. effective_weights is what the optimiser sees.
            if use_inverse_loss:
                effective_weights = controllers.apply_inverse_loss_coefficients(
                    current_weights, _INVERSE_LOSS_INDICES, inverse_loss_state
                )
            else:
                effective_weights = current_weights
            params, opt_state, training_loss, _, _ = update(
                params, opt_state, pairs, jnp.asarray(slab_t[0]), initial_values,
                replay_t, replay_values, observation_t, observation_x,
                observation_values, jnp.asarray(effective_weights, dtype=jnp.float32),
            )
            terms, physics = evaluate(
                params, monitor_pairs, jnp.asarray(slab_t[0]), initial_values,
                replay_t, replay_values, observation_t, observation_x,
                observation_values,
            )
            if use_inverse_loss:
                calibrate_now = epoch == inverse_loss_calibrate_epoch
                if epoch % max(1, cfg.inverse_loss_update_every) == 0 or calibrate_now:
                    inverse_loss_state = controllers.update_inverse_loss_state(
                        inverse_loss_state, np.asarray(terms), _INVERSE_LOSS_INDICES, cfg,
                        calibrate=calibrate_now,
                    )
            loss_value = float(np.dot(effective_weights, np.asarray(terms)))
            monitor_value = float(np.dot(base_weights, np.asarray(terms)))
            if monitor_value < best_loss:
                best_loss = monitor_value
                best_params, best_opt_state = params, opt_state
                best_weights = effective_weights.copy()
            ema_loss = (monitor_value if ema_loss is None else
                        cfg.early_stopping_ema_decay * ema_loss
                        + (1.0 - cfg.early_stopping_ema_decay) * monitor_value)
            if epoch % max(1, cfg.epochs // 10) == 0:
                progress.set_postfix(train=f"{float(training_loss):.2e}",
                                     ref=f"{monitor_value:.2e}",
                                     rk4=f"{float(terms[1]):.2e}")
            if wandb_run is not None and (epoch % cfg.log_every == 0 or epoch == cfg.epochs - 1):
                raw_terms = np.asarray(terms, dtype=float)
                log_values = {
                    "phase": "train", "slab": slab + 1, "epoch_in_slab": epoch + 1,
                    "loss/train_batch": float(training_loss),
                    "loss/adaptive_monitor": loss_value,
                    "loss/fixed_monitor": monitor_value,
                    "physics/alpha": float(physics[0]), "physics/beta": float(physics[1]),
                    "physics/nu": float(physics[2]),
                }
                for name, raw, weight in zip(LOSS_NAMES, raw_terms, current_weights):
                    log_values[f"loss/{name}"] = float(raw)
                    log_values[f"weight/{name}"] = float(weight)
                    log_values[f"weighted_loss/{name}"] = float(weight * raw)
                wandb_run.log(log_values)
            if cfg.early_stopping and epoch + 1 >= cfg.early_stopping_warmup:
                if ema_loss < best_ema_loss - cfg.early_stopping_min_delta:
                    best_ema_loss = ema_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                if epochs_without_improvement >= cfg.early_stopping_patience:
                    epochs_completed = epoch + 1
                    break

        params, opt_state = best_params, best_opt_state
        current_t = jnp.asarray(slab_t)
        current_values = vmap(lambda ti: grid_prediction(params, ti))(current_t)
        if slab == 0:
            past_t, past_values = current_t, current_values
        else:
            past_t = jnp.concatenate((past_t, current_t[1:]))
            past_values = jnp.concatenate((past_values, current_values[1:]))
        history.append(best_loss)
        weight_history.append(best_weights.copy())
        learned = tuple(float(value) for value in coefficients(params))
        parameter_history.append(learned)
        print(f"train slab {slab + 1:02d}: best loss {best_loss:.3e}; "
              f"epochs={epochs_completed}; alpha={learned[0]:.5g}, "
              f"beta={learned[1]:.5g}, nu={learned[2]:.5g}; "
              f"weights={dict(zip(LOSS_NAMES, np.round(best_weights, 5)))}")
        if (slab + 1) % cfg.checkpoint_every == 0 or slab + 1 == cfg.slabs:
            checkpoint_args = dict(
                params=params, opt_state=opt_state, next_slab=slab + 1,
                past_t=past_t, past_values=past_values, history=history,
                parameter_history=parameter_history, weight_history=weight_history,
                adaptive_trace=adaptive_trace,
                rng_state=rng.bit_generator.state, cfg=cfg,
            )
            save_checkpoint(checkpoint_directory / f"checkpoint_slab_{slab + 1:03d}.pkl",
                            **checkpoint_args)
            save_checkpoint(checkpoint_directory / "latest.pkl", **checkpoint_args)

    # Frozen-network extrapolation is saved as a diagnostic for both methods.
    full_tt, full_xx = np.meshgrid(t, x, indexing="xy")
    direct_prediction = np.asarray(vmap(lambda ti, xi: net(params, ti, xi))(
        jnp.asarray(full_tt.ravel()), jnp.asarray(full_xx.ravel())
    )).reshape(full_xx.shape)

    prediction = direct_prediction.copy()
    if cfg.forecast_strategy == "rk4_rollout":
        # No optimizer and no reference values are used after this point.
        physics = coefficients(params)
        state = grid_prediction(params, jnp.asarray(t[train_stop]))
        future_states = []
        for index in range(train_stop, t.size - 1):
            state = rk4_state(state, physics, jnp.asarray(t[index + 1] - t[index]))
            state_host = np.asarray(state)
            if not np.all(np.isfinite(state_host)):
                raise FloatingPointError(
                    "The FD-RK4 forecast became non-finite; increase --rk4-substeps"
                )
            future_states.append(state_host)
        if future_states:
            forecast_on_data_grid = periodic_linear_resample(
                np.stack(future_states), x_grid_np, x
            )
            prediction[:, train_stop + 1:] = forecast_on_data_grid.T

    learned_parameters = np.asarray(coefficients(params))
    return (x, t, reference, prediction, direct_prediction, np.asarray(history),
            learned_parameters, np.asarray(parameter_history),
            np.asarray(weight_history), np.asarray(adaptive_trace), wandb_run)


def save(cfg, x, t, reference, prediction, direct_prediction, history,
         learned_parameters, parameter_history, weight_history, adaptive_trace,
         wandb_run=None):
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    error = prediction - reference
    direct_error = direct_prediction - reference
    tolerance = 1e-7
    training_mask = t <= cfg.train_end_time + tolerance
    forecast_mask = ((t > cfg.train_end_time + tolerance)
                     & (t <= cfg.forecast_end_time + tolerance))
    if not np.any(training_mask) or not np.any(forecast_mask):
        raise ValueError("Saved results require non-empty training and forecast intervals")

    def relative_l2(values, mask):
        denominator = max(np.linalg.norm(reference[:, mask]), np.finfo(float).eps)
        return float(np.linalg.norm(values[:, mask] - reference[:, mask]) / denominator)

    train_relative_l2 = relative_l2(prediction, training_mask)
    forecast_relative_l2 = relative_l2(prediction, forecast_mask)
    total_relative_l2 = relative_l2(prediction, training_mask | forecast_mask)
    direct_forecast_relative_l2 = relative_l2(direct_prediction, forecast_mask)
    forecast_rmse = float(np.sqrt(np.mean(error[:, forecast_mask] ** 2)))
    forecast_mae = float(np.mean(np.abs(error[:, forecast_mask])))
    comparison_parameters = np.asarray((cfg.alpha, cfg.beta, cfg.nu), dtype=float)
    configured_weights = np.asarray((cfg.residual_weight, cfg.rk4_weight,
                                     cfg.initial_weight,
                                     cfg.boundary_weight, cfg.previous_weight,
                                     cfg.data_weight), dtype=float)
    trace_columns = np.asarray(
        ("slab", "epoch")
        + tuple(f"weight/{name}" for name in LOSS_NAMES)
        + tuple(f"loss/{name}" for name in LOSS_NAMES)
        + tuple(f"network_gradient_rms/{name}" for name in LOSS_NAMES)
        + tuple(f"physics_gradient_rms/{name}" for name in LOSS_NAMES)
        + ("network_gradient_cosine/residual_rk4",)
    )
    adaptive_trace = np.asarray(adaptive_trace, dtype=np.float64).reshape(
        -1, trace_columns.size
    )
    np.savez(out / "allen_cahn_fd_rk4_pinn.npz", x=x, t=t, prediction=prediction, reference=reference,
             direct_prediction=direct_prediction, training_mask=training_mask,
             forecast_mask=forecast_mask, train_end_time=cfg.train_end_time,
             forecast_end_time=cfg.forecast_end_time,
             prediction_mode=np.asarray(cfg.forecast_strategy),
             train_relative_l2=train_relative_l2,
             forecast_relative_l2=forecast_relative_l2,
             direct_forecast_relative_l2=direct_forecast_relative_l2,
             total_relative_l2=total_relative_l2,
             forecast_rmse=forecast_rmse, forecast_mae=forecast_mae,
             slab_loss=history, parameter_history=parameter_history, inferred_parameters=np.asarray(cfg.infer_parameters),
             loss_weight_names=np.asarray(LOSS_NAMES), configured_loss_weights=configured_weights,
             # Compatibility alias: these are the coefficients at the selected
             # best monitored checkpoint, not necessarily at the final epoch.
             slab_final_loss_weights=weight_history,
             slab_best_loss_weights=weight_history, adaptive_trace=adaptive_trace,
             adaptive_trace_columns=trace_columns,
             adaptive_group_names=np.asarray(
                 [" + ".join(group) for group in ADAPTIVE_GROUP_NAMES]
             ),
             adaptive_policy=np.asarray(ADAPTIVE_POLICY),
             adaptive_weights=np.asarray(cfg.adaptive_weights),
             normalize_rk4_defect=np.asarray(cfg.normalize_rk4_defect),
             deduplicate_replay=np.asarray(cfg.deduplicate_replay),
             seed=np.asarray(cfg.seed), model_version=np.asarray(MODEL_VERSION),
             alpha=learned_parameters[0], beta=learned_parameters[1], nu=learned_parameters[2])
    metrics = {
        "training_interval": [float(t[training_mask][0]), float(t[training_mask][-1])],
        "forecast_interval": [float(t[forecast_mask][0]), float(t[forecast_mask][-1])],
        "prediction_mode": cfg.forecast_strategy,
        "train_relative_l2": train_relative_l2,
        "forecast_relative_l2": forecast_relative_l2,
        "direct_forecast_relative_l2": direct_forecast_relative_l2,
        "total_relative_l2": total_relative_l2,
        "forecast_rmse": forecast_rmse,
        "forecast_mae": forecast_mae,
        "loss_weight_names": list(LOSS_NAMES),
        "configured_loss_weights": configured_weights.tolist(),
        "slab_final_loss_weights": np.asarray(weight_history).tolist(),
        "slab_best_loss_weights": np.asarray(weight_history).tolist(),
        "adaptive_groups": [list(group) for group in ADAPTIVE_GROUP_NAMES],
        "adaptive_policy": ADAPTIVE_POLICY,
        "adaptive_weights": cfg.adaptive_weights,
        "normalize_rk4_defect": cfg.normalize_rk4_defect,
        "deduplicate_replay": cfg.deduplicate_replay,
        "seed": cfg.seed,
        "model_version": MODEL_VERSION,
    }
    (out / "forecast_metrics.json").write_text(json.dumps(metrics, indent=2))
    np.savetxt(
        out / "forecast_metrics.csv",
        np.asarray([[train_relative_l2, forecast_relative_l2,
                     direct_forecast_relative_l2, total_relative_l2,
                     forecast_rmse, forecast_mae]]),
        delimiter=",",
        header=("train_relative_l2,forecast_relative_l2,direct_forecast_relative_l2,"
                "total_relative_l2,forecast_rmse,forecast_mae"),
        comments="",
    )
    for name, field, label in (("prediction", prediction, "u"), ("absolute_error", np.abs(error), "|error|")):
        fig, ax = plt.subplots(figsize=(7, 4))
        image = ax.pcolormesh(t, x, field, shading="auto", cmap="jet")
        fig.colorbar(image, ax=ax, label=label)
        ax.axvline(cfg.train_end_time, color="white", ls="--", lw=1.5,
                   label="forecast starts")
        ax.set(xlabel="t", ylabel="x", title=name.replace("_", " ").title())
        ax.legend(loc="upper right")
        fig.tight_layout(); fig.savefig(out / f"{name}.png", dpi=200); plt.close(fig)

    forecast_t = t[forecast_mask]
    for name, field, label in (
        ("forecast_prediction", prediction[:, forecast_mask], "u"),
        ("forecast_absolute_error", np.abs(error[:, forecast_mask]), "|forecast error|"),
    ):
        fig, ax = plt.subplots(figsize=(7, 4))
        image = ax.pcolormesh(forecast_t, x, field, shading="auto", cmap="jet")
        fig.colorbar(image, ax=ax, label=label)
        ax.set(xlabel="held-out forecast time", ylabel="x",
               title=name.replace("_", " ").title())
        fig.tight_layout(); fig.savefig(out / f"{name}.png", dpi=200); plt.close(fig)

    profile_times = (cfg.train_end_time,
                     cfg.train_end_time + 0.5 * (cfg.forecast_end_time - cfg.train_end_time),
                     cfg.forecast_end_time)
    profile_indices = tuple(int(np.argmin(np.abs(t - value))) for value in profile_times)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    for axis, time_index in zip(axes, profile_indices):
        axis.plot(x, reference[:, time_index], label="reference", linewidth=2)
        axis.plot(x, prediction[:, time_index], "--", label=cfg.forecast_strategy)
        axis.set(xlabel="x", title=f"t = {t[time_index]:.3g}")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("u")
    axes[-1].legend()
    fig.suptitle("Forecast profiles: all losses stop at t = " + f"{cfg.train_end_time:g}")
    fig.tight_layout(); fig.savefig(out / "forecast_profiles.png", dpi=200); plt.close(fig)

    # Convergence diagnostics are useful in both modes; in inverse mode they
    # show whether the selected coefficients have stabilised.
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(np.arange(1, history.size + 1), history, marker="o")
    ax.set(xlabel="time slab", ylabel="best training loss", title="Slabwise training convergence")
    ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(out / "training_loss.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for index, name in enumerate(LOSS_NAMES):
        values = np.asarray(weight_history)[:, index]
        if np.any(values > 0.0):
            ax.semilogy(np.arange(1, values.size + 1),
                        np.where(values > 0.0, values, np.nan), marker="o", label=name)
    ax.set(xlabel="time slab", ylabel="best-checkpoint effective weight",
           title="Selected loss coefficients")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "adaptive_weights.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for index, name in enumerate(PARAMETER_NAMES):
        style = "-" if name in cfg.infer_parameters and cfg.mode == "inverse" else "--"
        ax.semilogy(np.arange(1, parameter_history.shape[0] + 1), parameter_history[:, index],
                    style, marker="o", label=f"{name}{' (inferred)' if style == '-' else ' (fixed)'}")
    ax.set(xlabel="time slab", ylabel="coefficient value", title="Physics-parameter estimates")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(out / "parameter_estimates.png", dpi=200); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    for axis, time_index in zip(axes, (0, t.size // 2, t.size - 1)):
        axis.plot(x, reference[:, time_index], label="reference", linewidth=2)
        axis.plot(x, prediction[:, time_index], "--", label="PINN")
        axis.set(xlabel="x", title=f"t = {t[time_index]:.3g}")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("u")
    axes[-1].legend()
    fig.suptitle("Solution profiles")
    fig.tight_layout(); fig.savefig(out / "solution_profiles.png", dpi=200); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    for axis, time_index in zip(axes, (0, t.size // 2, t.size - 1)):
        axis.plot(x, np.abs(error[:, time_index]), color="crimson", linewidth=2)
        axis.set(xlabel="x", title=f"|error| at t = {t[time_index]:.3g}")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("absolute error")
    fig.suptitle("Spatial reconstruction error profiles")
    fig.tight_layout(); fig.savefig(out / "solution_profile_errors.png", dpi=200); plt.close(fig)

    # In inverse mode, cfg.alpha/beta/nu are comparison values for inferred
    # parameters (and fixed values for parameters excluded from --infer).
    if cfg.mode == "inverse":
        relative_parameter_error = 100.0 * (learned_parameters - comparison_parameters) / comparison_parameters
        np.savetxt(
            out / "parameter_estimates.csv",
            np.column_stack((comparison_parameters, learned_parameters, learned_parameters - comparison_parameters,
                             relative_parameter_error)),
            delimiter=",", header="reference,estimated,difference,relative_error_percent", comments="",
        )

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        positions = np.arange(len(PARAMETER_NAMES))
        width = 0.38
        axes[0].bar(positions - width / 2, comparison_parameters, width, label="comparison reference")
        axes[0].bar(positions + width / 2, learned_parameters, width, label="estimated")
        axes[0].set_yscale("log")
        axes[0].set_xticks(positions, PARAMETER_NAMES)
        axes[0].set_ylabel("coefficient value (log scale)")
        axes[0].set_title("Reference versus estimated parameters")
        axes[0].legend()
        axes[1].bar(positions, relative_parameter_error, color=["steelblue" if name in cfg.infer_parameters else "0.65" for name in PARAMETER_NAMES])
        axes[1].axhline(0.0, color="black", linewidth=0.8)
        axes[1].set_xticks(positions, PARAMETER_NAMES)
        axes[1].set_ylabel("relative difference (%)")
        axes[1].set_title("Final parameter difference")
        for position, value in zip(positions, relative_parameter_error):
            axes[1].annotate(f"{value:.2g}%", (position, value), ha="center", va="bottom" if value >= 0 else "top")
        fig.tight_layout(); fig.savefig(out / "parameter_comparison.png", dpi=200); plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        slab_indices = np.arange(1, parameter_history.shape[0] + 1)
        for index, name in enumerate(PARAMETER_NAMES):
            if name in cfg.infer_parameters:
                evolution = 100.0 * (parameter_history[:, index] - comparison_parameters[index]) / comparison_parameters[index]
                ax.plot(slab_indices, evolution, marker="o", label=name)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set(xlabel="time slab", ylabel="relative difference (%)", title="Parameter-estimation error by slab")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(out / "parameter_error_history.png", dpi=200); plt.close(fig)
    if wandb_run is not None:
        import wandb
        wandb_run.log({
            "final/train_relative_l2_error": train_relative_l2,
            "final/forecast_relative_l2_error": forecast_relative_l2,
            "final/direct_forecast_relative_l2_error": direct_forecast_relative_l2,
            "final/total_relative_l2_error": total_relative_l2,
            "final/mse": np.mean(error ** 2),
            "final/alpha": learned_parameters[0],
            "final/beta": learned_parameters[1],
            "final/nu": learned_parameters[2],
            "final/prediction": wandb.Image(str(out / "prediction.png")),
            "final/absolute_error": wandb.Image(str(out / "absolute_error.png")),
            "final/parameter_estimates": wandb.Image(str(out / "parameter_estimates.png")),
            "final/training_loss": wandb.Image(str(out / "training_loss.png")),
            "final/solution_profiles": wandb.Image(str(out / "solution_profiles.png")),
            "final/solution_profile_errors": wandb.Image(str(out / "solution_profile_errors.png")),
            "final/forecast_prediction": wandb.Image(str(out / "forecast_prediction.png")),
            "final/forecast_absolute_error": wandb.Image(str(out / "forecast_absolute_error.png")),
            "final/forecast_profiles": wandb.Image(str(out / "forecast_profiles.png")),
        })
        if cfg.mode == "inverse":
            wandb_run.log({"final/parameter_comparison": wandb.Image(str(out / "parameter_comparison.png")),
                           "final/parameter_error_history": wandb.Image(str(out / "parameter_error_history.png"))})
        wandb_run.finish()
    print(f"Training relative L2 error [0, {cfg.train_end_time:g}]: {train_relative_l2:.4e}")
    print(f"Forecast relative L2 error ({cfg.train_end_time:g}, {cfg.forecast_end_time:g}]: "
          f"{forecast_relative_l2:.4e}")
    print(f"Frozen-network direct forecast relative L2 error: {direct_forecast_relative_l2:.4e}")
    print("Estimated physics: "
          f"alpha={learned_parameters[0]:.8g}, beta={learned_parameters[1]:.8g}, nu={learned_parameters[2]:.8g}")
    print(f"Saved results to {out.resolve()}")


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", choices=("forward", "inverse"), default=SELECTED_CONFIG,
                   help="Select forward_adaptive_config.py or inverse_adaptive_config.py defaults.")
    p.add_argument("--data", default=Config.data_path); p.add_argument("--output", default=Config.output_dir)
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--epochs", type=int, default=Config.epochs); p.add_argument("--slabs", type=int, default=Config.slabs)
    p.add_argument("--pairs-per-step", type=int, default=Config.pairs_per_step)
    p.add_argument("--train-end-time", type=float, default=Config.train_end_time,
                   help="Last time allowed to enter any optimization loss.")
    p.add_argument("--forecast-end-time", type=float, default=Config.forecast_end_time,
                   help="End of the held-out evaluation interval.")
    p.add_argument("--forecast-strategy", choices=("rk4_rollout", "direct"),
                   default=Config.forecast_strategy,
                   help="Advance the frozen state with physics or directly query the frozen network.")
    p.add_argument("--early-stopping", action=argparse.BooleanOptionalAction,
                   default=Config.early_stopping)
    p.add_argument("--early-stopping-warmup", type=int, default=Config.early_stopping_warmup)
    p.add_argument("--early-stopping-patience", type=int, default=Config.early_stopping_patience)
    p.add_argument("--early-stopping-min-delta", type=float, default=Config.early_stopping_min_delta)
    p.add_argument("--residual-weight", type=float, default=Config.residual_weight)
    p.add_argument("--rk4-weight", type=float, default=Config.rk4_weight)
    p.add_argument("--initial-weight", type=float, default=Config.initial_weight,
                   help="Fixed initial/interface anchor coefficient.")
    p.add_argument("--boundary-weight", type=float, default=Config.boundary_weight)
    p.add_argument("--previous-weight", type=float, default=Config.previous_weight)
    p.add_argument("--deduplicate-replay", action=argparse.BooleanOptionalAction,
                   default=Config.deduplicate_replay,
                   help="Exclude the current interface from replay (recommended).")
    p.add_argument("--spatial-points", type=int, default=Config.spatial_points)
    p.add_argument("--rk4-substeps", type=int, default=Config.rk4_substeps,
                   help="RK4 substeps inside each saved time interval.")
    p.add_argument("--normalize-rk4-defect", action=argparse.BooleanOptionalAction,
                   default=Config.normalize_rk4_defect,
                   help="Divide each RK4 state defect by its dt before taking MSE.")
    p.add_argument("--adaptive-weights", action=argparse.BooleanOptionalAction,
                   default=Config.adaptive_weights)
    p.add_argument("--adaptive-update-every", type=int, default=Config.adaptive_update_every)
    p.add_argument("--adaptive-first-slab-warmup", type=int,
                   default=Config.adaptive_first_slab_warmup)
    p.add_argument("--adaptive-later-slab-warmup", type=int,
                   default=Config.adaptive_later_slab_warmup)
    p.add_argument("--adaptive-ema-decay", type=float, default=Config.adaptive_ema_decay)
    p.add_argument("--adaptive-rate", type=float, default=Config.adaptive_rate)
    p.add_argument("--adaptive-progress-power", type=float,
                   default=Config.adaptive_progress_power)
    p.add_argument("--adaptive-min-multiplier", type=float,
                   default=Config.adaptive_min_multiplier)
    p.add_argument("--adaptive-max-multiplier", type=float,
                   default=Config.adaptive_max_multiplier)
    p.add_argument("--adaptive-gradient-epsilon", type=float,
                   default=Config.adaptive_gradient_epsilon)
    p.add_argument(
        "--adaptive-max-temporal-gradient-ratio", type=float,
        default=Config.adaptive_max_temporal_gradient_ratio,
        help=("Safeguard cap for the weighted RK4-to-PDE network-gradient ratio. "
              "Use 1e30 to retain the grouped controller without the safeguard."),
    )
    p.add_argument("--rk4-previous-controller", choices=("grouped", "inverse_loss"),
                   default=Config.rk4_previous_controller,
                   help="'grouped' (this file's native grouped/priority-aware GradNorm) or "
                        "'inverse_loss' (rk4/previous get their own inverse-loss coefficient; "
                        "the constraint group then balances only boundary/data). Both need "
                        "--adaptive-weights.")
    p.add_argument("--inverse-loss-normalize", action=argparse.BooleanOptionalAction,
                   default=Config.inverse_loss_normalize)
    p.add_argument("--inverse-loss-ema-decay", type=float, default=Config.inverse_loss_ema_decay)
    p.add_argument("--inverse-loss-power", type=float, default=Config.inverse_loss_power)
    p.add_argument("--inverse-loss-min", type=float, default=Config.inverse_loss_min)
    p.add_argument("--inverse-loss-max", type=float, default=Config.inverse_loss_max)
    p.add_argument("--inverse-loss-warmup-epochs", type=int, default=Config.inverse_loss_warmup_epochs)
    p.add_argument("--inverse-loss-update-every", type=int, default=Config.inverse_loss_update_every)
    p.add_argument("--inverse-loss-epsilon", type=float, default=Config.inverse_loss_epsilon)
    p.add_argument("--architecture", choices=("modified", "vanilla"), default=Config.architecture)
    p.add_argument("--activation", choices=tuple(ACTIVATIONS), default=Config.activation)
    p.add_argument("--gate-activation", choices=("sigmoid", "tanh"), default=Config.gate_activation)
    p.add_argument("--residual-gate-init", type=float, default=Config.residual_gate_init)
    p.add_argument("--input-normalization", action=argparse.BooleanOptionalAction,
                   default=Config.normalize_inputs)
    p.add_argument("--residual-gates", action=argparse.BooleanOptionalAction,
                   default=Config.use_residual_gates)
    p.add_argument("--weight-factorization", action=argparse.BooleanOptionalAction,
                   default=Config.weight_factorization)
    p.add_argument("--lr-decay-spans-run", action=argparse.BooleanOptionalAction,
                   default=Config.lr_decay_spans_run,
                   help="Halve the learning rate once over the whole run (default) rather than once per slab; "
                        "pass --no-lr-decay-spans-run to reproduce runs archived before 2026-09-10.")
    p.add_argument("--hard-initial-condition", action=argparse.BooleanOptionalAction,
                   default=Config.hard_initial_condition,
                   help="Blend the exact initial state into the network output near t0.")
    p.add_argument("--hard-initial-transition", type=float, default=Config.hard_initial_transition,
                   help="Time constant of that blend (larger = anchored for longer).")
    p.add_argument("--method", choices=("vanilla", "fd", "fd_rk4"), default=Config.method)
    p.add_argument("--hidden", nargs="+", type=int, default=None,
                   help="Hidden widths, e.g. --hidden 64 64 64 64.")
    p.add_argument("--layers", type=int, default=None, help="Number of equal-width hidden layers.")
    p.add_argument("--neurons", type=int, default=None, help="Neurons in every layer when --layers is used.")
    p.add_argument("--mode", choices=("forward", "inverse"), default=Config.mode)
    p.add_argument("--alpha", type=float, default=Config.alpha, help="Forward coefficient; inverse comparison reference or fixed value.")
    p.add_argument("--beta", type=float, default=Config.beta, help="Forward coefficient; inverse comparison reference or fixed value.")
    p.add_argument("--nu", type=float, default=Config.nu, help="Forward coefficient; inverse comparison reference or fixed value.")
    p.add_argument("--observations", type=int, default=Config.observations, help="Sparse field observations per time slab in inverse mode.")
    p.add_argument("--observation-file", default=Config.observation_path,
                   help="Optional .npz sensor file with t_obs, x_obs, u_obs arrays for inverse mode.")
    p.add_argument("--initial-alpha", type=float, default=Config.initial_alpha)
    p.add_argument("--initial-beta", type=float, default=Config.initial_beta)
    p.add_argument("--initial-nu", type=float, default=Config.initial_nu)
    p.add_argument("--data-weight", type=float, default=Config.data_weight)
    p.add_argument("--infer", nargs="+", choices=PARAMETER_NAMES, default=Config.infer_parameters,
                   help="Coefficients to estimate in inverse mode; the remaining coefficients remain fixed.")
    p.add_argument("--wandb", action="store_true", help="Enable Weights & Biases monitoring.")
    p.add_argument("--wandb-project", default=Config.wandb_project)
    p.add_argument("--wandb-entity", default=Config.wandb_entity)
    p.add_argument("--wandb-run-name", default=Config.wandb_run_name)
    p.add_argument("--log-every", type=int, default=Config.log_every)
    p.add_argument("--checkpoint-dir", default=None, help="Fresh directory; defaults to OUTPUT/checkpoints.")
    p.add_argument("--checkpoint-every", type=int, default=Config.checkpoint_every)
    p.add_argument("--resume", default=Config.resume_checkpoint,
                   help="Path to a checkpoint_slab_*.pkl or latest.pkl file.")
    p.add_argument("--temporal-integrator", choices=("rk4", "heun"), default="rk4",
                   help="Integrator in the auxiliary training penalty; direct predictions are unchanged.")
    p.add_argument("--defect-normalization", choices=("raw", "rate"), default=None,
                   help="Alias for --[no-]normalize-rk4-defect; conflicting explicit options are rejected.")
    p.add_argument("--diagnostic-every", type=int, default=500,
                   help="Fixed-weight loss-gradient diagnostic interval; includes initial/final block states.")
    p.add_argument("--gradient-clip-norm", type=float, default=0.0,
                   help="Global gradient clipping threshold; 0 disables clipping (historical default).")
    p.set_defaults(early_stopping=False, adaptive_weights=False, forecast_strategy="direct")
    a = p.parse_args()
    if a.defect_normalization is not None:
        flag = a.defect_normalization == "rate"
        explicit = {"--normalize-rk4-defect", "--no-normalize-rk4-defect"}.intersection(sys.argv[1:])
        if explicit and a.normalize_rk4_defect != flag:
            p.error("conflicting defect normalisation options")
        a.normalize_rk4_defect = flag
    if a.mode != "forward" or a.config != "forward":
        p.error("the controlled study supports forward training only")
    if a.early_stopping:
        p.error("controlled paired training requires fixed updates and no early stopping")
    if a.forecast_strategy != "direct":
        p.error("controlled training saves direct predictions; evaluate common continuation separately")
    if a.checkpoint_every != 1:
        p.error("controlled training requires --checkpoint-every 1")
    if a.diagnostic_every <= 0 or not np.isfinite(a.gradient_clip_norm) or a.gradient_clip_norm < 0:
        p.error("--diagnostic-every must be positive and --gradient-clip-norm finite/nonnegative")
    if a.method != "fd_rk4" and a.rk4_weight != 0:
        p.error("use --rk4-weight 0 with --method fd or vanilla")
    if a.data_weight != 0:
        p.error("forward initial-only study requires --data-weight 0")
    if a.wandb:
        p.error("controlled source snapshot disables external logging")
    a.checkpoint_dir = a.checkpoint_dir or str(Path(a.output) / "checkpoints")
    if min(a.alpha, a.beta, a.nu, a.initial_alpha, a.initial_beta, a.initial_nu) <= 0:
        p.error("all coefficients and initial parameter guesses must be positive")
    if min(a.epochs, a.slabs, a.spatial_points, a.pairs_per_step, a.observations) <= 0:
        p.error("--epochs, --slabs, --spatial-points, --pairs-per-step, and --observations must be positive")
    requested_weights = (a.residual_weight, a.rk4_weight, a.initial_weight,
                         a.boundary_weight, a.previous_weight, a.data_weight)
    if not all(np.isfinite(value) and value >= 0.0 for value in requested_weights):
        p.error("all six loss weights must be finite and non-negative")
    if a.initial_weight == 0.0:
        p.error("--initial-weight must be positive so every slab has an interface anchor")
    if a.mode == "inverse" and not a.infer:
        p.error("inverse mode needs at least one coefficient in --infer")
    if a.checkpoint_every <= 0:
        p.error("--checkpoint-every must be positive")
    if a.rk4_substeps <= 0:
        p.error("--rk4-substeps must be positive")
    if not 0.0 <= a.train_end_time < a.forecast_end_time:
        p.error("--train-end-time must be non-negative and smaller than --forecast-end-time")
    if a.early_stopping_warmup < 0 or a.early_stopping_patience <= 0 or a.early_stopping_min_delta < 0:
        p.error("early-stopping warmup/min-delta must be non-negative and patience must be positive")
    if not 0.0 < a.residual_gate_init < 1.0:
        p.error("--residual-gate-init must lie strictly between 0 and 1")
    if a.adaptive_update_every <= 0:
        p.error("--adaptive-update-every must be positive")
    if min(a.adaptive_first_slab_warmup, a.adaptive_later_slab_warmup) < 0:
        p.error("adaptive warmup epochs must be non-negative")
    if not 0.0 <= a.adaptive_ema_decay < 1.0:
        p.error("--adaptive-ema-decay must lie in [0, 1)")
    if not np.isfinite(a.adaptive_rate) or not 0.0 < a.adaptive_rate <= 1.0:
        p.error("--adaptive-rate must lie in (0, 1]")
    if not np.isfinite(a.adaptive_progress_power) or a.adaptive_progress_power < 0.0:
        p.error("--adaptive-progress-power must be finite and non-negative")
    if not (np.isfinite(a.adaptive_min_multiplier)
            and np.isfinite(a.adaptive_max_multiplier)
            and 0.0 < a.adaptive_min_multiplier <= 1.0 <= a.adaptive_max_multiplier):
        p.error("adaptive multipliers must satisfy 0 < min <= 1 <= max")
    if not np.isfinite(a.adaptive_gradient_epsilon) or a.adaptive_gradient_epsilon <= 0.0:
        p.error("--adaptive-gradient-epsilon must be finite and positive")
    if (not np.isfinite(a.adaptive_max_temporal_gradient_ratio)
            or a.adaptive_max_temporal_gradient_ratio <= 0.0):
        p.error("--adaptive-max-temporal-gradient-ratio must be finite and positive")
    if (a.layers is None) != (a.neurons is None):
        p.error("use --layers and --neurons together")
    if a.layers is not None:
        if a.layers <= 0 or a.neurons <= 0:
            p.error("--layers and --neurons must be positive")
        hidden = (a.neurons,) * a.layers
    elif a.hidden is not None:
        if any(width <= 0 for width in a.hidden):
            p.error("all --hidden widths must be positive")
        hidden = tuple(a.hidden)
    else:
        hidden = tuple(Config.hidden)
    return Config(data_path=a.data, output_dir=a.output, seed=a.seed,
                  temporal_integrator=a.temporal_integrator, diagnostic_every=a.diagnostic_every,
                  gradient_clip_norm=a.gradient_clip_norm,
                  epochs=a.epochs, slabs=a.slabs, pairs_per_step=a.pairs_per_step,
                  train_end_time=a.train_end_time, forecast_end_time=a.forecast_end_time,
                  forecast_strategy=a.forecast_strategy,
                  early_stopping=a.early_stopping, early_stopping_warmup=a.early_stopping_warmup,
                  early_stopping_patience=a.early_stopping_patience,
                  early_stopping_min_delta=a.early_stopping_min_delta,
                  alpha=a.alpha, beta=a.beta, nu=a.nu,
                  residual_weight=a.residual_weight, rk4_weight=a.rk4_weight,
                  initial_weight=a.initial_weight,
                  boundary_weight=a.boundary_weight, previous_weight=a.previous_weight,
                  spatial_points=a.spatial_points, rk4_substeps=a.rk4_substeps,
                  architecture=a.architecture, activation=a.activation, method=a.method,
                  normalize_inputs=a.input_normalization, gate_activation=a.gate_activation,
                  use_residual_gates=a.residual_gates, residual_gate_init=a.residual_gate_init,
                  weight_factorization=a.weight_factorization,
                  lr_decay_spans_run=a.lr_decay_spans_run,
                  hard_initial_condition=a.hard_initial_condition,
                  hard_initial_transition=a.hard_initial_transition,
                  hidden=hidden, mode=a.mode, observations=a.observations,
                  observation_path=a.observation_file,
                  initial_alpha=a.initial_alpha, initial_beta=a.initial_beta,
                  initial_nu=a.initial_nu, data_weight=a.data_weight,
                  normalize_rk4_defect=a.normalize_rk4_defect,
                  deduplicate_replay=a.deduplicate_replay,
                  adaptive_weights=a.adaptive_weights,
                  adaptive_update_every=a.adaptive_update_every,
                  adaptive_first_slab_warmup=a.adaptive_first_slab_warmup,
                  adaptive_later_slab_warmup=a.adaptive_later_slab_warmup,
                  adaptive_ema_decay=a.adaptive_ema_decay,
                  adaptive_rate=a.adaptive_rate,
                  adaptive_progress_power=a.adaptive_progress_power,
                  adaptive_min_multiplier=a.adaptive_min_multiplier,
                  adaptive_max_multiplier=a.adaptive_max_multiplier,
                  adaptive_gradient_epsilon=a.adaptive_gradient_epsilon,
                  adaptive_max_temporal_gradient_ratio=a.adaptive_max_temporal_gradient_ratio,
                  rk4_previous_controller=a.rk4_previous_controller,
                  inverse_loss_normalize=a.inverse_loss_normalize,
                  inverse_loss_ema_decay=a.inverse_loss_ema_decay,
                  inverse_loss_power=a.inverse_loss_power,
                  inverse_loss_min=a.inverse_loss_min, inverse_loss_max=a.inverse_loss_max,
                  inverse_loss_warmup_epochs=a.inverse_loss_warmup_epochs,
                  inverse_loss_update_every=a.inverse_loss_update_every,
                  inverse_loss_epsilon=a.inverse_loss_epsilon,
                  infer_parameters=tuple(a.infer),
                  use_wandb=a.wandb, wandb_project=a.wandb_project,
                  wandb_entity=a.wandb_entity, wandb_run_name=a.wandb_run_name, log_every=a.log_every,
                  checkpoint_dir=a.checkpoint_dir, checkpoint_every=a.checkpoint_every, resume_checkpoint=a.resume)


if __name__ == "__main__":
    config = arguments()
    from controlled_runner import run
    run(config, sys.modules[__name__])
