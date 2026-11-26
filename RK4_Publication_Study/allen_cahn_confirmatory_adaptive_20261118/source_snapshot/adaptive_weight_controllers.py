"""Shared, PDE-agnostic adaptive loss-weighting controllers.

Extracted so that the unsteady-cylinder Navier--Stokes, reaction--diffusion,
and Burgers FD-RK4 PINN solvers under ``PINNGenenral/*/GPT/`` compare the
*same* controller code on different physics, rather than three hand-copied
(and potentially drifting) re-implementations.

Two independent controllers are provided:

* ``update_flat_gradnorm_state`` -- a single-budget, bounded GradNorm-style
  controller (one shared multiplicative budget across every active,
  non-anchor loss term). Originally written for
  ``ns_unsteady_cylinder/GPT/GPT-code-invloss.py``.
* ``update_grouped_gradnorm_state`` -- a grouped, priority-aware,
  per-group-budget-preserving GradNorm controller, ported from
  ``Allencahn/GPT/GPT-code-adaptive.py`` (model version 9, policy
  ``grouped_priority_gradnorm_v2``). Each semantic group (e.g. "the terms
  that describe the dynamics" vs "the terms that constrain the solution")
  keeps its own base-weight budget, so a nearly solved constraint term
  cannot steal weight from the physics terms.

And one targeted, cheap mechanism:

* ``update_inverse_loss_state`` / ``apply_inverse_loss_coefficients`` -- keep
  a *specific* term's weighted raw loss approximately constant relative to a
  frozen reference, without inspecting its gradient. Cheaper than either
  GradNorm variant (reuses the forward loss already computed) but has no
  budget conservation of its own.

Every function operates on a raw ``numpy`` vector ordered by the caller's own
``LOSS_NAMES`` tuple, plus a ``cfg`` object (any object exposing the needed
attributes by name -- a frozen dataclass in every current caller). Group
membership and fixed-anchor names are passed in explicitly; nothing here is
hard-coded to one PDE.

Required ``cfg`` attributes (all four callers currently share these field
names):
    adaptive_ema_decay, adaptive_rate, adaptive_progress_power,
    adaptive_min_multiplier, adaptive_max_multiplier, adaptive_gradient_epsilon,
    inverse_loss_ema_decay, inverse_loss_power, inverse_loss_min,
    inverse_loss_max, inverse_loss_epsilon.
"""
from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# Shared state constructors
# --------------------------------------------------------------------------


def initialize_gradnorm_state(base):
    """Host-side state for either GradNorm variant; reset at each time slab."""
    base = np.asarray(base, dtype=np.float64)
    return {
        "weights": base.copy(),
        "log_multipliers": np.zeros_like(base),
        "loss_ema": np.full_like(base, np.nan),
        "gradient_ema": np.full_like(base, np.nan),
        "initial_loss": np.full_like(base, np.nan),
    }


def initialize_inverse_loss_state(count):
    """Host-side EMA/reference/coefficient state for the inverse-loss terms."""
    return {
        "loss_ema": np.full(count, np.nan, dtype=np.float64),
        "reference": np.full(count, np.nan, dtype=np.float64),
        "coefficient": np.ones(count, dtype=np.float64),
    }


# --------------------------------------------------------------------------
# Flat (single-budget) GradNorm -- ns_unsteady_cylinder/GPT/GPT-code-invloss.py
# --------------------------------------------------------------------------


def flat_loss_mask(loss_names, base, fixed_names, extra_fixed=()):
    """Select balanceable terms: base > 0, excluding fixed anchors/priors."""
    active = np.asarray(base, dtype=np.float64) > 0.0
    for name in tuple(fixed_names) + tuple(extra_fixed):
        active[loss_names.index(name)] = False
    return active


def project_flat_weights(base, log_multipliers, active, minimum, maximum):
    """Bound multiplicative weights while preserving their single budget."""
    base = np.asarray(base, dtype=np.float64)
    log_multipliers = np.asarray(log_multipliers, dtype=np.float64)
    active = np.asarray(active, dtype=bool) & (base > 0.0)
    result = base.copy()
    indices = np.flatnonzero(active)
    if indices.size <= 1:
        return result
    lower_log, upper_log = np.log(minimum), np.log(maximum)
    budget = float(np.sum(base[indices]))

    def candidate(shift):
        multiplier = np.exp(np.clip(
            log_multipliers[indices] + shift, lower_log, upper_log
        ))
        return base[indices] * multiplier

    lower_shift, upper_shift = -128.0, 128.0
    for _ in range(80):
        midpoint = 0.5 * (lower_shift + upper_shift)
        if float(np.sum(candidate(midpoint))) < budget:
            lower_shift = midpoint
        else:
            upper_shift = midpoint
    result[indices] = candidate(0.5 * (lower_shift + upper_shift))
    return result


def update_flat_gradnorm_state(state, base, losses, gradient_rms, cfg, active):
    """One bounded, single-budget GradNorm-style controller update."""
    base = np.asarray(base, dtype=np.float64)
    losses = np.asarray(losses, dtype=np.float64)
    gradient_rms = np.asarray(gradient_rms, dtype=np.float64)
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
    if np.count_nonzero(valid) >= 2:
        progress = np.clip(loss_ema[valid] / initial_loss[valid], 1.0e-8, 1.0e8)
        log_progress = np.log(progress)
        log_progress -= np.mean(log_progress)
        influence = np.maximum(
            weights[valid] * gradient_ema[valid], cfg.adaptive_gradient_epsilon
        )
        log_influence = np.log(influence)
        log_target = (np.mean(log_influence)
                      + cfg.adaptive_progress_power * log_progress)
        correction = np.clip(log_target - log_influence, -np.log(2.0), np.log(2.0))
        log_multipliers[valid] += cfg.adaptive_rate * correction
        weights = project_flat_weights(
            base, log_multipliers, active,
            cfg.adaptive_min_multiplier, cfg.adaptive_max_multiplier,
        )
    return {
        "weights": weights,
        "log_multipliers": log_multipliers,
        "loss_ema": loss_ema,
        "gradient_ema": gradient_ema,
        "initial_loss": initial_loss,
    }


# --------------------------------------------------------------------------
# Grouped, priority-aware, budget-preserving GradNorm
# -- ported from Allencahn/GPT/GPT-code-adaptive.py
# --------------------------------------------------------------------------


def project_grouped_weights(base, log_multipliers, active, minimum, maximum, groups):
    """Bound multipliers while preserving EACH group's own base-weight budget."""
    base = np.asarray(base, dtype=np.float64)
    log_multipliers = np.asarray(log_multipliers, dtype=np.float64)
    active = np.asarray(active, dtype=bool) & (base > 0.0)
    result = base.copy()
    lower_log, upper_log = np.log(minimum), np.log(maximum)
    for group in groups:
        indices = np.asarray([index for index in group if active[index]], dtype=np.int64)
        if indices.size <= 1:
            continue
        budget = float(np.sum(base[indices]))

        def candidate(shift, indices=indices):
            multiplier = np.exp(np.clip(
                log_multipliers[indices] + shift, lower_log, upper_log
            ))
            return base[indices] * multiplier

        lower_shift, upper_shift = -128.0, 128.0
        for _ in range(80):
            midpoint = 0.5 * (lower_shift + upper_shift)
            if float(np.sum(candidate(midpoint))) < budget:
                lower_shift = midpoint
            else:
                upper_shift = midpoint
        result[indices] = candidate(0.5 * (lower_shift + upper_shift))
    return result


def update_grouped_gradnorm_state(state, base, losses, gradient_rms, cfg,
                                  loss_names, fixed_names, groups):
    """Grouped, priority-aware, budget-preserving GradNorm controller.

    Ported from ``update_adaptive_state`` in
    ``Allencahn/GPT/GPT-code-adaptive.py``. The target is priority-aware,
    ``w_i * g_i (proportional to) b_i * progress_i**power``, not an
    equal-gradient rule, and a group with any invalid member statistic eases
    back toward its configured base priorities instead of holding a stale
    extreme.
    """
    base = np.asarray(base, dtype=np.float64)
    losses = np.asarray(losses, dtype=np.float64)
    gradient_rms = np.asarray(gradient_rms, dtype=np.float64)
    active = base > 0.0
    for name in fixed_names:
        active[loss_names.index(name)] = False
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
    for group in groups:
        indices = np.asarray([index for index in group if active[index]], dtype=np.int64)
        if indices.size < 2:
            continue
        if not np.all(valid[indices]):
            correction = np.clip(-log_multipliers[indices], -np.log(2.0), np.log(2.0))
            log_multipliers[indices] += cfg.adaptive_rate * correction
            group_mask = np.zeros_like(active)
            group_mask[indices] = True
            projected = project_grouped_weights(
                base, log_multipliers, group_mask,
                cfg.adaptive_min_multiplier, cfg.adaptive_max_multiplier, groups,
            )
            weights[indices] = projected[indices]
            log_multipliers[indices] = np.log(weights[indices] / base[indices])
            continue
        group_mask = np.zeros_like(active)
        group_mask[indices] = True
        progress = np.clip(loss_ema[indices] / initial_loss[indices], 1.0e-8, 1.0e8)
        desired_log_multipliers = log_multipliers.copy()
        desired_log_multipliers[indices] = (
            cfg.adaptive_progress_power * np.log(progress) - np.log(gradient_ema[indices])
        )
        target_weights = project_grouped_weights(
            base, desired_log_multipliers, group_mask,
            cfg.adaptive_min_multiplier, cfg.adaptive_max_multiplier, groups,
        )
        target_log_multipliers = np.log(target_weights[indices] / base[indices])
        correction = np.clip(
            target_log_multipliers - log_multipliers[indices], -np.log(2.0), np.log(2.0)
        )
        log_multipliers[indices] += cfg.adaptive_rate * correction
        projected = project_grouped_weights(
            base, log_multipliers, group_mask,
            cfg.adaptive_min_multiplier, cfg.adaptive_max_multiplier, groups,
        )
        weights[indices] = projected[indices]
        log_multipliers[indices] = np.log(weights[indices] / base[indices])
    log_multipliers[~active] = 0.0

    return {
        "weights": weights,
        "log_multipliers": log_multipliers,
        "loss_ema": loss_ema,
        "gradient_ema": gradient_ema,
        "initial_loss": initial_loss,
    }


# --------------------------------------------------------------------------
# Inverse-loss normalisation -- targeted, cheap, no budget conservation
# --------------------------------------------------------------------------


def update_inverse_loss_state(state, raw_losses, indices, cfg, *, calibrate):
    """Advance the inverse-loss controller by one observation.

    ``raw_losses`` is the full ``LOSS_NAMES``-ordered vector of unweighted
    losses; ``indices`` selects the terms this controller owns (order fixed
    for the lifetime of a run). ``calibrate`` freezes the current EMA as the
    reference the first time it is set, so every coefficient departs from 1.
    """
    picked = np.asarray(raw_losses, dtype=np.float64)[list(indices)]
    loss_ema = state["loss_ema"].copy()
    reference = state["reference"].copy()
    coefficient = state["coefficient"].copy()
    decay = cfg.inverse_loss_ema_decay
    epsilon = cfg.inverse_loss_epsilon
    for i, value in enumerate(picked):
        if not (np.isfinite(value) and value >= 0.0):
            continue
        clipped = max(float(value), epsilon)
        loss_ema[i] = (clipped if not np.isfinite(loss_ema[i])
                       else decay * loss_ema[i] + (1.0 - decay) * clipped)
    if calibrate:
        for i in range(len(indices)):
            if np.isfinite(loss_ema[i]):
                reference[i] = max(loss_ema[i], epsilon)
    for i in range(len(indices)):
        if (np.isfinite(loss_ema[i]) and np.isfinite(reference[i])
                and loss_ema[i] > 0.0):
            ratio = reference[i] / loss_ema[i]
            coefficient[i] = float(np.clip(
                ratio ** cfg.inverse_loss_power,
                cfg.inverse_loss_min, cfg.inverse_loss_max,
            ))
    return {"loss_ema": loss_ema, "reference": reference, "coefficient": coefficient}


def apply_inverse_loss_coefficients(weights, indices, state):
    """Scale the inverse-loss-owned entries of a weight vector (copy-safe)."""
    scaled = np.asarray(weights, dtype=np.float64).copy()
    for slot, index in enumerate(indices):
        scaled[index] *= state["coefficient"][slot]
    return scaled
