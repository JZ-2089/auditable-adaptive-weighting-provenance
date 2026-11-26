"""Defaults for the adaptive-weight forward Allen--Cahn experiment."""

import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.mode = "forward"
    config.seed = 42
    config.input_dim = 2

    config.data = data = ml_collections.ConfigDict()
    data.path = "../data/allen_cahn-3-600.mat"
    data.output_dir = "invloss_forward_output"
    data.observation_path = None

    config.physics = physics = ml_collections.ConfigDict()
    physics.method = "fd_rk4"
    physics.alpha = 5.0
    physics.beta = 5.0
    physics.nu = 1e-4

    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "modified"
    arch.num_layers = 4
    arch.hidden_dim = 128
    arch.hidden_dims = None
    arch.out_dim = 1
    arch.activation = "tanh"
    arch.normalize_inputs = True
    # This solver had no hard initial condition, unlike the reaction-diffusion,
    # Burgers and cylinder solvers; without it training collapses onto the
    # residual's stable equilibrium. Added 2026-09-10.
    arch.hard_initial_condition = True
    arch.hard_initial_transition = 0.2
    arch.gate_activation = "sigmoid"
    arch.use_residual_gates = True
    arch.residual_gate_init = 0.1
    arch.reparam = ml_collections.ConfigDict(
        {"type": "weight_fact", "mean": 0.5, "stddev": 0.1}
    )

    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = "Adam"
    optim.beta1 = 0.9
    optim.beta2 = 0.999
    optim.eps = 1e-8
    optim.learning_rate = 5e-4
    optim.parameter_learning_rate = 5e-3
    optim.decay_rate = 0.5
    optim.decay_steps = 5_000

    config.training = training = ml_collections.ConfigDict()
    training.max_epochs = 5_000
    # Four slabs cover only the observed interval [0, 1].  The held-out
    # interval (1, 3] is never used by a training loss.
    training.time_slabs = 4
    training.rk_pairs_per_step = 8
    training.spatial_points = 128
    training.rk4_substeps = 1
    training.early_stopping = early = ml_collections.ConfigDict()
    early.enabled = True
    early.warmup_epochs = 1_000
    early.patience = 750
    early.min_delta = 1e-5
    early.ema_decay = 0.9

    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.residual = 0.97
    # Kept as a fixed anchor by the adaptive controller, but exposed so all
    # terms in the objective have an explicit, reproducible coefficient.
    # Raised from 1.0: with a unit coefficient the optimiser can satisfy the
    # residual by collapsing onto u = -1, an exact solution, ignoring the
    # initial state. See AC_TORUS_FIX.md.
    weighting.initial = 100.0
    weighting.boundary = 1.0
    weighting.previous = 1.0
    # A version-8 three-seed, held-out validation sweep selected zero RK4
    # weight for this forward data/model/budget.  Version 9 uses cleaner replay
    # semantics, so treat that result as conservative evidence rather than a
    # fresh validation.  For a fixed total physics budget, compare this default
    # with --residual-weight 0.97 --rk4-weight 0.03.
    weighting.rk4 = 0.03
    weighting.data = 0.0
    # Keep the current slab interface in L_initial only.  Disable this solely
    # when reproducing checkpoints from the original solver.
    weighting.deduplicate_replay = True
    # A one-step RK4 defect is O(dt) relative to the differential residual.
    # Dividing it by dt makes the two physics losses comparable before the
    # controller balances their network-gradient influence.
    weighting.normalize_rk4_defect = True
    weighting.adaptive = adaptive = ml_collections.ConfigDict()
    # The grouped controller is opt-in: the historical fixed setting above had
    # the best validation median, while version-8 adaptive + RK4=0.03
    # approximately matched its fixed counterpart on one confirmation seed.
    adaptive.enabled = False
    adaptive.update_every = 50
    adaptive.first_slab_warmup_epochs = 500
    adaptive.later_slab_warmup_epochs = 100
    adaptive.ema_decay = 0.9
    adaptive.rate = 0.1
    adaptive.progress_power = 0.25
    adaptive.min_multiplier = 0.01
    adaptive.max_multiplier = 100.0
    adaptive.gradient_epsilon = 1e-12

    # Which controller sets the rk4/previous coefficients (needs adaptive.enabled):
    #   "grouped" -- this solver's native grouped/priority-aware GradNorm.
    #   "inverse_loss" -- rk4/previous get their own inverse-loss coefficient
    #     c_i = clip((L_ref_i / EMA[l_i])**power, min, max) via the shared
    #     adaptive_weight_controllers.py, pulled out of the grouped budget.
    weighting.rk4_previous_controller = "grouped"
    weighting.inverse_loss = inverse_loss = ml_collections.ConfigDict()
    inverse_loss.enabled = True
    inverse_loss.ema_decay = 0.99
    inverse_loss.power = 1.0
    inverse_loss.min_coefficient = 1.0e-2
    inverse_loss.max_coefficient = 1.0e2
    inverse_loss.warmup_epochs = 200
    inverse_loss.update_every = 1
    inverse_loss.epsilon = 1.0e-12

    config.forecast = forecast = ml_collections.ConfigDict()
    forecast.train_end_time = 1.0
    forecast.end_time = 3.0
    # ``rk4_rollout`` advances the state with the learned/fixed Allen--Cahn
    # physics after t=1. ``direct`` evaluates the frozen neural network OOD.
    forecast.strategy = "rk4_rollout"

    # Kept for a common schema; these values are inactive in forward mode.
    config.inverse = inverse = ml_collections.ConfigDict()
    inverse.parameters = ("alpha", "beta", "nu")
    inverse.observations_per_slab = 500
    inverse.initial_alpha = 1.0
    inverse.initial_beta = 1.0
    inverse.initial_nu = 1e-3

    config.logging = logging = ml_collections.ConfigDict()
    logging.use_wandb = False
    logging.wandb_project = "allen-cahn-fd-rk4-pinn"
    logging.wandb_entity = None
    logging.wandb_run_name = "invloss-forward-0-1-to-3"
    logging.log_every_steps = 100

    config.saving = saving = ml_collections.ConfigDict()
    saving.checkpoint_dir = "checkpoints/invloss_forward"
    saving.checkpoint_every_slabs = 1
    saving.resume_checkpoint = None
    return config
