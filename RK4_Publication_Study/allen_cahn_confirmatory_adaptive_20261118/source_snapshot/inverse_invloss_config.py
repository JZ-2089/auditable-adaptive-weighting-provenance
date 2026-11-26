"""Conservative copied defaults for inverse Allen--Cahn estimation."""

import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.mode = "inverse"
    config.seed = 42
    config.input_dim = 2

    config.data = data = ml_collections.ConfigDict()
    data.path = "../data/allen_cahn-3-600.mat"
    data.output_dir = "inverse_copy_output"
    data.observation_path = None  # optional .npz: t_obs, x_obs, u_obs

    # For inferred coefficients these are comparison references. Coefficients
    # omitted from inverse.parameters remain fixed at these values.
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
    optim.decay_steps = 10_000

    config.training = training = ml_collections.ConfigDict()
    training.max_epochs = 10_000
    # Estimate the coefficients from observations no later than t=1.
    training.time_slabs = 4
    training.rk_pairs_per_step = 8
    training.spatial_points = 128
    training.rk4_substeps = 1
    training.early_stopping = early = ml_collections.ConfigDict()
    early.enabled = True
    early.warmup_epochs = 3_000
    early.patience = 2_000
    early.min_delta = 1e-5
    early.ema_decay = 0.9

    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.residual = 1.0
    # Kept as a fixed anchor by the adaptive controller, but exposed so all
    # terms in the objective have an explicit, reproducible coefficient.
    weighting.initial = 1.0
    weighting.boundary = 1.0
    weighting.previous = 1.0
    weighting.rk4 = 0.1
    weighting.data = 10.0
    # Keep the current slab interface in L_initial only.  Disable this solely
    # when reproducing checkpoints from the original solver.
    weighting.deduplicate_replay = True
    # No full inverse sweep was performed, so retain the original raw-defect
    # semantics and static coefficients by default.  Normalised/grouped
    # adaptation remains available through the corresponding CLI flags.
    weighting.normalize_rk4_defect = False
    weighting.adaptive = adaptive = ml_collections.ConfigDict()
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

    config.forecast = forecast = ml_collections.ConfigDict()
    forecast.train_end_time = 1.0
    forecast.end_time = 3.0
    forecast.strategy = "rk4_rollout"

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
    logging.wandb_run_name = "inverse"
    logging.log_every_steps = 100

    config.saving = saving = ml_collections.ConfigDict()
    saving.checkpoint_dir = "checkpoints/inverse_copy"
    saving.checkpoint_every_slabs = 1
    saving.resume_checkpoint = None
    return config
