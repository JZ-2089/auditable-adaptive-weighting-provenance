# Confirmatory Allen--Cahn RK4 weighting study

## Frozen question

On Allen--Cahn, does a rate-normalised RK4 temporal-consistency loss reduce
held-out field error relative to an otherwise identical zero-RK4 sequential
finite-difference PINN, and does the previously selected grouped adaptive
controller improve the result?

The grouped controller was selected using only the three development seeds
4103, 4127 and 4153.  The safeguard variant was rejected before this campaign.
None of the ten seeds below appeared in that pilot or in the earlier
confirmation campaign.

## Arms and matched design

The ten fresh paired seeds are 5101, 5113, 5147, 5167, 5191, 5227, 5261,
5281, 5309 and 5333.  Every seed is run under three arms:

* `zero_rk4`: RK4 coefficient zero and no adaptive controller;
* `fixed_rate1`: rate-normalised RK4 coefficient one, held fixed;
* `grouped_rate1`: the same initial coefficient with the selected grouped,
  budget-preserving adaptive controller.

All arms use four sequential blocks, 5,000 updates per block, 256 periodic
spatial nodes, eight temporal pairs per update, four RK4 substeps, a
4-by-128 modified MLP, hard initial-condition enforcement and identical
seed-specific initialisation and pair schedules.  There is no early stopping
and no gradient clipping.  Arm order is deterministically randomised within
each seed.  Training uses only the initial condition and PDE; reference fields
are not available to the optimiser.

## Evaluation and inference

The training interval is [0,0.8], validation is (0.8,1.0], and the locked test
interval is (1.0,1.2].  Direct final-network relative L2 error over the test
interval is the primary outcome.  A common DOP853 continuation from the
learned state at t=0.8 is a secondary outcome.  Training retention, terminal
error, interface error, wall time, non-finite updates and adaptive-weight
trajectories are diagnostics.

The primary contrast is `grouped_rate1 - zero_rk4`; the second predeclared
contrast is `fixed_rate1 - zero_rk4`.  Negative differences favour RK4.
Report all ten paired runs, mean and median errors, the paired mean difference,
its two-sided 95% t confidence interval, a deterministic paired bootstrap 95%
interval, win count, and a one-sided exact sign-flip randomisation p-value.
Apply Holm correction across the two predeclared direct-test comparisons.
The grouped-versus-fixed contrast and continuation results are secondary and
are reported without converting a null result into a positive claim.

No run may be replaced or omitted because of its error.  A run is repeated
only for a recorded infrastructure failure, using the same arm and seed.
All 30 training runs must finish before the test reference is evaluated.

