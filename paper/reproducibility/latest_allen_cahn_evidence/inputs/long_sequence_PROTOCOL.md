# Allen--Cahn long-sequence development protocol

This exploratory study asks whether grouped adaptive RK4 improves sequential
training retention as the number of fixed-width temporal blocks grows.  It is
separate from the completed ten-seed confirmation campaign.

The study uses three development seeds (6101, 6131 and 6173), three methods
(zero-RK4, fixed rate-normalised RK4 with coefficient one, and the selected
grouped adaptive controller), and 4, 8 or 12 blocks.  Every block spans 0.2
time units and receives 5,000 updates.  The corresponding training endpoints
are 0.8, 1.6 and 2.4.  Every model is queried for a further 0.8 time units;
errors are reported over horizons 0.2, 0.4 and 0.8.

The main development diagnostic is final-network error over the complete
training interval, followed by mean and maximum block-interface error.  Direct
and common numerical-continuation forecast errors are secondary diagnostics.
The comparison uses paired seeds.  These three-seed results select whether a
new ten-seed long-sequence confirmation is justified; they are not themselves
treated as confirmatory evidence.

No explicit causal residual weighting is used.  Sequential block order,
hard initial enforcement and previous-block replay are shared by all arms.
An explicit causal-weighting ablation, if run, will be kept separate so that
its contribution is not confounded with grouped adaptive weighting.
