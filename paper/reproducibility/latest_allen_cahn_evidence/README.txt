Allen--Cahn evidence bundle for the DIM adaptive-weighting manuscript
=====================================================================

Purpose
-------
This directory is the single generation and audit route for the Allen--Cahn
tables and figures added on 2026-11-24.  It prevents manuscript values from
being copied manually from run directories.

Evidence hierarchy
------------------
1. The ten-seed campaign is an independent locked campaign.  Its
   grouped-versus-fixed controller comparison is a secondary contrast under
   the frozen protocol, because the registered primary and second contrasts
   compare the two RK4 arms with zero-RK4.
2. The 4/8/12-block campaign is a three-seed development study.  It does not
   support confirmatory inference.
3. Fixed-versus-zero results are retained in the copied records as a component
   control, but the companion RK4 paper owns that numerical-method question.

Files
-----
evidence_sources.json registers the repository source and SHA-256 digest of
every copied input.  inputs/ contains byte-for-byte copies.  The generator
refuses to run if a copied input no longer matches its registered digest.
generated/ contains the compact manuscript summary, LaTeX tables, PDF/PNG
figures, and generated_outputs.sha256.

Regeneration
------------
From this directory run:

  python generate_and_audit.py

The script verifies 30/30 ten-seed runs, 27/27 long-sequence runs, all expected
seed-arm cells, equal update budgets in the ten-seed campaign, and the absence
of non-finite or clipped-gradient updates.  It also records the inferential
status of each comparison in generated/dim_evidence_summary.json.

Controller diagnostics
----------------------
The controller-mechanism diagnostic was generated independently from complete
adaptive traces in the frozen campaign and copied here with its own source
hash.  The generator verifies its inferential-status tag and all 3,840 trace
records before copying the source table and figure to generated/.  It is
descriptive and must not be presented as a causal mediation analysis.
