# Auditable adaptive weighting provenance

This repository accompanies *Auditable provenance records for adaptive loss
weighting in sequential physics-informed learning: an Allen--Cahn case study*.
It contains the frozen code, reference inputs, 57 raw prediction arrays, 30
continuation-field arrays, compact records for all 57 runs, generated evidence,
and the claim-to-run provenance package used by the manuscript.

## What is included

- `RK4_Publication_Study/`: the two frozen Allen--Cahn campaigns and the 91
  hash-bound arrays referenced by the audit record;
- `paper/reproducibility/latest_allen_cahn_evidence/`: registered inputs and
  generated manuscript tables/figures;
- `paper/reproducibility/provenance/`: the schema, audit record, W3C PROV
  projection, RO-Crate 1.3 metadata, claim--evidence matrix and validators;
- `environment.yml` and pinned CPU/GPU requirement files.

Machine-specific paths and hostnames in historical JSON records were replaced
deterministically. Scientific arrays, scalar results, seeds, configurations,
timestamps, update counts and numerical values were not changed. The public
provenance package was regenerated after redaction, so its hashes describe the
files in this repository.

## Validate the released record

The provenance validator itself uses only the Python standard library:

```bash
python paper/reproducibility/provenance/validate_provenance.py --verify-external
python paper/reproducibility/provenance/query_claim_evidence.py --claim C1
```

Regenerate the Allen--Cahn tables and figures:

```bash
python -m pip install -r requirements-validation.txt
python paper/reproducibility/latest_allen_cahn_evidence/generate_and_audit.py
```

## Training environment

For CPU validation use `requirements-cpu.txt`. For CUDA 12 training use
`requirements-gpu-cu12.txt`; the recorded runs used Python 3.13.1 and JAX
0.4.38. Exact GPU retraining is expensive and is not run in continuous
integration. The source snapshot and protocol files preserve the full command
configuration for the confirmatory and long-sequence campaigns.

## Data scope

The repository contains every array referenced by the formal audit record.
Redundant checkpoint copies, scheduler logs and intermediate diagnostics not
used by a manuscript claim are excluded. See `DATA_MANIFEST.csv` and
`SHA256SUMS.txt` for paths, sizes and checksums.

## Citation and licence

Citation metadata are provided in `CITATION.cff`. The reuse licence is awaiting
author approval; see `LICENSE_STATUS.md` before reusing the material.

Repository URL: https://github.com/JZ-2089/auditable-adaptive-weighting-provenance
