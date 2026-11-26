# Small audit comparison

This auxiliary experiment checks which defects are detected by three explicitly configured validation layers. It does not compare the full capabilities of PROV, RO-Crate, MLflow or other systems. The cases were specified in `audit_comparison_spec.json` before their first execution; they are developer-designed verification cases, not independently sampled errors or a preregistered scientific study.

Run from any working directory:

```bash
PYTHONDONTWRITEBYTECODE=1 python /path/to/provenance/audit_comparison.py --write
PYTHONDONTWRITEBYTECODE=1 python /path/to/provenance/audit_comparison.py --check
```

The runner requires Python 3.10 or later, the standard library, `profile_validation.py`, the schema-checking component in `validate_provenance.py` and the adjacent schema. The whole provenance directory must be present. No neural-network training, accelerator, external service or third-party Python library is required. The complete audit record and its packaged artifacts are read without modification. External prediction arrays are outside this small experiment; use the full validator with `--verify-external` to verify those separately.

The generated files are:

- `audit_comparison_results.json`: every case, observed errors, aggregate counts and the SHA-256 digests of the specification, runner, both checker modules, schema and source record.
- `audit_comparison_table.tex`: a compact main-text table generated directly from the JSON results.

The three layers share the same file inventory and SHA-256 checks. The second additionally checks identifier links; the third uses the reusable application-profile checker. The fixed Allen--Cahn campaign counts and observed numerical outcomes are not used to detect the injected defects.

Each case operates on a separate copy. When a metadata field is changed, the checksum of the exported audit record is updated before any layer sees it. This models an internally hashed export containing a metadata mistake, rather than giving every layer a trivial stale-checksum signal. In the byte-integrity case, bytes of one packaged metric artifact change while the expected inventory remains fixed. Appended JSON whitespace leaves the represented numerical values unchanged but still constitutes an integrity mismatch.

The two valid controls are the original record and a copy with the top-level entity lists reversed. The six structural/integrity faults remove a prediction mechanism, move an evaluation outside its prediction interval, break a checkpoint reference, remove the required limiting-evidence field, promote a secondary claim to primary without changing its linked metrics, or alter result-file bytes. Removing a required disclosure field tests schema completeness; it does not test the discovery of limitations that the researcher never recorded.

A separate blind-spot probe replaces one claim's prose with a false universal guarantee while preserving the typed relationships and numerical evidence. Its result must be reported even if every layer accepts it: consistency and integrity checks do not establish the truth of prose or the validity of a scientific argument. This probe is not included in the six-fault detection denominator.

Every listed case is reported. The proportions are finite test-suite counts, not estimates of real-world sensitivity, specificity or error prevalence. Passing the two controls offers a narrow regression check; it does not establish a general false-positive rate, usability benefit, cross-study portability or external validation.

The auxiliary report fingerprints the input record and should not be inserted into that same record's artifact inventory, because this would create a circular hash dependency. It is distributed alongside the full-package validation report instead. Rebuild the comparison after changing the source record, schema or checker, and retain the updated report with the corresponding release.
