# Cleverest++ (audit prototype)

This is a benchmark-agnostic hardening/prototyping layer created during the independent audit of **Cleverest**. It is not represented as the authors' code or as a prospectively validated paper result.

Implemented components:

- strict JSON/base64 candidate output;
- argv arrays and executable allowlists, with **no shell**;
- cheap JSON/XML/PDF/text format validation;
- coverage-fingerprint diversity selection;
- separated stdout/stderr capture;
- sanitizer signatures and optional expected-stack identity;
- ambiguous classification whenever both program versions crash.

These directly address the tested command-injection POC and the output-spoof/crash-identity weaknesses in the original artifact. The larger proposed pipeline—retrieval limited to information available at commit time, multi-candidate generation, validity quotas, coverage-delta feedback, minimization, and budgeted short fuzzing—is specified in `../tmp/improvement_design.md`.

## Test

```bash
uv run --project cleverest_plus --with pytest --with ruff pytest
uv run --project cleverest_plus --with ruff ruff check cleverest_plus tests
```

## Evaluation integrity

The released artifact allows an exact *retrospective* replay. Existing ablations have a union of 15/36 BIC bugs versus the 7/36 default (more than 2x), but only 24/36 BFC bugs versus 13/36 (below the 26 needed for 2x). This uses extra configurations selected after seeing benchmark outcomes, so it is an upper bound—not an honest prospective Cleverest++ score. No doubled score is claimed without a preregistered fresh run.
