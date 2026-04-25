# Experiment Track

## Purpose

This file is the human-readable research tracker for the project.

Use it to record:

- what has actually been run
- whether the validated estimator passed
- what still needs attention before paper writing

This should be updated after meaningful benchmark runs.

## Current Supported Pipeline

Command:

```bash
python experiments/run.py
```

Single-benchmark debugging:

```bash
python experiments/run.py --benchmark poisson
python experiments/run.py --benchmark variable_coefficient
python experiments/run.py --benchmark l_shaped
```

## Acceptance Standard

For each supported benchmark, record:

- true error
- estimated error
- residual contribution
- boundary lifting contribution
- effectivity `η`
- whether the run is stable enough to keep

Target interpretation:

- `η < 1`: reject, estimator not reliable
- `1 <= η <= 2`: acceptable and strong
- `η > 2`: warning, likely too loose for a central result

## Run Log Template

### Poisson

- Status: pending
- Notes:

### Variable Coefficient

- Status: pending
- Notes:

### L-Shaped

- Status: pending
- Notes:

## Legacy Context

The archived log in `exp_log.md` showed why this tracker is needed:

- old runs mixed exploratory and validated claims
- unsupported benchmarks polluted the main narrative
- legacy effectivities below `1` made the repository hard to trust

This tracker exists so future runs stay aligned with the refactored validated scope.
