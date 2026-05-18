# AGENTS.md

## Purpose
This repository is a Python toolbox for processing GNSS TEC observations to study traveling ionospheric disturbances (TIDs).

Primary workflow:
1. Estimate optimal ionospheric pierce point (IPP) projection height (`autofocus`).
2. Estimate TID wave parameters (for example wavelength, period, phase speed).

Input observations are sparse line-of-sight (LOS) point measurements (receiver-satellite TEC geometry). Current implementations often interpolate points to images, but this is an implementation strategy, not a required long-term assumption.

## Stack and Tooling
- Python `>=3.11`
- Dependency and environment management: `uv`
- Config/runtime orchestration: `hydra`
- Core numerics/data: `numpy`, `xarray`, `scipy`, `dask`
- Tests: `pytest`

## Fast Start Commands
- Install deps: `uv sync`
- Run tests: `uv run pytest`
- Run autofocus CLI: `uv run autofocus`
- Run parameter estimation CLI: `uv run param`

## Repo Layout
- `src/gnss_tid/`: installable package and CLIs
- `src/gnss_tid/conf/`: Hydra configs (`focus/`, `event/`, `client/`)
- `tests/`: algorithm and primitive regression tests
- `notebooks/`: exploratory analysis (not authoritative for behavior)
- Top-level scripts (`plot_save_all.py`, `batch_run_save_all.py`): utility workflows

## Pipeline Notes
- `autofocus` (`gnss_tid.cli.autofocus`) reads point data, computes focused products, writes NetCDF (`autofocus.h5` by default).
- `param` (`gnss_tid.cli.parameters`) opens focused output, runs Dask-enabled parameter estimation, writes Zarr.
- Hydra is configured with `hydra.job.chdir: true`; runs may execute in per-run output directories. Keep path handling explicit.
- Current default path is mostly point-data -> gridded image products -> spectral/parameter inference.
- Agents should keep interfaces flexible for future point-based methods that may bypass image interpolation.

## Data and Config Assumptions
- Some event configs reference machine-specific absolute paths (for example under `/disk1/...`).
- When adding configs, prefer portable path conventions or document host-specific assumptions in the config file.
- Keep scientific defaults explicit in YAML and avoid burying magic numbers in Python.

## Engineering Expectations for Agents
- Prefer reusable building blocks over one-off scripts:
  - Factor core logic into small, composable functions/classes with clear inputs/outputs.
  - Keep orchestration (CLI/Hydra/workflow glue) thin and separate from scientific logic.
  - Reuse existing primitives before adding new parallel implementations.
- Make minimal, surgical edits aligned with existing style.
- Preserve public interfaces unless the task explicitly asks for API changes.
- Avoid introducing new code that hard-codes image-only assumptions into shared abstractions.
- For scientific algorithm changes:
  - Add or update tests in `tests/` to cover behavior, not just execution.
  - Prefer deterministic fixtures/synthetic signals similar to existing tests.
  - Validate both numerical sanity (`finite`, dimensions) and scientific plausibility (e.g., peak frequency/wavelength consistency).
- For performance-sensitive code paths, avoid eager `.compute()` unless necessary.

## Testing Policy
- Minimum before handoff: `uv run pytest` for touched behavior.
- If full suite is too expensive, run focused tests first and state what was skipped.
- New features should include at least one regression-style test.
- Run `uv run ruff check gnss_tid` before handoff for code changes in `gnss_tid`. Ignore `notebooks/` and `scripts/` for that check.

## Common Guardrails
- Do not commit secrets, absolute local credentials, or host-only paths unless explicitly required.
- Be careful with coordinate transforms and units; document assumptions near the code that uses them.
- Keep output formats stable (`NetCDF` for autofocus stage, `Zarr` for parameter stage) unless migration is requested.
