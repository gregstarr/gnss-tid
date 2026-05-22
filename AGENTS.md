## Rules
- use uv for python management and running `uv add ...`, `uv run ...`
- use pytest for testing `uv run pytest ...`
- lint after changes `uv run ruff check --fix ...`
- format after changes `uv run ruff format ...`
- functions should have a complete docstring including inputs and outputs
- functions that have xarray dataarrays or datasets as inputs or outputs should document the expected variables, dims and coords in their docstrings
