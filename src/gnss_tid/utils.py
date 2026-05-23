from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

PathLike = str | Path
PathInput = PathLike | Iterable[PathLike]


def normalize_paths(
    paths: PathInput,
    *,
    expand_user: bool = True,
    resolve: bool = True,
) -> list[Path]:
    """
    Normalize a flexible path input into a list of pathlib.Path objects.

    Accepts:
      - single string path
      - single pathlib.Path
      - iterable of strings and/or pathlib.Path
    Any entry can be a glob pattern (e.g. 'src/**/*.py').

    Parameters
    ----------
    paths : PathInput
        A path or collection of paths (str/Path).
    expand_user : bool, default True
        Whether to expand '~' to the user home.
    resolve : bool, default False
        Whether to resolve paths to absolute (following symlinks).

    Returns
    -------
    list[pathlib.Path]
        List of Path objects with glob patterns expanded.
    """

    def _to_iter(x: PathInput) -> Iterable[PathLike]:
        # If it's a single string or Path, wrap in a list
        if isinstance(x, (str, Path)):
            return [x]
        # Else assume it's already an iterable of paths
        return x

    def _is_glob_pattern(s: str) -> bool:
        # Basic heuristic: contains any glob metacharacters
        return any(ch in s for ch in "*?[]")

    def _process_single(p: PathLike) -> list[Path]:
        # Convert to Path and (optionally) expand ~
        p = Path(p)
        if expand_user:
            p = p.expanduser()

        s = str(p)

        if _is_glob_pattern(s):
            # Handle glob pattern:
            # Use Path.glob starting from the appropriate root
            # Example: "src/**/*.py" -> base = "src", pattern = "**/*.py"
            # Example: "**/*.py" -> base = ".", pattern = "**/*.py"
            parts = p.parts

            # Find first part that contains a glob char
            idx = None
            for i, part in enumerate(parts):
                if _is_glob_pattern(part):
                    idx = i
                    break

            if idx is None:
                # No glob, just treat as normal path
                return [_finalize_path(p)]

            base = Path(*parts[:idx]) if idx > 0 else Path(".")
            pattern = Path(*parts[idx:])

            matches = list(base.glob(str(pattern)))
            return [_finalize_path(m) for m in matches]

        # Not a glob pattern, just a normal path
        return [_finalize_path(p)]

    def _finalize_path(p: Path) -> Path:
        if resolve:
            try:
                return p.resolve()
            except FileNotFoundError:
                # `resolve()` with strict=False in 3.11+, but for portability:
                return p.absolute()
        return p

    normalized: list[Path] = []
    for item in _to_iter(paths):
        normalized.extend(_process_single(item))

    return normalized
