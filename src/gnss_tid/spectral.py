import logging
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Literal

import numpy as np
import pandas as pd
import xarray
from joblib import Parallel, delayed
from matplotlib import pyplot as plt

from .center_finding import run_smoothed_center_finder
from .coords import Local2D
from .fft import fft_patches, make_patches, make_wavenum_grid
from .image import ImageMakerBase, generate_image
from .parallel_logging import WorkerChannel, worker_channels
from .plotting import (
    plot_center_finder,
    plot_density_map,
    plot_objective_surfaces,
    plot_peak_patch_spectrum,
    plot_pre_center_finder,
)
from .pointdata import TimeWindow, get_data, make_time_windows

logger = logging.getLogger(__name__)


def _fmt_dt(dt: np.datetime64) -> str:
    """Format np.datetime64 as HH:MM:SS for log messages."""
    s = str(dt)
    return s.split("T")[1][:8] if "T" in s else s


@dataclass(frozen=True, eq=False)
class SpectralConfig:
    """Static inputs shared across spectral-focusing workers.

    Bundling these into one object keeps worker signatures short. The pipeline
    passes it directly to joblib tasks so large numeric and datetime arrays can
    be shared through read-only memmaps. Observation ``rx`` and ``sv`` columns
    are converted to categoricals once per run to avoid repeatedly serializing
    strings. Workers copy observation windows before processing them.

    Fields cannot be reassigned, but nested objects are not immutable. Equality
    and hashing are identity-based because DataFrame value comparison is
    ambiguous.

    ``logscale_objective`` selects the two-pass smoothing reduction used for
    the main sequence: when ``True``, heights come from a rolling mean over
    ``log10(objective)`` (geometric mean); when ``False``, from an arithmetic
    mean of the raw linear-power objective.  Both reductions are always
    computed and both height tracks' image sequences are saved; this flag
    decides which sequence carries the FFT patches and is used for center
    finding.  It has no effect on stored patch powers, which are always
    linear, and no effect on the single-pass branch (where ``argmax`` over
    height is invariant to monotone transforms).

    ``normalization`` controls the amplitude scaling used only to calculate
    patch spectra: ``"image"`` z-scores the complete image, ``"patch"``
    z-scores every extracted patch, and ``None`` preserves the raw image
    amplitudes.  The stored ``image`` is never normalized.

    ``density_thresh`` and ``min_patch_coverage`` implement the patch density
    gate: when ``density_thresh`` is not ``None``, pixels with ``density`` at
    or below the threshold are cut from a local copy of the image inside
    :func:`compute_patch_spectra` before the patch FFT (the stored ``image``
    stays unmasked), each patch's coverage is the fraction of its pixels above
    the threshold, and :func:`extract_patch_peaks` masks ``F``/``Fx``/``Fy``
    wherever coverage falls below ``min_patch_coverage`` so the objective,
    height selection and center-finding seed only see data-supported patches.
    ``density_thresh=None`` (default) applies no cut and
    ``min_patch_coverage=0.0`` (default) disables the gate; both preserve the
    legacy behaviour.
    """

    obs: pd.DataFrame
    rx: pd.DataFrame
    image_maker: ImageMakerBase
    tec_name: str
    block_size: int
    block_step: int
    window: np.ndarray
    logscale_objective: bool
    lat_limits: tuple[float, float]
    lon_limits: tuple[float, float]
    proj: Local2D
    density_thresh: float | None = None
    min_patch_coverage: float = 0.0
    normalization: Literal["image", "patch"] | None = None


def compute_patch_spectra(
    img: xarray.Dataset,
    block_size: int,
    block_step: int,
    window: xarray.DataArray,
    density_thresh: float | None = None,
    normalization: Literal["image", "patch"] | None = None,
) -> xarray.Dataset:
    """Compute FFT patches with optional density cutting and normalization.

    Constructs overlapping blocks via a rolling window, applies the FFT
    windowing function, and returns the linear 2-D power spectrum of each
    block.  The horizontal resolution (km per pixel) is inferred from the
    spacing of ``img.x`` so the resulting ``kx``/``ky`` coords come out in
    cycles per km.

    When ``density_thresh`` is given and ``img`` carries a ``density``
    variable, pixels with ``density <= density_thresh`` are cut (NaN) from a
    local copy of the image — on the already-high-passed image, at patch time —
    and each patch's coverage (the fraction of its pixels above the threshold)
    is returned alongside.  The cut image is zero-filled before the FFT, so
    the input spectra are always finite; the coverage field lets callers gate
    peak powers from patches dominated by data voids.  The stored ``image``
    in ``img`` is never modified.

    ``normalization="image"`` z-scores the full image before blocks are
    extracted.  ``normalization="patch"`` z-scores each extracted block.  In
    both modes normalization applies only to the image copy used for patch
    spectra; ``img.image`` is not modified.  ``None`` preserves raw amplitude.
    Constant blocks — e.g. fully-void patches after the density cut, whose
    standard deviation is exactly zero — are mapped to all-zero spectra
    instead of dividing by zero, so the returned spectra stay finite for the
    ungated ``argmax`` in :func:`extract_patch_peaks`.

    Args:
        img: Image-maker ``Dataset`` with variables ``image`` and ``density``
            on dims ``(y, x)`` and a uniformly spaced ``x`` coordinate (in
            km).
        block_size: Edge length (pixels) of each square FFT block.
        block_step: Stride (in pixels) between consecutive block centres.
        window: FFT windowing array of shape ``(1, 1, block_size, block_size)``.
        density_thresh: Minimum ``density`` (data points per pixel) required
            for a pixel to enter the patch FFT; ``None`` (default) applies no
            cut and skips the coverage computation.
        normalization: ``"image"`` z-scores the full image, ``"patch"``
            z-scores each extracted block, and ``None`` (default) leaves
            amplitudes unchanged.  Constant blocks map to zero spectra.

    Returns:
        A ``Dataset`` with variable ``patch``: 2-D power spectra with dims
        ``(py, px, kx, ky)`` where ``px``, ``py`` are spatial patch centres
        and ``kx``, ``ky`` are wavenumbers (cycles km⁻¹).  When the cut is
        applied, a ``patch_coverage`` variable ``(py, px)`` holds the fraction
        of each patch's pixels with ``density > density_thresh``.
    """
    if normalization not in (None, "image", "patch"):
        raise ValueError(
            "normalization must be one of None, 'image', or 'patch'; "
            f"got {normalization!r}"
        )
    hres = float(np.diff(img.x.values).mean())
    wavenum = make_wavenum_grid(block_size, hres, shift=True)
    image = img.image
    coverage = None
    if density_thresh is not None and "density" in img:
        cut = image.where(img.density > density_thresh)
        coverage = make_patches(
            cut.notnull().astype("float32"), block_size, block_step
        ).mean(("kx", "ky"))
        image = cut.fillna(0)
    if normalization == "image":
        std = image.std(("x", "y"))
        image = xarray.where(
            std > 0, (image - image.mean(("x", "y"))) / std.where(std > 0), 0.0
        )
    patches = make_patches(image, block_size, block_step)
    if normalization == "patch":
        std = patches.std(("kx", "ky"))
        patches = xarray.where(
            std > 0,
            (patches - patches.mean(("kx", "ky"))) / std.where(std > 0),
            0.0,
        )
    F = fft_patches(patches, window, Nfft=block_size, shift=True)
    spectra = (abs(F) ** 2).assign_coords(kx=wavenum, ky=wavenum)
    out = xarray.Dataset({"patch": spectra})
    if coverage is not None:
        out["patch_coverage"] = coverage
    return out


def extract_patch_peaks(
    patch: xarray.DataArray | xarray.Dataset, min_coverage: float = 0.0
) -> xarray.Dataset:
    """Summarise an FFT patch array by its spectral peak.

    Selects the wavenumber ``(kx, ky)`` of the maximum power at each spatial
    location and computes the total objective (sum over ``(px, py)``).  The
    argmax always runs on the *ungated*, finite spectra — never on NaN.

    When a ``patch_coverage`` field is available (Dataset input carrying the
    coverage variable from :func:`compute_patch_spectra`) and
    ``min_coverage > 0``, ``F``, ``Fx`` and ``Fy`` are masked (NaN) wherever
    the patch coverage falls below ``min_coverage`` and the objective is
    recomputed from the gated powers, so patches dominated by data voids are
    excluded from the peak statistics.  A ``DataArray`` input or
    ``min_coverage=0`` reproduces the ungated behaviour.

    Args:
        patch: ``DataArray`` with dims ``(..., px, py, kx, ky)``, or a
            ``Dataset`` holding that array as variable ``patch`` and
            optionally ``patch_coverage`` ``(..., px, py)``.
        min_coverage: Minimum patch coverage (fraction of pixels above
            ``density_thresh``) required for a patch's peak powers to count;
            ``0`` (default) disables the gate.

    Returns:
        A ``Dataset`` with variables:

        - ``F`` ``(..., px, py)``: power at the spectral peak (NaN where the
          coverage gate excludes the patch).
        - ``Fx``, ``Fy`` ``(..., px, py)``: wavenumbers at the peak (gated
          like ``F``).
        - ``objective`` ``(...,)``: sum of ``F`` over ``(px, py)`` (NaN
          skipped).
    """
    if isinstance(patch, xarray.Dataset):
        coverage = patch.get("patch_coverage", None)
        patch_da = patch["patch"]
    else:
        coverage = None
        patch_da = patch
    summary = (
        patch_da.isel(patch_da.argmax(dim=["kx", "ky"]))
        .to_dataset(name="F")
        .reset_coords()
        .rename_vars({"kx": "Fx", "ky": "Fy"})
    )
    if coverage is not None and min_coverage > 0:
        keep = coverage >= min_coverage
        for var in ("F", "Fx", "Fy"):
            summary[var] = summary[var].where(keep)
    return summary.assign(objective=lambda x: x.F.sum(dim=["px", "py"]))


def build_patch_stack(
    ts: TimeWindow,
    heights: np.ndarray,
    cfg: SpectralConfig,
    ch: WorkerChannel | None = None,
) -> xarray.Dataset | None:
    """Compute FFT patches at each height without keeping the gridded images.

    Used by the first-pass worker in two-pass mode, where only the objective
    surface is needed.  Skipping the per-height image concat saves a noticeable
    chunk of memory when ``len(heights)`` is large.

    Args:
        ts: Time window for this slice.
        heights: Array of IPP heights (km) to evaluate.
        cfg: Shared spectral configuration.
        ch: Optional worker channel; one progress tick is emitted per height.

    Returns:
        A ``Dataset`` with variable ``patch`` with dims
        ``(height, py, px, kx, ky)``, plus ``patch_coverage``
        ``(height, py, px)`` when ``cfg.density_thresh`` applies the density
        cut, or ``None`` if data retrieval failed for any height.
    """
    ch = ch or WorkerChannel()
    patches = []
    for height in heights:
        ch.info(
            "[%s - %s]: height = %.1f",
            _fmt_dt(ts.start_time),
            _fmt_dt(ts.end_time),
            height,
        )
        data = get_data(
            cfg.obs,
            cfg.rx,
            ts,
            height,
            cfg.lat_limits,
            cfg.lon_limits,
            proj=cfg.proj,
        )
        if data is None:
            ch.warning("[%s - %s]: FAIL", _fmt_dt(ts.start_time), _fmt_dt(ts.end_time))
            return None
        img = generate_image(data, cfg.image_maker, cfg.tec_name)
        patches.append(
            compute_patch_spectra(
                img,
                cfg.block_size,
                cfg.block_step,
                cfg.window,
                density_thresh=cfg.density_thresh,
                normalization=cfg.normalization,
            )
        )
        ch.report()
    return xarray.concat(patches, "height").assign_coords(height=heights)


def build_image_patch_stack(
    ts: TimeWindow,
    heights: np.ndarray,
    cfg: SpectralConfig,
    ch: WorkerChannel | None = None,
) -> xarray.Dataset | None:
    """Compute images and FFT patches at every requested height.

    Used by the single-pass worker, which needs both image data and patches
    so it can return the best height's full slice in one go.

    Args:
        ts: Time window for this slice.
        heights: Array of IPP heights (km) to process.
        cfg: Shared spectral configuration.
        ch: Optional worker channel; one progress tick is emitted per height.

    Returns:
        A ``Dataset`` with dims ``(height, x, y)`` containing variables
        ``image``, ``density``, ``n`` (data-point count per height),
        ``patch`` ``(height, py, px, kx, ky)``, and ``patch_coverage``
        ``(height, py, px)`` when ``cfg.density_thresh`` applies the density
        cut, or ``None`` if data retrieval failed for any height.
    """
    ch = ch or WorkerChannel()
    images = []
    patches = []
    npts = []
    for height in heights:
        ch.info(
            "[%s - %s]: height = %.1f",
            _fmt_dt(ts.start_time),
            _fmt_dt(ts.end_time),
            height,
        )
        data = get_data(
            cfg.obs,
            cfg.rx,
            ts,
            height,
            cfg.lat_limits,
            cfg.lon_limits,
            proj=cfg.proj,
        )
        if data is None:
            ch.warning("[%s - %s]: FAIL", _fmt_dt(ts.start_time), _fmt_dt(ts.end_time))
            return None
        npts.append(len(data))
        img = generate_image(data, cfg.image_maker, cfg.tec_name)
        patches.append(
            compute_patch_spectra(
                img,
                cfg.block_size,
                cfg.block_step,
                cfg.window,
                density_thresh=cfg.density_thresh,
                normalization=cfg.normalization,
            )
        )
        images.append(img)
        ch.report()

    patches_ds = xarray.concat(patches, "height").assign_coords(height=heights)
    return (
        xarray.concat(images, "height")
        .assign_coords(height=heights)
        .assign(n=(["height"], npts))
        .merge(patches_ds)
    )


def compute_height_objectives_for_slice(
    ts: TimeWindow,
    time: np.datetime64,
    heights: np.ndarray,
    cfg: SpectralConfig,
    ch: WorkerChannel | None = None,
) -> xarray.DataArray | None:
    """First-pass worker: return per-height FFT objectives for one time slice.

    Args:
        ts: Time window for this slice.
        time: Start timestamp; used as the ``time`` coordinate.
        heights: Array of heights (km) to evaluate.
        cfg: Shared spectral configuration.
        ch: Optional worker channel carrying the log and progress queues.

    Returns:
        A ``DataArray`` of objective values with dims ``(time, height)``,
        or ``None`` if data retrieval failed for any height.
    """
    with ch or WorkerChannel() as ch:
        ch.info(
            "[%s - %s]: processing heights", _fmt_dt(ts.start_time), _fmt_dt(ts.end_time)
        )
        patches = build_patch_stack(ts, heights, cfg, ch=ch)
        if patches is None:
            return None
        return extract_patch_peaks(
            patches, min_coverage=cfg.min_patch_coverage
        ).objective.expand_dims(time=[time])


def _select_best_height_for_slice(
    ts: TimeWindow,
    time: np.datetime64,
    heights: np.ndarray,
    cfg: SpectralConfig,
    ch: WorkerChannel | None = None,
) -> tuple[xarray.Dataset, xarray.DataArray] | None:
    """Single-pass worker: compute, select best height, and return full slice.

    Computes images and FFT patches at every candidate height, picks the
    height with the maximum objective, and returns the data at that height
    together with the per-height objective surface.  The returned slice
    already contains ``F``, ``Fx``, ``Fy``, and ``objective``, so no further
    ``extract_patch_peaks`` call is needed downstream.

    Args:
        ts: Time window for this slice.
        time: Start timestamp; used as the ``time`` coordinate.
        heights: Array of IPP heights (km) to evaluate.
        cfg: Shared spectral configuration.
        ch: Optional worker channel carrying the log and progress queues;
            one progress tick is emitted per height.

    Returns:
        A 2-tuple ``(best_slice, objectives)``:

        - ``best_slice``: ``Dataset`` with ``image``, ``density``, ``patch``,
          ``height``, ``F``, ``Fx``, ``Fy``, ``objective``, expanded along
          ``time``.
        - ``objectives``: ``DataArray`` with dims ``(time, height)``.

        Returns ``None`` if data retrieval failed for any height.
    """
    with ch or WorkerChannel() as ch:
        ch.info(
            "[%s - %s]: single-pass slice", _fmt_dt(ts.start_time), _fmt_dt(ts.end_time)
        )
        data_all = build_image_patch_stack(ts, heights, cfg, ch=ch)
        if data_all is None:
            return None
        summary = extract_patch_peaks(data_all, min_coverage=cfg.min_patch_coverage)
        best_idx = int(summary.objective.argmax())
        best_height = float(heights[best_idx])
        best_slice = (
            data_all.isel(height=best_idx)
            .drop_vars("height")
            .merge(summary.isel(height=best_idx).drop_vars("height"))
            .expand_dims(time=[time])
            .assign(height=("time", [best_height]))
        )
        obj_da = summary.objective.expand_dims(time=[time])
        return best_slice, obj_da


def _rebuild_slice_at_height(
    ts: TimeWindow,
    time: np.datetime64,
    height: float,
    cfg: SpectralConfig,
    ch: WorkerChannel | None = None,
    with_patch: bool = True,
) -> xarray.Dataset | None:
    """Second-pass worker: rebuild the image (and optionally patch) at a fixed focus height.

    Args:
        ts: Time window for this slice.
        time: Start timestamp; used as the ``time`` coordinate.
        height: IPP height (km) at which to evaluate this time slice.
        cfg: Shared spectral configuration.
        ch: Optional worker channel carrying the log and progress queues;
            one progress tick is emitted at the end of the slice.
        with_patch: Whether to compute the FFT patch; the image-only alt-track
            pass sets this to ``False`` to skip the patch computation.

    Returns:
        A ``Dataset`` with variables ``image`` and ``density`` (plus ``patch``,
        and ``patch_coverage`` when the density cut applies, when
        ``with_patch``), expanded along ``time``, or ``None`` if data
        retrieval failed.
    """
    with ch or WorkerChannel() as ch:
        ch.info("focused slice time=%s height=%.1f", _fmt_dt(time), height)
        data = get_data(
            cfg.obs,
            cfg.rx,
            ts,
            height,
            cfg.lat_limits,
            cfg.lon_limits,
            proj=cfg.proj,
        )
        if data is None:
            return None
        img = generate_image(data, cfg.image_maker, cfg.tec_name)
        if with_patch:
            p = compute_patch_spectra(
                img,
                cfg.block_size,
                cfg.block_step,
                cfg.window,
                density_thresh=cfg.density_thresh,
                normalization=cfg.normalization,
            )
            img = img.assign(patch=p.patch)
            if "patch_coverage" in p:
                img = img.assign(patch_coverage=p.patch_coverage)
        ch.report()
        return img.expand_dims(time=[time])


def _build_image_at_height(
    ts: TimeWindow,
    time: np.datetime64,
    height: float,
    cfg: SpectralConfig,
    ch: WorkerChannel | None = None,
) -> xarray.Dataset | None:
    """Second-pass worker: rebuild the image at a fixed focus height.

    Image-only variant of :func:`_rebuild_slice_at_height`, for the
    alternative height track whose image sequence is saved for comparison
    only.

    Args:
        ts: Time window for this slice.
        time: Start timestamp; used as the ``time`` coordinate.
        height: IPP height (km) at which to evaluate this time slice.
        cfg: Shared spectral configuration.
        ch: Optional worker channel carrying the log and progress queues;
            one progress tick is emitted at the end of the slice.

    Returns:
        A ``Dataset`` with variables ``image`` and ``density``, expanded
        along ``time``, or ``None`` if data retrieval failed.
    """
    return _rebuild_slice_at_height(ts, time, height, cfg, ch, with_patch=False)


def _valid_height_inputs(
    track: xarray.DataArray,
    time_windows: list,
    times: list,
) -> list:
    """Build ``(window, start-time, height)`` triples for valid windows.

    Args:
        track: Height track ``(time,)`` with a ``height`` coordinate; windows
            whose track height is NaN (failed first pass) are skipped.
        time_windows: Time windows aligned with ``times``.
        times: Window start timestamps, matching the track's labels.

    Returns:
        A list of ``(ts, time, height)`` triples for the windows with a
        finite track height.
    """
    return [
        (ts, time, track.isel(time=ii).height.item())
        for ii, (ts, time) in enumerate(zip(time_windows, times, strict=True))
        if not track.isel(time=ii).height.isnull().item()
    ]


def _select_focus_height(smoothed: xarray.DataArray) -> xarray.DataArray:
    """Pick the highest-objective height per time from a smoothed surface.

    Args:
        smoothed: Smoothed objective surface with dims ``(time, height)``;
            all-NaN time rows (failed windows) are preserved.

    Returns:
        A ``(time,)`` DataArray of the maximum smoothed objective along
        ``height``, with a ``height`` coordinate recording the selected
        height; dropped time rows are reindexed back as NaN.
    """
    smooth_nonull = smoothed.dropna(dim="time")
    return smooth_nonull.isel(height=smooth_nonull.argmax(dim="height")).reindex(
        time=smoothed.time
    )


def _build_sparse_image(
    data_focused: xarray.Dataset, density_thresh: float | None
) -> xarray.Dataset:
    """Flatten the focus-height image stack into a sparse ``(time, row)`` table.

    Pixels with density at or below ``density_thresh`` are masked out — skipped
    entirely when ``density_thresh`` is ``None`` — and rows/times that end up
    all-NaN are dropped.

    Args:
        data_focused: Focused dataset with variables ``image`` and ``density``
            on dims ``(time, x, y)``.
        density_thresh: Minimum image density (data points per pixel) required
            for a pixel to be included in the sparse image; ``None`` keeps
            every pixel.

    Returns:
        A ``Dataset`` with ``image`` stacked as a sparse ``(time, row)`` table
        with ``x``/``y`` index coordinates.
    """
    image = data_focused.image
    if density_thresh is not None and "density" in data_focused:
        image = image.where(data_focused.density > density_thresh)
    return (
        image.stack(row=("x", "y"))
        .reset_index("row")
        .dropna(dim="time", how="all")
        .dropna(dim="row", how="all")
        .reset_coords()
    )


def _save_objective_plots(
    obj_surface: xarray.DataArray,
    main_height: xarray.DataArray,
    smoothed: xarray.DataArray | None = None,
    smoothed_height: xarray.DataArray | None = None,
    log_smoothed: xarray.DataArray | None = None,
    log_height: xarray.DataArray | None = None,
) -> None:
    """Render and save the linear and log10 objective-surface diagnostics.

    Writes ``plots/objective_linear.png`` from the raw linear-power surface
    and ``plots/objective_log.png`` from the log10 of the positive-masked
    surface.  Non-positive objectives are masked to NaN before the log so
    fully-gated windows (objective 0) do not become ``-inf``.  Each plot
    optionally overlays a smoothed objective surface and its selected height
    track; ``None`` smoothed inputs render the raw surface only.

    Args:
        obj_surface: Objective surface ``(time, height)`` with variable
            ``objective`` values on those dims.
        main_height: Raw-track selected height ``(time,)``.
        smoothed: Optional smoothed linear-power surface ``(time, height)``
            overlaid on the linear plot.
        smoothed_height: Optional height track ``(time,)`` for ``smoothed``.
        log_smoothed: Optional smoothed log10 surface ``(time, height)``
            overlaid on the log plot.
        log_height: Optional height track ``(time,)`` for ``log_smoothed``.
    """
    fig, _ = plot_objective_surfaces(obj_surface, main_height, smoothed, smoothed_height)
    fig.savefig("plots/objective_linear.png")
    plt.close(fig)
    fig, _ = plot_objective_surfaces(
        np.log10(obj_surface.where(obj_surface > 0)),
        main_height,
        log_smoothed,
        log_height,
    )
    fig.savefig("plots/objective_log.png")
    plt.close(fig)


def _focus_per_slice(
    time_windows: list,
    times: list,
    heights: np.ndarray,
    cfg: SpectralConfig,
    n_jobs: int,
) -> xarray.Dataset:
    """Run the single-pass focusing branch and return the per-time best slice.

    Each worker selects the best height for its window and returns the
    corresponding image plus patch summary; no further ``extract_patch_peaks``
    call is needed downstream. Configuration is passed directly to tasks so
    joblib can share large input arrays through read-only memmaps.

    Args:
        time_windows: Observation row indices and time bounds for each slice.
        times: Start timestamps corresponding to ``time_windows``.
        heights: Candidate IPP heights in km.
        cfg: Configuration with categorical observation identifiers.
        n_jobs: Joblib worker count; one executes sequentially.

    Returns:
        Dataset with image and density on ``(time, x, y)``, patch spectra on
        ``(time, py, px, kx, ky)``, and selected height and objective on
        ``(time,)``. Failed slices are reindexed to missing values.
    """
    logger.info("run_spectral_focusing: single pass (no smoothing), n_jobs=%d", n_jobs)
    total = len(time_windows) * len(heights)
    with (
        worker_channels(total, desc="focus (single pass)") as ch,
        Parallel(n_jobs=n_jobs, max_nbytes="1M", mmap_mode="r") as parallel,
    ):
        single_slices = parallel(
            delayed(_select_best_height_for_slice)(ts, time, heights, cfg, ch)
            for ts, time in zip(time_windows, times, strict=True)
        )

    valid_results = [r for r in single_slices if r is not None]
    if not valid_results:
        raise RuntimeError("single pass produced no valid slices")

    valid_slices, per_height_objs = zip(*valid_results, strict=True)
    data_focused = xarray.concat(valid_slices, "time").reindex(time=times)

    obj_surface = xarray.concat(per_height_objs, dim="time").reindex(time=times)
    _save_objective_plots(obj_surface, data_focused.height)
    return data_focused


def _focus_with_time_smoothing(
    time_windows: list,
    times: list,
    heights: np.ndarray,
    cfg: SpectralConfig,
    n_jobs: int,
    time_window: int,
) -> xarray.Dataset:
    """Run the two-pass focusing branch: objectives → smoothing → reconstruction.

    Both smoothing reductions are computed: an arithmetic mean on the linear
    objective surface and a geometric mean (mean of log10 power).  Each
    reduction selects its own height track; ``cfg.logscale_objective`` picks
    the track whose sequence carries the FFT patches and is used for center
    finding, while both tracks' objectives and image sequences are kept in
    the result.

    One pool spans pass 1, the main reconstruction pass and the image-only
    alt pass. All tasks receive the same configuration object so joblib can
    reuse read-only memmaps of its large input arrays across phases.

    Args:
        time_windows: Observation row indices and time bounds for each slice.
        times: Start timestamps corresponding to ``time_windows``.
        heights: Candidate IPP heights in km.
        cfg: Configuration with categorical observation identifiers.
        n_jobs: Joblib worker count; one executes sequentially.
        time_window: Centered rolling-mean window length in time steps.

    Returns:
        Dataset with image and density on ``(time, x, y)``, patch spectra on
        ``(time, py, px, kx, ky)``, both smoothed height/objective tracks on
        ``(time,)``, and alternative-track image/density on ``(time, x, y)``.
        Dimension coordinates are inherited from the image and spectral grids.
    """
    # First pass — objectives only (images discarded to save memory)
    logger.info("run_spectral_focusing: first pass (objectives), n_jobs=%d", n_jobs)
    total_pass1 = len(time_windows) * len(heights)
    t_pass1_start = perf_counter()
    with Parallel(n_jobs=n_jobs, max_nbytes="1M", mmap_mode="r") as parallel:
        with worker_channels(total_pass1, desc="focus pass 1 (objectives)") as ch:
            objectives = parallel(
                delayed(compute_height_objectives_for_slice)(ts, time, heights, cfg, ch)
                for ts, time in zip(time_windows, times, strict=True)
            )
        logger.info("pass 1 done in %.1f s", perf_counter() - t_pass1_start)

        obj_data = xarray.concat(
            filter(lambda x: x is not None, objectives), dim="time"
        ).reindex(time=times)

        # Height smoothing — arithmetic mean on the objective surface (linear
        # power) and geometric mean (mean of log10 power).  The geometric mean
        # down-weights occasional huge peaks in favour of heights with
        # consistently moderate objectives; logscale_objective decides which
        # track's sequence is used for center finding.
        rolling = {"time": time_window, "center": True, "min_periods": 1}
        smoothed_linear = obj_data.rolling(**rolling).mean()
        # Fully-gated windows yield objective 0; log10(0) would be -inf, which the
        # rolling mean does not skip (unlike NaN), so non-positive objectives are
        # masked to NaN first and fully-gated windows are dropped exactly like
        # failed ones.
        smoothed_log = np.log10(obj_data.where(obj_data > 0)).rolling(**rolling).mean()

        focus_linear = _select_focus_height(smoothed_linear)
        focus_log = _select_focus_height(smoothed_log)
        focus_raw = _select_focus_height(obj_data)

        focus_height = focus_log if cfg.logscale_objective else focus_linear
        focus_alt = focus_linear if cfg.logscale_objective else focus_log
        alt_suffix = "linear" if cfg.logscale_objective else "log"

        _save_objective_plots(
            obj_data,
            focus_raw.height,
            smoothed_linear,
            focus_linear.height,
            smoothed_log,
            focus_log.height,
        )

        # Second pass — reconstruct images and FFT patches at the selected focus
        # heights, then an image-only pass at the alternative track's heights
        second_pass_inputs = _valid_height_inputs(focus_height, time_windows, times)
        if not second_pass_inputs:
            raise RuntimeError("no valid focus heights after first pass; cannot continue")

        logger.info(
            "run_spectral_focusing: second pass (%d / %d windows have valid focus height)",
            len(second_pass_inputs),
            len(time_windows),
        )
        t_pass2_start = perf_counter()
        with worker_channels(
            len(second_pass_inputs), desc="focus pass 2 (reconstruction)"
        ) as ch:
            focused_slices = parallel(
                delayed(_rebuild_slice_at_height)(ts, time, height, cfg, ch)
                for ts, time, height in second_pass_inputs
            )
        logger.info("pass 2 done in %.1f s", perf_counter() - t_pass2_start)

        valid_slices = [s for s in focused_slices if s is not None]
        if not valid_slices:
            raise RuntimeError("second pass produced no valid slices")

        # Alt-track sequence: windows where both tracks selected the same height
        # and the main rebuild succeeded already have an identical image, so only
        # the remaining heights are rebuilt image-only.
        data_focused = xarray.concat(valid_slices, "time")
        main_height_by_time = {time: h for _, time, h in second_pass_inputs}
        rebuilt_labels = set(data_focused.time.values)
        alt_inputs = _valid_height_inputs(focus_alt, time_windows, times)
        reused_times = [
            time
            for _, time, h in alt_inputs
            if time in rebuilt_labels and main_height_by_time[time] == h
        ]
        alt_rebuild = [
            (ts, time, h)
            for ts, time, h in alt_inputs
            if time not in rebuilt_labels or main_height_by_time[time] != h
        ]
        data_alt = None
        if alt_rebuild:
            logger.info(
                "run_spectral_focusing: alt pass (%d / %d windows)",
                len(alt_rebuild),
                len(time_windows),
            )
            t_alt_start = perf_counter()
            with worker_channels(len(alt_rebuild), desc="focus alt pass (images)") as ch:
                alt_slices = parallel(
                    delayed(_build_image_at_height)(ts, time, height, cfg, ch)
                    for ts, time, height in alt_rebuild
                )
            logger.info("alt pass done in %.1f s", perf_counter() - t_alt_start)
            valid_alt = [s for s in alt_slices if s is not None]
            if valid_alt:
                data_alt = xarray.concat(valid_alt, "time").reindex(time=focus_alt.time)
            else:
                logger.warning("run_spectral_focusing: alt pass produced no valid slices")

        extras = {
            "objective_smoothed_linear": focus_linear.reset_coords(drop=True),
            "objective_smoothed_log": focus_log.reset_coords(drop=True),
            "height_linear": focus_linear.height.reset_coords(drop=True),
            "height_log": focus_log.height.reset_coords(drop=True),
        }
        if alt_inputs:
            alt_image = data_alt.image if data_alt is not None else None
            alt_density = data_alt.density if data_alt is not None else None
            if reused_times:
                reused = data_focused[["image", "density"]].sel(time=reused_times)
                if alt_image is None:
                    alt_image = reused.image
                    alt_density = reused.density
                else:
                    alt_image = alt_image.combine_first(reused.image)
                    alt_density = alt_density.combine_first(reused.density)
            if alt_image is not None:
                extras[f"image_{alt_suffix}"] = alt_image.reindex(time=focus_alt.time)
                extras[f"density_{alt_suffix}"] = alt_density.reindex(time=focus_alt.time)
        return (
            data_focused.merge(
                extract_patch_peaks(data_focused, min_coverage=cfg.min_patch_coverage)
            )
            .reindex(time=focus_height.time)
            .assign(height=focus_height.height)
            .assign(**extras)
        )


def run_spectral_focusing(
    cfg: SpectralConfig,
    window_size: int,
    step: int,
    heights: np.ndarray,
    n_jobs: int,
    time_window: int = 1,
    center_finder: Callable | None = None,
) -> xarray.Dataset:
    """Unified spectral focusing pipeline.

    Orchestrates spectral focusing over a sequence of time windows in two modes
    depending on ``time_window``:

    **Single-pass** (``time_window=1``):

    1. **Init**: find the time window with the most data and initialize the
       image maker's interpolation grid.
    2. **Single pass**: for each time window, compute images and FFT-patch
       objectives at all candidate heights, then immediately select the height
       with the highest objective.  No smoothing is applied.

    **Two-pass** (``time_window>1``):

    1. **Init**: same as above.
    2. **First pass**: for each time window compute only the per-height FFT
       objectives (images are discarded to save memory).
    3. **Height smoothing**: apply both rolling reductions over time to the
       objective surface — an arithmetic mean of the linear objective and a
       geometric mean of log10 — selecting one height track per reduction.
       ``logscale_objective`` picks the track used for the main sequence.
    4. **Second pass**: for each time window reconstruct the image and FFT
       patches at the selected focus height, plus an image-only
       reconstruction at the alternative track's heights.

    When ``cfg.density_thresh`` is set, the patch spectra are computed on a
    density-cut, zero-filled copy of the image and peak powers from patches
    whose coverage (fraction of pixels above ``cfg.density_thresh``) falls
    below ``cfg.min_patch_coverage`` are gated to NaN, so the objective, the
    height selection and the center-finding seed only see data-supported
    patches; the stored ``image`` stays unmasked.  ``cfg.density_thresh=None``
    (the dataclass default) disables the cut and the gate.

    ``cfg.normalization`` may z-score each image (``"image"``) or each patch
    (``"patch"``) before its FFT; ``None`` leaves raw amplitudes unchanged.
    This scaling affects only patch spectra, never the stored images.

    Observation identifiers ``rx`` and ``sv`` are converted to categoricals
    on a run-local DataFrame without changing the caller's observations.
    Large numeric, datetime, and categorical-code arrays are automatically
    shared with process workers through joblib's read-only input memmaps.
    Worker results are returned normally, not through shared output buffers.

    Both modes optionally finish with a global center-finding step that fits
    source parameters jointly across all times on the
    ``logscale_objective``-selected sequence; the center-finding
    initialization window is the time with the highest raw linear objective.
    The step runs only when ``center_finder`` is provided; when
    ``center_finder=None`` center finding is skipped (logged at INFO level)
    and the returned dataset has no ``center``, ``wavelength``, ``offset`` or
    ``phase`` variables.

    Args:
        cfg: Shared spectral configuration (observations, receiver table,
            image maker, FFT-block geometry, windowing, region bounds, TEC
            variable name, patch density gate, and normalization mode).  See
            :class:`SpectralConfig` for fields.
        window_size: Number of unique time steps per sliding window.
        step: Stride (in time steps) between consecutive windows.
        heights: Array of IPP heights (km) to search.
        n_jobs: Number of parallel worker processes.
        time_window: Rolling-mean window length (in time steps) applied to the
            height-objective surface.  ``1`` disables smoothing (single-pass
            mode).  Default ``1``.
        center_finder: Callable ``(c0, w0, x, y, image) -> dict`` used for
            global source-parameter estimation.  ``None`` skips center
            finding entirely.  Default ``None``.

    Returns:
        An ``xarray.Dataset`` with dimensions ``(time,)`` and variables:

        - ``image`` ``(time, x, y)``: gridded TEC image at the focus height
          selected by the ``logscale_objective`` smoothing (two-pass mode).
        - ``patch`` ``(time, px, py, kx, ky)``: FFT patch spectra (computed on
          the density-cut, zero-filled image when ``cfg.density_thresh`` is
          set; ungated raw diagnostics).
        - ``patch_coverage`` ``(time, px, py)``: fraction of each patch's
          pixels above ``cfg.density_thresh`` (only present when the density
          cut is applied).
        - ``F``, ``Fx``, ``Fy`` ``(time, px, py)``: peak power and wavenumbers
          (NaN where the patch coverage falls below
          ``cfg.min_patch_coverage`` when the density cut applies).
        - ``objective`` ``(time,)``: summed peak power at the selected heights
          (linear power; NaN-skipping, so gated patches are excluded; argmax
          over this picks the initialization window).
        - ``height`` ``(time,)``: focus height (km) of the main sequence.
        - ``objective_smoothed_linear``, ``objective_smoothed_log`` ``(time,)``:
          both smoothed objectives at their respective selected heights
          (two-pass mode only; linear power for the arithmetic-mean track,
          log10 units for the geometric-mean track).
        - ``height_linear``, ``height_log`` ``(time,)``: height tracks selected
          by each smoothing (two-pass mode only).
        - ``image_linear``/``image_log``, ``density_linear``/``density_log``
          ``(time, x, y)``: image sequence at the height track not chosen by
          ``logscale_objective`` (two-pass mode only; the suffix names that
          track's smoothing, and the chosen track is stored as
          ``image``/``density``).
        - ``center`` ``(ci,)``: estimated source center ``[x, y]`` (km, global;
          only present when center finding runs).
        - ``wavelength`` ``(time,)``: estimated horizontal wavelength (km; only
          present when center finding runs).
        - ``offset`` ``(time,)``: estimated DC offset (only present when center
          finding runs).
        - ``phase`` ``(time,)``: estimated wave phase (rad; only present when
          center finding runs).

        Attributes include ``coord_center`` as ``(mean_lat, mean_lon)``.
    """
    cfg = replace(cfg, obs=cfg.obs.astype({"rx": "category", "sv": "category"}))
    cfg.image_maker.initialize_from_bounds(
        proj=cfg.proj,
        lat_limits=cfg.lat_limits,
        lon_limits=cfg.lon_limits,
    )

    time_windows = make_time_windows(cfg.obs["time"], window_size, step)
    times = [w.start_time for w in time_windows]
    Path("plots").mkdir(exist_ok=True)

    t_total_start = perf_counter()
    if time_window == 1:
        t_pass_start = perf_counter()
        data_focused = _focus_per_slice(time_windows, times, heights, cfg, n_jobs)
        logger.info("single pass done in %.1f s", perf_counter() - t_pass_start)
    else:
        data_focused = _focus_with_time_smoothing(
            time_windows, times, heights, cfg, n_jobs, time_window
        )

    init_slice = data_focused.isel(time=data_focused.objective.argmax())

    # Persist diagnostic inputs before plotting so they survive plot failures.
    # init_slice.reset_coords().to_netcdf("plots/init_slice.nc")
    # sparse_img.reset_coords().to_netcdf("plots/sparse_img.nc")
    logger.info(
        "init_slice diagnostics | time=%s height=%s objective=%s F max=%s",
        np.datetime_as_string(init_slice.time.values, unit="s"),
        float(init_slice.height.values),
        float(init_slice.objective.values),
        float(np.nanmax(init_slice.F.values)),
    )

    fig, _ = plot_pre_center_finder(init_slice, density_thresh=cfg.density_thresh)
    fig.savefig("plots/pre_center_finder.png")
    plt.close(fig)

    fig, _ = plot_density_map(init_slice)
    fig.savefig("plots/density.png")
    plt.close(fig)

    fig, _ = plot_peak_patch_spectrum(init_slice)
    fig.savefig("plots/peak_patch.png")
    plt.close(fig)

    fig, _ = plot_center_finder(data_focused, density_thresh=cfg.density_thresh)
    fig.savefig("plots/center_init.png")
    plt.close(fig)

    coord_center = (np.mean(cfg.lat_limits), np.mean(cfg.lon_limits))
    if center_finder is None:
        logger.info("center_finder not configured; skipping center finding")
        result = data_focused
    else:
        sparse_img = _build_sparse_image(data_focused, cfg.density_thresh)

        t_center_start = perf_counter()
        params = run_smoothed_center_finder(
            px=data_focused.px.values,
            py=data_focused.py.values,
            F=data_focused.F.values,
            Fx=data_focused.Fx.values,
            Fy=data_focused.Fy.values,
            sparse_x=sparse_img.x.values,
            sparse_y=sparse_img.y.values,
            sparse_image=sparse_img.image.values.T,
            image_x=data_focused.x.values,
            image_y=data_focused.y.values,
            center_finder=center_finder,
        )
        logger.info(
            "center finding done in %.1f s (%d iterations)",
            perf_counter() - t_center_start,
            len(params["history"]["metric"]),
        )
        result = data_focused.assign(
            center=("ci", params["center"]),
            wavelength=xarray.DataArray(
                params["wavelength"], coords={"time": sparse_img.time}
            ),
            offset=xarray.DataArray(params["offset"], coords={"time": sparse_img.time}),
            phase=xarray.DataArray(params["phase"], coords={"time": sparse_img.time}),
        )

    logger.info("total time: %.1f s", perf_counter() - t_total_start)
    return result.assign_attrs(coord_center=coord_center)
