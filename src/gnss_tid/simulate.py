"""Composable TID synthetic-simulation pipeline.

Implements the analytical-wave scenario core (waves, Gaussian envelopes,
source summation), geometry samplers, and the analytical dTEC forward model.
The mathematical model is independent of observation geometry and imaging
code: a :class:`Scenario` evaluates dTEC (TECU) at arbitrary ``(x, y, t)``
coordinates, samplers produce locations/times only, and forward models
evaluate the physical scene at those coordinates.

Units and conventions
---------------------
- ``x``/``y`` are local Cartesian kilometers in the chosen projection (a
  plain Cartesian km frame for grid controls; a :class:`~gnss_tid.coords.Local2D`
  frame for geographic samplers).
- ``t_seconds`` are relative to the scenario ``reference_time`` epoch.
- dTEC and amplitudes are shell-equivalent TECU evaluated with **no
  elevation-dependent slant mapping**; no ionospheric background or
  detrending is involved in the analytical mode.
- Carrier phase uses ``t`` in seconds; wavelength is in km.
- Direction ``theta`` for planar waves is measured counterclockwise from +x.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import dask
import dask.array as da
import numpy as np
import pandas as pd
import pymap3d
import xarray as xr

# ---------------------------------------------------------------------------
# Waves and envelopes
# ---------------------------------------------------------------------------


@runtime_checkable
class Wave(Protocol):
    """Carrier wave supplying propagation coordinates and phase."""

    def uv(self, x, y) -> tuple:
        """Return ``(u, v)`` propagation/transverse coordinates in km."""
        ...

    def phase(self, u, t_seconds) -> xr.DataArray | np.ndarray:
        """Return carrier phase in radians at distance ``u`` (km) and time."""
        ...


@runtime_checkable
class Envelope(Protocol):
    """Dimensionless amplitude multiplier evaluated in wave coordinates."""

    def amplitude(self, u, v, t_seconds) -> xr.DataArray | np.ndarray:
        """Return the dimensionless envelope multiplier."""
        ...


@dataclass(frozen=True)
class PlanarWave:
    """Unbounded planar carrier wave.

    Attributes:
        wavelength_km: Carrier wavelength in km (must be positive).
        period_s: Carrier period in seconds (must be positive).
        direction_rad: Propagation direction counterclockwise from +x.
        center_km: ``(x0, y0)`` wave center in km.
        phase_rad: Carrier phase at the center at ``t_seconds = 0``.
    """

    wavelength_km: float
    period_s: float
    direction_rad: float = 0.0
    center_km: tuple[float, float] = (0.0, 0.0)
    phase_rad: float = 0.0

    def __post_init__(self) -> None:
        _validate_positive(self.wavelength_km, "wavelength_km")
        _validate_positive(self.period_s, "period_s")
        if not np.all(np.isfinite(np.asarray(self.direction_rad, dtype=float))):
            raise ValueError("direction_rad must be finite")
        if len(self.center_km) != 2 or not np.all(np.isfinite(self.center_km)):
            raise ValueError("center_km must be two finite values")

    def uv(self, x, y):
        """Return propagation and transverse coordinates for planar geometry.

        Args:
            x: Along-x positions in km (any array-like or scalar).
            y: Along-y positions in km.

        Returns:
            Tuple ``(u, v)`` in km where
            ``u = (x-x0) cos(theta) + (y-y0) sin(theta)`` and
            ``v = -(x-x0) sin(theta) + (y-y0) cos(theta)``.
        """
        theta = self.direction_rad
        dx = x - self.center_km[0]
        dy = y - self.center_km[1]
        u = dx * np.cos(theta) + dy * np.sin(theta)
        v = -dx * np.sin(theta) + dy * np.cos(theta)
        return u, v

    def phase(self, u, t_seconds):
        """Return carrier phase in radians.

        Args:
            u: Along-propagation distance in km.
            t_seconds: Time relative to the scenario epoch in seconds.

        Returns:
            Phase ``2*pi/lambda*u - 2*pi/T*t + phi0`` in radians.
        """
        return (
            2 * np.pi / self.wavelength_km * u
            - 2 * np.pi / self.period_s * t_seconds
            + self.phase_rad
        )


@dataclass(frozen=True)
class CircularWave:
    """Concentric (circular) carrier wave expanding outward from a center.

    Attributes:
        wavelength_km: Carrier wavelength in km (must be positive).
        period_s: Carrier period in seconds (must be positive).
        center_km: ``(x0, y0)`` center of the circular source in km.
        phase_rad: Carrier phase at zero radius at ``t_seconds = 0``.
    """

    wavelength_km: float
    period_s: float
    center_km: tuple[float, float] = (0.0, 0.0)
    phase_rad: float = 0.0

    def __post_init__(self) -> None:
        _validate_positive(self.wavelength_km, "wavelength_km")
        _validate_positive(self.period_s, "period_s")
        if len(self.center_km) != 2 or not np.all(np.isfinite(self.center_km)):
            raise ValueError("center_km must be two finite values")

    def uv(self, x, y):
        """Return radial and (zero) transverse coordinates.

        Args:
            x: Along-x positions in km.
            y: Along-y positions in km.

        Returns:
            Tuple ``(u, v)`` where ``u = hypot(x-x0, y-y0)`` is the radial
            distance in km and ``v = 0``.
        """
        u = np.hypot(x - self.center_km[0], y - self.center_km[1])
        v = 0.0 * np.asarray(u)
        return u, v

    def phase(self, u, t_seconds):
        """Return carrier phase in radians (outward propagation).

        Args:
            u: Radial distance in km.
            t_seconds: Time relative to the scenario epoch in seconds.

        Returns:
            Phase ``2*pi/lambda*u - 2*pi/T*t + phi0`` in radians; crests move
            outward at ``lambda/T``.
        """
        return (
            2 * np.pi / self.wavelength_km * u
            - 2 * np.pi / self.period_s * t_seconds
            + self.phase_rad
        )


@dataclass(frozen=True)
class GaussianEnvelope:
    """Moving, spreading Gaussian envelope in wave coordinates.

    The envelope center moves along +u at ``alpha * lambda / T`` (the
    ``speed_ratio`` times the carrier phase speed) and the along-propagation
    width spreads as ``sigma_u0 * sqrt(1 + (t/t_s)^2)``. The square-root
    amplitude prefactor approximately conserves packet energy for
    one-dimensional width spreading; this is not a full gravity-wave
    dispersion solver (no radial geometric attenuation of expanding rings,
    altitude-dependent propagation, neutral winds, or varying ion-neutral
    coupling are modeled).

    Attributes:
        sigma_u_km: Initial along-propagation (or radial) standard deviation
            in km (must be positive).
        sigma_v_km: Transverse standard deviation in km, or ``None`` to omit
            the transverse term (required for a full circular annulus).
        speed_ratio: Envelope-speed / carrier-phase-speed ratio
            (``>= 0``; 0 gives a stationary envelope).
        center_u_km: Initial along-propagation center in km; for circular
            waves this is the initial ring radius (``>= 0``).
        spreading_time_s: Spreading timescale in seconds; ``None`` disables
            spreading (constant width, unit prefactor).
    """

    sigma_u_km: float
    sigma_v_km: float | None = None
    speed_ratio: float = 0.9
    center_u_km: float = 0.0
    spreading_time_s: float | None = None

    def __post_init__(self) -> None:
        _validate_positive(self.sigma_u_km, "sigma_u_km")
        if self.sigma_v_km is not None:
            _validate_positive(self.sigma_v_km, "sigma_v_km")
        if self.speed_ratio < 0 or not np.isfinite(self.speed_ratio):
            raise ValueError("speed_ratio must be finite and >= 0")
        if not np.isfinite(self.center_u_km):
            raise ValueError("center_u_km must be finite")
        if self.spreading_time_s is not None:
            _validate_positive(self.spreading_time_s, "spreading_time_s")

    def amplitude(self, u, v, t_seconds, phase_speed: float):
        """Return the dimensionless envelope multiplier.

        Args:
            u: Along-propagation (or radial) distance in km.
            v: Transverse coordinate in km (ignored when ``sigma_v_km`` is
                ``None``).
            t_seconds: Time relative to the scenario epoch in seconds.
            phase_speed: Carrier phase speed in km/s (``lambda/T`` of the
                paired wave), supplied by the scenario evaluator so the
                envelope center moves at ``speed_ratio * phase_speed``.

        Returns:
            Envelope values ``E(u, v, t)`` with the same broadcast shape as
            ``u``; unit multiplier when not spreading and centered at the
            origin.
        """
        ts = self.spreading_time_s
        if ts is None or np.isinf(ts):
            sigma_u = self.sigma_u_km
            prefactor = 1.0
        else:
            sigma_u = self.sigma_u_km * np.sqrt(1.0 + (t_seconds / ts) ** 2)
            prefactor = np.sqrt(self.sigma_u_km / sigma_u)
        center = self.center_u_km + self.speed_ratio * phase_speed * t_seconds
        exp = -((u - center) ** 2) / (2.0 * sigma_u**2)
        if self.sigma_v_km is not None:
            exp = exp - v**2 / (2.0 * self.sigma_v_km**2)
        return prefactor * np.exp(exp)


# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Source:
    """One independent TID source: carrier wave, optional envelope, amplitude.

    Attributes:
        wave: Carrier wave (:class:`PlanarWave` or :class:`CircularWave`).
        envelope: Optional :class:`GaussianEnvelope`; ``None`` gives the
            unbounded-cosine case with a unit amplitude multiplier.
        amplitude_tecu: Signed dTEC amplitude in TECU (scalar or trial-indexed
            array-like).
    """

    wave: Wave
    envelope: GaussianEnvelope | None
    amplitude_tecu: float | xr.DataArray

    def __post_init__(self) -> None:
        values = np.asarray(self.amplitude_tecu, dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("amplitude_tecu must be finite")


@dataclass(frozen=True)
class Scenario:
    """Analytical-wave scenario: one or more sources, or zero for noise-only.

    Attributes:
        sources: Tuple of :class:`Source` (may be empty for a noise-only
            control).
        reference_time: UTC scenario epoch; ``t_seconds`` are relative to it.
    """

    sources: tuple[Source, ...]
    reference_time: np.datetime64

    def evaluate(self, x, y, t_seconds) -> xr.DataArray | np.ndarray:
        """Evaluate the summed dTEC field at the given coordinates.

        Args:
            x: Along-x positions in km (scalars, NumPy arrays, or
                xarray/Dask DataArrays). xarray inputs are broadcast by
                dimension name together with any trial-indexed source
                parameters; NumPy inputs must already be broadcast against
                each other (e.g. via ``np.meshgrid``).
            y: Along-y positions in km.
            t_seconds: Times relative to ``reference_time`` in seconds.

        Returns:
            dTEC in TECU (sum of all source contributions) with the broadcast
            shape of ``x``, ``y``, ``t_seconds``, and trial-indexed source
            parameters. Works on NumPy, xarray, and Dask-backed inputs. An
            empty source list returns zeros with the requested broadcast
            dimensions.
        """
        if not self.sources:
            zero = 0.0 * (x + y + t_seconds)
            return zero
        if not all(isinstance(a, xr.DataArray) for a in (x, y, t_seconds)):
            return self._evaluate_sources(list(self.sources), x, y, t_seconds)

        # Broadcast every trial-indexed source parameter with x/y/t by dim
        # name, then evaluate on raw arrays (no Python loops over trials).
        parameters = _TrialParameterCollector()
        for i, source in enumerate(self.sources):
            parameters.add("amplitude_tecu", i, source.amplitude_tecu)
            for name, value in vars(source.wave).items():
                parameters.add(name, i, value)
            if source.envelope is not None:
                for name, value in vars(source.envelope).items():
                    parameters.add(name, i, value)
        pieces = xr.broadcast(x, y, t_seconds, *parameters.arrays)
        has_trial = "trial" in pieces[0].dims
        lead_dims = [d for d in pieces[0].dims if d != "trial"]
        order = (["trial"] if has_trial else []) + lead_dims
        pieces = tuple(p.transpose(*order).data for p in pieces)
        x, y, t_seconds = pieces[0], pieces[1], pieces[2]
        sources = parameters.rebind(self.sources, pieces[3:])
        total = self._evaluate_sources(sources, x, y, t_seconds)
        return xr.DataArray(total, dims=tuple(order))

    def _evaluate_sources(self, sources, x, y, t_seconds):
        """Sum source contributions on broadcast-ready raw arrays.

        NumPy inputs are broadcast against each other so scalar, vector, and
        meshgrid combinations all work; Dask inputs must already share a
        broadcast-compatible shape.

        Args:
            sources: Source instances with broadcast-ready parameters.
            x: Along-x positions in km (raw NumPy/Dask arrays or scalars).
            y: Along-y positions in km.
            t_seconds: Times in seconds relative to the epoch.

        Returns:
            Summed dTEC in TECU with the broadcast array shape.
        """
        if not isinstance(x, da.Array):
            x, y, t_seconds = np.broadcast_arrays(
                *(np.asarray(a, dtype=float) for a in (x, y, t_seconds))
            )
        total = None
        for source in sources:
            u, v = source.wave.uv(x, y)
            phase = source.wave.phase(u, t_seconds)
            if source.envelope is not None:
                phase_speed = source.wave.wavelength_km / source.wave.period_s
                envelope = source.envelope.amplitude(u, v, t_seconds, phase_speed)
            else:
                envelope = 1.0
            contribution = source.amplitude_tecu * envelope * np.cos(phase)
            total = contribution if total is None else total + contribution
        return total


class _TrialParameterCollector:
    """Collects trial-indexed source parameters for broadcast evaluation.

    Gathers xarray-backed parameter values from all sources into a flat
    list of broadcast pieces so ``Scenario.evaluate`` can broadcast them
    together with the coordinate inputs, then rebinds the broadcast raw
    arrays back onto fresh source instances.
    """

    def __init__(self) -> None:
        self._entries: list[tuple[str, int, xr.DataArray]] = []

    def add(self, name: str, source_index: int, value) -> None:
        """Register one parameter value if it is trial-indexed.

        Args:
            name: Parameter attribute name.
            source_index: Index of the owning source.
            value: Parameter value; only xarray DataArrays are collected.
        """
        if isinstance(value, xr.DataArray) and "trial" in value.dims:
            self._entries.append((name, source_index, value))

    @property
    def arrays(self) -> list[xr.DataArray]:
        """Return the collected trial-indexed arrays in registration order.

        Returns:
            List of xarray DataArrays with a ``trial`` dimension.
        """
        return [entry[2] for entry in self._entries]

    def rebind(self, sources: tuple[Source, ...], pieces) -> list[Source]:
        """Rebind broadcast raw arrays onto fresh source instances.

        Args:
            sources: Original sources in registration order.
            pieces: Broadcast raw arrays (NumPy/Dask) matching
                :attr:`arrays` in registration order.

        Returns:
            New source list with broadcast parameters replacing the
            trial-indexed DataArrays. Wave/envelope parameters are rebound
            onto replaced wave/envelope dataclass instances.
        """
        by_source: dict[int, dict[str, object]] = {}
        for (name, idx, _), piece in zip(self._entries, pieces, strict=True):
            by_source.setdefault(idx, {})[name] = piece
        rebound = []
        for i, source in enumerate(sources):
            updates = by_source.get(i, {})
            wave = source.wave
            envelope = source.envelope
            wave_updates = {k: v for k, v in updates.items() if hasattr(wave, k)}
            env_updates = (
                {k: v for k, v in updates.items() if hasattr(envelope, k)}
                if envelope is not None
                else {}
            )
            if wave_updates:
                wave = dataclasses.replace(wave, **wave_updates)
            if envelope is not None and env_updates:
                envelope = dataclasses.replace(envelope, **env_updates)
            amplitude = updates.get("amplitude_tecu", source.amplitude_tecu)
            rebound.append(Source(wave, envelope, amplitude) if updates else source)
        return rebound


def _validate_positive(value, name: str) -> None:
    """Raise ValueError unless ``value`` is finite and positive.

    Accepts scalars, trial-indexed xarray DataArrays, and NumPy/Dask arrays
    whose values must all be finite and positive.

    Args:
        value: Numeric value or array to validate.
        name: Parameter name used in the error message.

    Raises:
        ValueError: If the value is not finite or not positive.
    """
    if isinstance(value, xr.DataArray):
        arr = np.asarray(value.values, dtype=float)
    elif isinstance(value, da.Array):
        arr = np.asarray(value.compute(), dtype=float)
    elif isinstance(value, np.ndarray):
        arr = np.asarray(value, dtype=float)
    else:
        arr = np.asarray(value, dtype=float).reshape(())
    if arr.size > 1:
        if not np.all(np.isfinite(arr)) or np.any(arr <= 0):
            raise ValueError(f"{name} values must be finite and > 0")
        return
    value = float(arr.reshape(()))
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and > 0, got {value}")


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GridSamples:
    """Uniform-grid sample geometry.

    Attributes:
        x: 1-D coordinate array (dim ``x``, km).
        y: 1-D coordinate array (dim ``y``, km).
        t_seconds: 1-D coordinate array (dim ``time``, seconds relative to
            the scenario epoch).
        time: Actual datetime timestamps (dim ``time``).
    """

    x: xr.DataArray
    y: xr.DataArray
    t_seconds: xr.DataArray
    time: xr.DataArray


@dataclass
class GNSSSamples:
    """Epoch-level GNSS sample geometry (one row per original observation).

    Attributes:
        obs: DataFrame with one row per original observation containing
            ``obs_id``, ``time``, ``rx``, ``sv``, ``az``, ``el``, ``x`` (km),
            ``y`` (km), and ``t_seconds``. ``obs_id`` is stable and
            independent of DataFrame index labels.
        rx: Receiver metadata, unique by receiver identity, with geodetic
            ``lat``/``lon`` (degrees), ``alt`` (km), and ECEF coordinates
            (km).
        proj: Shared :class:`~gnss_tid.coords.Local2D` projection frame.
        true_height_km: True thin-shell height in km used for IPP projection.
    """

    obs: pd.DataFrame
    rx: pd.DataFrame
    proj: object
    true_height_km: float


@runtime_checkable
class Sampler(Protocol):
    """Geometry sampler returning locations and times, not TEC."""

    def sample(self, reference_time: np.datetime64) -> GridSamples | GNSSSamples:
        """Return sample geometry for the given scenario epoch."""
        ...


class GridSampler:
    """Regular ``(time, y, x)`` grid sampler (primary fast path).

    Attributes:
        start_time: Requested start timestamp (UTC, timezone-naive).
        end_time: Requested end timestamp, aligned to the cadence.
        cadence_s: Positive raw sampling cadence in seconds.
        xlim: ``(min, max)`` x extent in km.
        ylim: ``(min, max)`` y extent in km.
        hres: Positive grid spacing in km.
    """

    def __init__(
        self,
        start_time,
        end_time,
        cadence_s: float,
        xlim: tuple[float, float] = (-1500.0, 1500.0),
        ylim: tuple[float, float] = (-1500.0, 1500.0),
        hres: float = 20.0,
    ) -> None:
        """Configure the master time axis and grid extents.

        Args:
            start_time: Requested start timestamp; both endpoints are
                included in the raw epochs.
            end_time: Requested end timestamp; must be aligned to the
                cadence from ``start_time``.
            cadence_s: Positive finite sampling cadence in seconds.
            xlim: ``(min, max)`` x extent in km.
            ylim: ``(min, max)`` y extent in km.
            hres: Positive grid spacing in km.

        Raises:
            ValueError: If the cadence or resolution is not positive/finite
                or the end time is not cadence-aligned.
        """
        _validate_positive(cadence_s, "cadence_s")
        _validate_positive(hres, "hres")
        self.start_time = np.datetime64(pd.Timestamp(start_time).to_pydatetime())
        self.end_time = np.datetime64(pd.Timestamp(end_time).to_pydatetime())
        self.cadence_s = float(cadence_s)
        self.xlim = tuple(xlim)
        self.ylim = tuple(ylim)
        self.hres = float(hres)
        span = (self.end_time - self.start_time) / np.timedelta64(1, "s")
        if span < 0 or not np.isclose(span % self.cadence_s, 0):
            raise ValueError(
                "end_time must be aligned to cadence_s from start_time; got a "
                f"span of {span} s with cadence {self.cadence_s} s"
            )

    def sample(self, reference_time: np.datetime64) -> GridSamples:
        """Build the labeled grid coordinate arrays.

        Args:
            reference_time: Scenario epoch; ``t_seconds`` are computed
                relative to it.

        Returns:
            :class:`GridSamples` with 1-D ``x`` (dim ``x``, km), ``y``
            (dim ``y``, km), ``t_seconds`` (dim ``time``, seconds from
            ``reference_time``), and ``time`` (dim ``time``, actual
            datetime timestamps). Dask-backed for broadcasting.
        """
        reference_time = np.datetime64(reference_time)
        time = pd.date_range(
            pd.Timestamp(self.start_time),
            pd.Timestamp(self.end_time),
            freq=pd.Timedelta(seconds=self.cadence_s),
        )
        t_seconds = (time - pd.Timestamp(reference_time)).total_seconds().values
        x = np.arange(self.xlim[0], self.xlim[1] + self.hres, self.hres)
        y = np.arange(self.ylim[0], self.ylim[1] + self.hres, self.hres)
        return GridSamples(
            x=xr.DataArray(da.from_array(x, chunks=-1), coords=[("x", x)]),
            y=xr.DataArray(da.from_array(y, chunks=-1), coords=[("y", y)]),
            t_seconds=xr.DataArray(
                da.from_array(t_seconds, chunks=-1), coords=[("time", time)]
            ),
            time=xr.DataArray(time, coords=[("time", time)]),
        )


# ---------------------------------------------------------------------------
# Shared epoch-level helpers (GNSS samplers)
# ---------------------------------------------------------------------------


def master_epochs(
    start_time,
    end_time,
    cadence_s: float,
) -> np.ndarray:
    """Build the master requested raw-epoch axis before filtering.

    Args:
        start_time: Requested start timestamp (UTC, timezone-naive); both
            endpoints are included.
        end_time: Requested end timestamp; must be aligned to the cadence.
        cadence_s: Positive finite raw sampling cadence in seconds.

    Returns:
        1-D array of ``np.datetime64`` epochs from ``start_time`` to
        ``end_time`` inclusive at ``cadence_s`` spacing.

    Raises:
        ValueError: If the cadence is not positive/finite or the end time
            is not cadence-aligned.
    """
    _validate_positive(cadence_s, "cadence_s")
    start = np.datetime64(pd.Timestamp(start_time).to_pydatetime())
    end = np.datetime64(pd.Timestamp(end_time).to_pydatetime())
    span = (end - start) / np.timedelta64(1, "s")
    if span < 0 or not np.isclose(span % cadence_s, 0):
        raise ValueError(
            "end_time must be aligned to cadence_s from start_time; got a "
            f"span of {span} s with cadence {cadence_s} s"
        )
    return start + np.arange(int(span / cadence_s) + 1) * np.timedelta64(
        round(cadence_s * 1e6), "us"
    )


def join_receiver_metadata(
    obs: pd.DataFrame,
    rx: pd.DataFrame,
    tol: float = 1e-6,
) -> pd.DataFrame:
    """Join canonical receiver metadata onto observations with validation.

    Deduplicates identical receiver records within ``tol`` and validates a
    many-to-one join; missing or conflicting receiver positions raise.

    Args:
        obs: Observation DataFrame with a ``rx`` column (one row per
            original observation).
        rx: Receiver metadata, unique by receiver identity, with columns
            ``rx``, ``lat`` (deg), ``lon`` (deg), ``alt`` (km), and ECEF
            coordinates ``x_ecef``/``y_ecef``/``z_ecef`` (km) when present.
        tol: Numerical tolerance for deduplicating identical receiver
            records.

    Returns:
        Observation DataFrame with receiver metadata columns joined
        one-to-one (row count unchanged).

    Raises:
        ValueError: If receiver records conflict beyond ``tol``, receiver
            positions are missing, or the join would expand observations.
    """
    if "rx" not in obs.columns or "rx" not in rx.columns:
        raise ValueError("obs and rx must both contain a 'rx' column")
    geo_cols = [c for c in ("lat", "lon", "alt") if c in rx.columns]
    if not geo_cols:
        raise ValueError("rx must contain receiver position columns")
    rx_unique = rx.drop_duplicates(subset=["rx"])
    if len(rx_unique) != rx["rx"].nunique():
        raise ValueError("rx contains conflicting duplicate receiver records")
    # verify duplicate records agree within tol
    for name, group in rx.groupby("rx"):
        for col in geo_cols:
            values = group[col].to_numpy(dtype=float)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"receiver {name} has nonfinite {col}")
            if np.ptp(values) > tol:
                raise ValueError(
                    f"receiver {name} has conflicting {col} records beyond "
                    f"tolerance {tol}"
                )
    merged = obs.merge(rx_unique, on="rx", how="left", validate="many_to_one")
    if len(merged) != len(obs):
        raise ValueError(
            "receiver join expanded observations; receiver metadata must be unique by rx"
        )
    for col in geo_cols:
        if merged[col].isna().any():
            missing = merged.loc[merged[col].isna(), "rx"].unique()
            raise ValueError(f"missing receiver positions for: {list(missing)}")
    return merged


def assign_master_slots(
    times: np.ndarray,
    epochs: np.ndarray,
    tolerance_s: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign observation timestamps to master slots.

    Args:
        times: Observation timestamps (``np.datetime64`` array).
        epochs: Master requested raw-epoch axis.
        tolerance_s: Optional nearest-slot tolerance in seconds; must be
            less than half the raw cadence. 0 enforces exact matching.

    Returns:
        Tuple ``(slot_index, matched)`` where ``slot_index`` gives the
        assigned master slot per row (-1 for unmatched) and ``matched`` is
        the boolean mask of rows assigned to a slot.

    Raises:
        ValueError: If the tolerance is not less than half the raw cadence.
    """
    cadence = (
        (epochs[1] - epochs[0]) / np.timedelta64(1, "s") if len(epochs) > 1 else np.inf
    )
    if tolerance_s >= cadence / 2:
        raise ValueError(
            "timestamp tolerance must be less than half the raw cadence; "
            f"got {tolerance_s} s with cadence {cadence} s"
        )
    slot_index = np.full(len(times), -1, dtype=int)
    if tolerance_s == 0:
        # exact timestamp matching against the master axis
        pos = np.searchsorted(epochs, times)
        pos = np.clip(pos, 0, len(epochs) - 1)
        exact = epochs[pos] == times
        slot_index[exact] = pos[exact]
    else:
        pos = np.searchsorted(epochs, times)
        pos = np.clip(pos, 1, len(epochs) - 1)
        left = np.abs((times - epochs[pos - 1]) / np.timedelta64(1, "s"))
        right = np.abs((epochs[pos] - times) / np.timedelta64(1, "s"))
        nearest = np.where(left <= right, pos - 1, pos)
        dist = np.minimum(left, right)
        near = dist <= tolerance_s
        slot_index[near] = nearest[near]
    matched = slot_index >= 0
    return slot_index, matched


def build_master_windows(
    epochs: np.ndarray,
    window: int,
    step: int,
    slot_index: np.ndarray,
    obs_id: np.ndarray,
    drop_incomplete: bool = True,
) -> list:
    """Build positional ``TimeWindow`` objects on the master epoch axis.

    A window's logical start/end and membership do not move when an epoch
    has no rows; a final incomplete window is dropped by default, not
    internal empty slots. Membership is built only after final filtering,
    using the fixed schedule and retained ``obs_id`` mapping.

    Args:
        epochs: Master requested raw-epoch axis.
        window: Positive count of master slots per window.
        step: Positive count of master slots between window starts.
        slot_index: Assigned master slot per observation (-1 unmatched).
        obs_id: Stable observation IDs aligned to ``slot_index``.
        drop_incomplete: Whether to drop a final incomplete window.

    Returns:
        List of :class:`gnss_tid.pointdata.TimeWindow` with logical
        start/end times on the master axis and ``row_indices`` into the
        retained observation rows. Empty windows keep empty ``row_indices``.

    Raises:
        ValueError: If ``window`` or ``step`` is not a positive count.
    """
    from .pointdata import TimeWindow

    if window <= 0 or step <= 0:
        raise ValueError("window and step must be positive counts of master slots")
    order = np.argsort(slot_index, kind="stable")
    sorted_slots = slot_index[order]
    windows = []
    start = 0
    n_epochs = len(epochs)
    while start < n_epochs:
        stop = start + window
        if stop > n_epochs:
            if not drop_incomplete:
                stop = n_epochs
                windows.append((start, stop))
            break
        windows.append((start, stop))
        start += step
    result = []
    for w_start, w_stop in windows:
        lo, hi = np.searchsorted(sorted_slots, [w_start, w_stop])
        rows = order[lo:hi]
        result.append(
            TimeWindow(
                epochs[w_start],
                epochs[min(w_stop, n_epochs) - 1],
                rows,
            )
        )
    return result


# ---------------------------------------------------------------------------
# RealGNSSSampler
# ---------------------------------------------------------------------------


class RealGNSSSampler:
    """Epoch-level GNSS sampler from loaded real observations.

    Works from the original epoch-level observations, before
    ``pointdata.get_data()``'s receiver/satellite window aggregation. Every
    retained geometry row receives a simulated measurement (geometry-only
    availability by default); an opt-in observed-availability mode excludes
    rows where a selected loaded TEC column is nonfinite.

    Attributes:
        obs: Epoch-level observation DataFrame from
            ``pointdata.load_observations()``.
        rx: Receiver metadata DataFrame (unique by receiver identity).
        proj: Shared :class:`~gnss_tid.coords.Local2D` projection frame.
        true_height_km: True thin-shell height in km for IPP projection.
        el_min: Minimum elevation filter in degrees.
        start_time: Requested schedule start (UTC, timezone-naive).
        end_time: Requested schedule end, aligned to the cadence.
        cadence_s: Positive raw sampling cadence in seconds.
        availability: ``"geometry"`` or ``"observed"`` availability policy.
        tec_column: Selected TEC column for observed availability.
        timestamp_tolerance_s: Optional nearest-slot tolerance in seconds.
    """

    def __init__(
        self,
        obs: pd.DataFrame,
        rx: pd.DataFrame,
        proj,
        true_height_km: float,
        start_time,
        end_time,
        cadence_s: float,
        el_min: float = 0.0,
        availability: str = "geometry",
        tec_column: str | None = None,
        timestamp_tolerance_s: float = 0.0,
    ) -> None:
        """Store observation tables and sampling settings.

        Args:
            obs: Epoch-level observation DataFrame with ``time``, ``rx``,
                ``sv``, ``az``, ``el`` columns (from
                ``pointdata.load_observations()``).
            rx: Receiver metadata with ``rx``, ``lat``, ``lon``, ``alt``.
            proj: Shared :class:`~gnss_tid.coords.Local2D` projection.
            true_height_km: True thin-shell height in km.
            start_time: Requested schedule start (both endpoints included).
            end_time: Requested schedule end, cadence-aligned.
            cadence_s: Positive raw sampling cadence in seconds.
            el_min: Minimum elevation filter in degrees.
            availability: ``"geometry"`` gives every retained geometry row a
                simulated measurement; ``"observed"`` additionally excludes
                rows where ``tec_column`` is nonfinite after loader
                filtering.
            tec_column: Selected TEC column for observed availability
                (required when ``availability == "observed"``).
            timestamp_tolerance_s: Optional nearest-slot tolerance in
                seconds; must be less than half the raw cadence.

        Raises:
            ValueError: If required columns are missing, availability
                settings are inconsistent, or the cadence is not positive.
        """
        for col in ("time", "rx", "sv", "az", "el"):
            if col not in obs.columns:
                raise ValueError(f"obs is missing required column '{col}'")
        if availability not in ("geometry", "observed"):
            raise ValueError("availability must be 'geometry' or 'observed'")
        if availability == "observed" and not tec_column:
            raise ValueError("observed availability requires a tec_column")
        if availability == "observed" and tec_column not in obs.columns:
            raise ValueError(f"tec_column '{tec_column}' not in obs")
        _validate_positive(cadence_s, "cadence_s")
        self.obs = obs.reset_index(drop=True).copy()
        self.obs["obs_id"] = np.arange(len(self.obs))
        self.rx = rx
        self.proj = proj
        self.true_height_km = float(true_height_km)
        self.el_min = float(el_min)
        self.start_time = np.datetime64(pd.Timestamp(start_time).to_pydatetime())
        self.end_time = np.datetime64(pd.Timestamp(end_time).to_pydatetime())
        self.cadence_s = float(cadence_s)
        self.availability = str(availability)
        self.tec_column = tec_column
        self.timestamp_tolerance_s = float(timestamp_tolerance_s)
        self.unmatched_rows = 0

    def sample(self, reference_time: np.datetime64) -> GNSSSamples:
        """Build epoch-level GNSS sample geometry.

        Assigns observations to master slots, applies elevation and
        availability filtering, projects retained rows to IPPs with
        ``pointdata.project_to_ipp()`` and
        ``pointdata.convert_to_local_coords()``, and computes ``t_seconds``
        from each retained row's actual timestamp relative to the scenario
        epoch.

        Args:
            reference_time: Scenario epoch; ``t_seconds`` are relative to it.

        Returns:
            :class:`GNSSSamples` with one row per retained original
            observation: ``obs_id``, ``time``, ``rx``, ``sv``, ``az``,
            ``el``, ``x`` (km), ``y`` (km), ``t_seconds``, and receiver
            metadata. Evaluating the analytical field yields one dTEC value
            per retained observation.

        Raises:
            ValueError: If the timestamp tolerance exceeds half the cadence.
        """
        reference_time = np.datetime64(reference_time)
        epochs = master_epochs(self.start_time, self.end_time, self.cadence_s)
        times = pd.to_datetime(self.obs["time"]).to_numpy()
        slot_index, matched = assign_master_slots(
            times, epochs, self.timestamp_tolerance_s
        )
        self.unmatched_rows = int((~matched).sum())

        obs = self.obs.loc[matched].copy()
        slots = slot_index[matched]

        # geometry filtering: elevation cutoff, consistent with loader criteria
        valid = obs["el"].to_numpy() >= self.el_min
        valid &= np.isfinite(obs["az"].to_numpy())
        valid &= np.isfinite(obs["el"].to_numpy())
        # observed availability: finite mask of the selected TEC column
        if self.availability == "observed":
            valid &= np.isfinite(obs[self.tec_column].to_numpy())
        obs = obs.loc[valid].reset_index(drop=True)
        slots = slots[valid]

        # join canonical receiver metadata with many-to-one validation
        obs = join_receiver_metadata(obs, self.rx)

        # project to IPP at the true shell height using the existing helpers
        from .pointdata import convert_to_local_coords, project_to_ipp

        rx_positions = obs[["lat", "lon", "alt"]].to_numpy()
        obs = project_to_ipp(obs, rx_positions, self.true_height_km)
        obs = convert_to_local_coords(obs, self.proj)

        t_seconds = (
            (pd.to_datetime(obs["time"]) - pd.Timestamp(reference_time))
            .dt.total_seconds()
            .to_numpy()
        )
        obs = obs.assign(t_seconds=t_seconds)

        retained = np.where(valid)[0]
        obs = obs.assign(obs_id=self.obs["obs_id"].to_numpy()[matched][retained])
        assert len(obs) == len(np.unique(obs["obs_id"]))
        return GNSSSamples(
            obs=obs,
            rx=self.rx.drop_duplicates(subset=["rx"]).reset_index(drop=True),
            proj=self.proj,
            true_height_km=self.true_height_km,
        )


# ---------------------------------------------------------------------------
# OrbitGNSSSampler
# ---------------------------------------------------------------------------


def ecef_to_geodetic(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert ECEF coordinates (km) to geodetic latitude/longitude/altitude.

    Uses ``pymap3d.ecef2geodetic`` (validated round trips) rather than the
    legacy spherical conversion, which returns spherical latitude and
    radius-derived altitude and treats them as geodetic.

    Args:
        x: ECEF x in km (scalar or array).
        y: ECEF y in km.
        z: ECEF z in km.

    Returns:
        Tuple ``(lat, lon, alt_km)`` of geodetic coordinates in degrees and
        kilometers.

    Raises:
        ValueError: If the round trip fails beyond tolerance.
    """
    lat, lon, alt_m = pymap3d.ecef2geodetic(
        np.asarray(x, dtype=float) * 1000.0,
        np.asarray(y, dtype=float) * 1000.0,
        np.asarray(z, dtype=float) * 1000.0,
    )
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    alt_km = np.asarray(alt_m, dtype=float) / 1000.0
    # validate the ECEF-to-geodetic round trip
    check_x, check_y, check_z = pymap3d.geodetic2ecef(lat, lon, alt_m)
    err = np.max(
        np.abs(
            np.stack(
                [
                    np.asarray(check_x) - np.asarray(x, dtype=float) * 1000.0,
                    np.asarray(check_y) - np.asarray(y, dtype=float) * 1000.0,
                    np.asarray(check_z) - np.asarray(z, dtype=float) * 1000.0,
                ]
            )
        )
    )
    if err > 1e-3:  # sub-millimeter tolerance for the float64 round trip
        raise ValueError(f"ECEF-to-geodetic round trip failed: max error {err} m")
    return lat, lon, alt_km


def build_rx_metadata_from_ecef(
    receivers: np.ndarray,
    rx_names: list[str] | None = None,
) -> pd.DataFrame:
    """Derive canonical geodetic receiver metadata from ECEF coordinates.

    Args:
        receivers: Shape ``(N, 3)`` ECEF coordinates of receivers in km.
        rx_names: Optional receiver names; defaults to ``rx0..rxN-1``.

    Returns:
        Receiver metadata DataFrame with one record per ``rx``: geodetic
        ``lat``/``lon`` (degrees), ellipsoidal ``alt`` normalized to km,
        and ECEF coordinates ``x_ecef``/``y_ecef``/``z_ecef`` in km.

    Raises:
        ValueError: If the ECEF-to-geodetic round trip fails or receivers
            are empty/nonfinite.
    """
    receivers = np.asarray(receivers, dtype=float)
    if receivers.ndim != 2 or receivers.shape[1] != 3 or not len(receivers):
        raise ValueError("receivers must be a nonempty (N, 3) ECEF array in km")
    if not np.all(np.isfinite(receivers)):
        raise ValueError("receivers must be finite ECEF coordinates in km")
    if rx_names is None:
        rx_names = [f"rx{i}" for i in range(len(receivers))]
    if len(rx_names) != len(receivers):
        raise ValueError("rx_names length must match the receiver count")
    lat, lon, alt_km = ecef_to_geodetic(receivers[:, 0], receivers[:, 1], receivers[:, 2])
    return pd.DataFrame(
        {
            "rx": rx_names,
            "lat": lat,
            "lon": lon,
            "alt": alt_km,
            "x_ecef": receivers[:, 0],
            "y_ecef": receivers[:, 1],
            "z_ecef": receivers[:, 2],
        }
    )


def ray_aer(
    receivers: np.ndarray,
    satellites: np.ndarray,
    rx_metadata: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute azimuth/elevation from receiver-to-satellite rays.

    Uses the canonical geodetic receiver metadata (derived from the exact
    ECEF coordinates) as the observer so the rays and metadata agree.

    Args:
        receivers: Shape ``(N, 3)`` receiver ECEF coordinates in km.
        satellites: Shape ``(N, 3)`` satellite ECEF coordinates in km.
        rx_metadata: Canonical receiver metadata from
            :func:`build_rx_metadata_from_ecef` (one record per receiver).

    Returns:
        Tuple ``(az, el)`` of azimuth/elevation in degrees.

    Raises:
        ValueError: If ray agreement checks fail.
    """
    receivers = np.asarray(receivers, dtype=float)
    satellites = np.asarray(satellites, dtype=float)
    if receivers.shape != satellites.shape:
        raise ValueError("receivers and satellites must have matching shapes")
    if not np.all(np.isfinite(receivers)) or not np.all(np.isfinite(satellites)):
        raise ValueError("receivers and satellites must be finite ECEF coordinates")
    # map each receiver row to its canonical metadata record by name; names
    # must repeat in the same order as unique metadata records
    lookup = rx_metadata.set_index("rx")
    unique_names = list(rx_metadata["rx"])
    lats = lookup.loc[unique_names, "lat"].to_numpy(dtype=float)
    lons = lookup.loc[unique_names, "lon"].to_numpy(dtype=float)
    alts = lookup.loc[unique_names, "alt"].to_numpy(dtype=float)
    idx = {name: i for i, name in enumerate(unique_names)}
    n_rx = len(unique_names)
    names = np.repeat(unique_names, len(receivers) // n_rx)
    if len(names) != len(receivers):
        raise ValueError(
            "receiver count mismatch: metadata has "
            f"{n_rx} records but {len(receivers)} rows were given"
        )
    az = np.empty(len(receivers))
    el = np.empty(len(receivers))
    for i, (_r, s, name) in enumerate(zip(receivers, satellites, names, strict=True)):
        j = idx[name]
        a, e, _ = pymap3d.ecef2aer(
            s[0] * 1000.0,
            s[1] * 1000.0,
            s[2] * 1000.0,
            lats[j],
            lons[j],
            alts[j] * 1000.0,
        )
        az[i] = a
        el[i] = e
    # validate ray agreement: azimuth/elevation must round-trip the ray
    for i in range(len(receivers)):
        j = idx[names[i]]
        check_x, check_y, check_z = pymap3d.aer2ecef(
            az[i],
            el[i],
            1000.0,
            lats[j],
            lons[j],
            alts[j] * 1000.0,
        )
        direction = (satellites[i] - receivers[i]) / np.linalg.norm(
            satellites[i] - receivers[i]
        )
        point = np.array([check_x, check_y, check_z]) / 1000.0
        if not np.all(np.isfinite(point)):
            raise ValueError("ray agreement check produced nonfinite coordinates")
        # the 1-km look-direction point must lie along the receiver-to-satellite ray
        off_ray = np.linalg.norm(np.cross(point - receivers[i], direction))
        if off_ray > 1e-6:
            raise ValueError(
                f"ray agreement check failed at receiver {names[i]}: "
                f"off-ray distance {off_ray} km"
            )
    return az, el


class OrbitGNSSSampler:
    """Geometry-only GNSS sampler from simulated orbit geometry.

    Reuses orbit and receiver-distribution utilities in
    :mod:`gnss_tid.simulation` to produce geometry-only GNSS rows in the
    same shape as :class:`RealGNSSSampler`. Receiver metadata is derived
    from the exact ECEF coordinates with validated ECEF-to-geodetic round
    trips; azimuth/elevation are computed from that same observer. Random
    geometry generation is reproducible from a seed.

    Attributes:
        n_rx: Number of receivers.
        n_sat: Number of satellites.
        lat_range: Receiver latitude range in degrees.
        lon_range: Receiver longitude range in degrees.
        el_min: Minimum elevation filter in degrees.
        seed: Geometry-generation seed (separate from parameter and noise
            streams).
    """

    def __init__(
        self,
        n_rx: int = 20,
        n_sat: int = 8,
        lat_range: tuple[float, float] = (35.0, 45.0),
        lon_range: tuple[float, float] = (-110.0, -100.0),
        el_min: float = 15.0,
        seed: int = 0,
        receivers: np.ndarray | None = None,
        satellites: np.ndarray | None = None,
        start_time=None,
        end_time=None,
        cadence_s: float = 60.0,
    ) -> None:
        """Configure the orbit sampler geometry.

        Args:
            n_rx: Number of random receivers (ignored when ``receivers`` is
                supplied).
            n_sat: Number of satellites at typical GPS orbital altitude
                (ignored when ``satellites`` is supplied).
            lat_range: Receiver latitude range in degrees.
            lon_range: Receiver longitude range in degrees.
            el_min: Minimum elevation filter in degrees.
            seed: Geometry-generation seed for reproducibility.
            receivers: Optional explicit ``(N, 3)`` receiver ECEF array in
                km; deterministic geometry without random generation.
            satellites: Optional explicit ``(N, 3)`` satellite ECEF array in
                km; static geometry without orbital motion.
            start_time: Requested schedule start; both endpoints included.
            end_time: Requested schedule end, cadence-aligned.
            cadence_s: Positive raw sampling cadence in seconds.
        """
        self.n_rx = int(n_rx)
        self.n_sat = int(n_sat)
        self.lat_range = tuple(lat_range)
        self.lon_range = tuple(lon_range)
        self.el_min = float(el_min)
        self.seed = int(seed)
        self.cadence_s = float(cadence_s)
        self.start_time = start_time
        self.end_time = end_time
        self._receivers = (
            None if receivers is None else np.asarray(receivers, dtype=float)
        )
        self._satellites = (
            None if satellites is None else np.asarray(satellites, dtype=float)
        )

    def sample(self, reference_time: np.datetime64, proj) -> GNSSSamples:
        """Build geometry-only GNSS sample geometry over the requested time axis.

        Generates (or takes preconfigured) receiver ECEF coordinates and
        satellite positions at each requested epoch (deterministic orbital
        motion from fixed elements when ``satellites`` is not supplied),
        derives canonical geodetic receiver metadata from those exact
        coordinates, computes azimuth/elevation from that same observer,
        filters below-elevation rays, and projects retained rays to IPPs
        with the same helpers as :class:`RealGNSSSampler`.

        Args:
            reference_time: Scenario epoch (UTC, timezone-naive);
                ``t_seconds`` are relative to it.
            proj: Shared :class:`~gnss_tid.coords.Local2D` projection.

        Returns:
            :class:`GNSSSamples` with geometry-only rows in the same shape
            as :class:`RealGNSSSampler` output: ``obs_id``, ``time``, ``rx``,
            ``sv``, ``az``, ``el``, ``x`` (km), ``y`` (km), ``t_seconds``,
            and receiver metadata.

        Raises:
            ValueError: If geometry generation or ray validation fails.
        """
        rng = np.random.default_rng(self.seed)
        if self._receivers is not None:
            receivers = self._receivers
        else:
            receivers = self._generate_receivers(rng)
        if self._satellites is not None:
            satellite_fn = None
            satellites0 = self._satellites
        else:
            satellite_fn = self._satellite_motion(rng)
            satellites0 = satellite_fn(0.0)

        rx_metadata = build_rx_metadata_from_ecef(receivers)
        names = [f"rx{i}" for i in range(len(receivers))]
        sv_names = [f"SV{j}" for j in range(len(satellites0))]

        if self.start_time is None or self.end_time is None:
            # no requested schedule: single-epoch geometry at the scenario epoch
            epochs = np.array([np.datetime64(reference_time)])
            t_offsets = np.array([0.0])
        else:
            epochs = master_epochs(self.start_time, self.end_time, self.cadence_s)
            t_offsets = (epochs - reference_time) / np.timedelta64(1, "s")
        frames = []
        obs_id = 0
        for t_seconds in t_offsets:
            satellites = (
                satellites0 if satellite_fn is None else satellite_fn(float(t_seconds))
            )
            rx_flat = np.repeat(receivers, len(satellites), axis=0)
            sat_flat = np.tile(satellites, (len(receivers), 1))
            az, el = ray_aer(rx_flat, sat_flat, rx_metadata)
            valid = (el >= self.el_min) & np.isfinite(az) & np.isfinite(el)
            if not valid.any():
                continue
            frame = pd.DataFrame(
                {
                    "time": np.full(
                        int(valid.sum()),
                        np.datetime64(reference_time)
                        + np.timedelta64(round(float(t_seconds) * 1e6), "us"),
                    ),
                    "rx": np.repeat(names, len(satellites))[valid],
                    "sv": np.tile(sv_names, len(receivers))[valid],
                    "az": az[valid],
                    "el": el[valid],
                    "t_seconds": np.full(int(valid.sum()), float(t_seconds)),
                }
            )
            frame["obs_id"] = np.arange(obs_id, obs_id + len(frame))
            obs_id += len(frame)
            frames.append(frame)
        if not frames:
            raise ValueError("no rays above the configured minimum elevation")
        obs = pd.concat(frames, ignore_index=True)

        from .pointdata import convert_to_local_coords, project_to_ipp

        # canonical receiver metadata (geodetic from the exact ECEF coords),
        # retaining original receiver ECEF coordinates in km
        rx_table = rx_metadata
        obs = join_receiver_metadata(obs, rx_table)
        obs = project_to_ipp(obs, obs[["lat", "lon", "alt"]].to_numpy(), 350.0)
        obs = convert_to_local_coords(obs, proj)
        assert len(obs) == len(np.unique(obs["obs_id"]))
        return GNSSSamples(
            obs=obs,
            rx=rx_table,
            proj=proj,
            true_height_km=350.0,
        )

    def _satellite_motion(self, rng: np.random.Generator):
        """Build a deterministic satellite motion function from fixed elements.

        Uses circular orbits at a common radius with fixed inclination and
        phase per satellite, giving a reproducible ephemeris-free geometry
        that changes with time.

        Args:
            rng: Geometry random generator.

        Returns:
            Callable ``f(t_seconds) -> (N, 3)`` satellite ECEF coordinates
            in km.
        """
        radius = 26560.0
        inclination = np.radians(55.0)
        # use a faster-than-realistic orbital period so short smoke windows
        # see changing geometry
        omega = 2 * np.pi / 3600.0  # 1-hour orbital period
        phases = rng.uniform(0, 2 * np.pi, self.n_sat)
        nodes = rng.uniform(0, 2 * np.pi, self.n_sat)
        cos_i, sin_i = np.cos(inclination), np.sin(inclination)

        def position(t_seconds: float) -> np.ndarray:
            """Return satellite ECEF positions at a time offset.

            Args:
                t_seconds: Time in seconds relative to the scenario epoch.

            Returns:
                Shape ``(N, 3)`` satellite ECEF coordinates in km.
            """
            out = np.empty((len(phases), 3))
            for i, (phase, node) in enumerate(zip(phases, nodes, strict=True)):
                angle = phase + omega * t_seconds
                x_orb = radius * np.cos(angle)
                y_orb = radius * np.sin(angle) * cos_i
                z_orb = radius * np.sin(angle) * sin_i
                c, s = np.cos(node), np.sin(node)
                out[i] = [x_orb * c - y_orb * s, x_orb * s + y_orb * c, z_orb]
            return out

        return position

    def _generate_receivers(self, rng: np.random.Generator) -> np.ndarray:
        """Generate random receiver locations uniform over the range.

        Args:
            rng: Geometry random generator.

        Returns:
            Shape ``(N, 3)`` receiver ECEF coordinates in km.
        """
        from .simulation import SimulationScenario

        SimulationScenario(None)
        lons = rng.uniform(self.lon_range[0], self.lon_range[1], self.n_rx)
        sin_lats = rng.uniform(
            np.sin(np.radians(self.lat_range[0])),
            np.sin(np.radians(self.lat_range[1])),
            self.n_rx,
        )
        lats = np.degrees(np.arcsin(sin_lats))
        from .coords import spherical2ecef

        return spherical2ecef(lats, lons, np.full(self.n_rx, 6378.137))

    def _generate_satellites(self, rng: np.random.Generator) -> np.ndarray:
        """Generate satellite positions at typical GPS orbital altitude.

        Args:
            rng: Geometry random generator.

        Returns:
            Shape ``(N, 3)`` satellite ECEF coordinates in km.
        """
        radius = 26560.0
        phi = rng.uniform(0, 2 * np.pi, self.n_sat)
        cos_theta = rng.uniform(-1, 1, self.n_sat)
        sin_theta = np.sqrt(1 - cos_theta**2)
        return np.stack(
            [
                radius * sin_theta * np.cos(phi),
                radius * sin_theta * np.sin(phi),
                radius * cos_theta,
            ],
            axis=1,
        )


# ---------------------------------------------------------------------------
# Forward model
# ---------------------------------------------------------------------------


@runtime_checkable
class ForwardModel(Protocol):
    """Forward model evaluating a physical scene at sample geometry."""

    def measure(self, samples) -> xr.DataArray:
        """Return noise-free dTEC in TECU at the given samples."""
        ...


class AnalyticalTECForwardModel:
    """Analytical dTEC forward model (shell-equivalent, no slant mapping).

    Initialized with a :class:`Scenario` and the experiment's global labeled
    trial coordinate. ``measure`` evaluates the scenario at the sampler's
    coordinates and returns noise-free dTEC in TECU:

    - grid samples give ``(trial, time, y, x)``;
    - GNSS samples give ``(trial, obs)`` exactly aligned to
      :attr:`GNSSSamples.obs` rows.

    Constant source parameters are broadcast explicitly along ``trial`` so
    no source silently loses the dimension; an empty source list produces a
    zero field with the complete requested sample and trial dimensions.
    """

    def __init__(self, scenario: Scenario, trials) -> None:
        """Store the scenario and global trial coordinate.

        Args:
            scenario: Analytical-wave scenario to evaluate.
            trials: Global labeled trial coordinate (values or xarray
                coordinate array); the experiment owns it, initially
                ``0..n_trials-1``.
        """
        self.scenario = scenario
        self.trials = np.asarray(trials)

    def measure(self, samples) -> xr.DataArray:
        """Evaluate the scenario at the sampler's coordinates.

        Args:
            samples: :class:`GridSamples` or :class:`GNSSSamples` produced by
                a sampler.

        Returns:
            Noise-free dTEC in TECU as an xarray DataArray with dims
            ``(trial, time, y, x)`` for grid samples or ``(trial, obs)`` for
            GNSS samples, aligned to ``self.trials`` and the sample rows.

        Raises:
            TypeError: If the sample type is not supported.
        """
        if isinstance(samples, GridSamples):
            return self._measure_grid(samples)
        if isinstance(samples, GNSSSamples):
            return self._measure_gnss(samples)
        raise TypeError(f"unsupported sample type: {type(samples).__name__}")

    def _trial_coord(self) -> xr.DataArray:
        """Return the global trial coordinate as an xarray DataArray.

        Returns:
            1-D DataArray with dim ``trial`` holding ``self.trials``.
        """
        return xr.DataArray(self.trials, coords=[("trial", self.trials)])

    def _measure_grid(self, samples: GridSamples) -> xr.DataArray:
        """Evaluate the scenario on the uniform grid.

        Args:
            samples: Grid sample geometry.

        Returns:
            dTEC DataArray with dims ``(trial, time, y, x)``, lazily
            evaluated via Dask broadcasting (no Python loops over trials,
            pixels, or timestamps).
        """
        trial = self._trial_coord()
        field = self.scenario.evaluate(samples.x, samples.y, samples.t_seconds)
        if not isinstance(field, xr.DataArray):
            field = xr.DataArray(field, dims=("trial", "time", "y", "x"))
        if "trial" not in field.dims:
            field = field.expand_dims(trial=trial.values)
        field = field.transpose("trial", "time", "y", "x").assign_coords(
            trial=trial,
            x=samples.x,
            y=samples.y,
            t_seconds=samples.t_seconds,
        )
        if not dask.is_dask_collection(field.data):
            field = field.chunk({})
        return field

    def _measure_gnss(self, samples: GNSSSamples) -> xr.DataArray:
        """Evaluate the scenario at epoch-level IPP positions.

        Args:
            samples: GNSS sample geometry with one row per observation.

        Returns:
            dTEC DataArray with dims ``(trial, obs)``, exactly aligned to
            ``samples.obs`` rows. Coordinates ``x``, ``y``, and
            ``t_seconds`` share the ``obs`` dimension (no square
            broadcast).
        """
        obs = samples.obs
        trial = self._trial_coord()
        x = xr.DataArray(obs["x"].to_numpy(), coords=[("obs", np.arange(len(obs)))])
        y = xr.DataArray(obs["y"].to_numpy(), coords=[("obs", np.arange(len(obs)))])
        t = xr.DataArray(
            obs["t_seconds"].to_numpy(), coords=[("obs", np.arange(len(obs)))]
        )
        field = self.scenario.evaluate(x, y, t)
        if not isinstance(field, xr.DataArray):
            field = xr.DataArray(field, dims=("trial", "obs"))
        if "trial" not in field.dims:
            field = field.expand_dims(trial=trial.values)
        field = field.transpose("trial", "obs").assign_coords(
            trial=trial,
            x=x,
            y=y,
            t_seconds=t,
        )
        if not dask.is_dask_collection(field.data):
            field = field.chunk({})
        return field


# ---------------------------------------------------------------------------
# Noise and dataset construction
# ---------------------------------------------------------------------------


def sample_parameters(
    scenario: Scenario,
    n_trials: int,
    seed: int,
    ranges: dict | None = None,
) -> Scenario:
    """Sample per-trial source parameters from [min, max] ranges.

    Builds a new :class:`Scenario` whose scalar source parameters are
    replaced with trial-indexed xarray DataArrays drawn from uniform
    distributions over the requested ranges. Parameters not listed in
    ``ranges`` keep their constant values.

    Args:
        scenario: Template scenario with constant parameters.
        n_trials: Number of trials to sample.
        seed: Seed for the parameter-sampling random stream (separate from
            geometry and noise streams).
        ranges: Mapping of ``"{source_index}.{field}"`` keys to
            ``[min, max]`` pairs, e.g. ``{"0.wavelength_km": [150, 400]}``.

    Returns:
        A new :class:`Scenario` with trial-indexed parameters where
        requested; the ``reference_time`` is unchanged.

    Raises:
        ValueError: If a range is not a two-element finite pair with
            ``min <= max`` or names an unknown source.
    """
    if ranges is None:
        ranges = {}
    rng = np.random.default_rng(seed)
    new_sources = []
    for i, source in enumerate(scenario.sources):
        source_ranges = {
            key.split(".", 1)[1]: value
            for key, value in ranges.items()
            if key.split(".", 1)[0] == str(i)
        }
        new_sources.append(_sample_source(source, source_ranges, n_trials, rng))
    unknown = [
        k for k in ranges if not 0 <= int(k.split(".", 1)[0]) < len(scenario.sources)
    ]
    if unknown:
        raise ValueError(f"ranges reference unknown sources: {unknown}")
    return Scenario(tuple(new_sources), scenario.reference_time)


def _sample_source(
    source: Source,
    ranges: dict,
    n_trials: int,
    rng: np.random.Generator,
) -> Source:
    """Sample one source's parameters from ranges.

    Args:
        source: Template source with constant parameters.
        ranges: Mapping of parameter name to ``[min, max]`` pairs for this
            source.
        n_trials: Number of trials.
        rng: Parameter-sampling random generator.

    Returns:
        A new :class:`Source` with requested parameters replaced by
        trial-indexed DataArrays.
    """
    wave = source.wave
    envelope = source.envelope
    if not ranges:
        return source
    wave_fields = {}
    env_fields = {}
    amp_range = ranges.get("amplitude_tecu")
    for name, value in ranges.items():
        if name == "amplitude_tecu":
            continue
        if hasattr(wave, name):
            wave_fields[name] = value
        elif envelope is not None and hasattr(envelope, name):
            env_fields[name] = value
        else:
            raise ValueError(f"unknown parameter '{name}' for source wave/envelope")

    def draw(pair):
        """Draw n_trials uniform values from a [min, max] range pair.

        Args:
            pair: Two-element ``(min, max)`` sequence.

        Returns:
            1-D array of ``n_trials`` uniform samples.

        Raises:
            ValueError: If the pair is invalid.
        """
        pair = list(pair)
        if len(pair) != 2 or not all(np.isfinite(pair)) or pair[0] > pair[1]:
            raise ValueError(f"invalid [min, max] range: {pair}")
        return rng.uniform(pair[0], pair[1], n_trials)

    coords = [("trial", np.arange(n_trials))]
    new_wave = dataclasses.replace(
        wave,
        **{
            name: xr.DataArray(draw(pair), coords=coords)
            for name, pair in wave_fields.items()
        },
    )
    new_envelope = (
        dataclasses.replace(
            envelope,
            **{
                name: xr.DataArray(draw(pair), coords=coords)
                for name, pair in env_fields.items()
            },
        )
        if envelope is not None
        else None
    )
    new_amplitude = (
        xr.DataArray(draw(amp_range), coords=coords)
        if amp_range
        else source.amplitude_tecu
    )
    return Source(new_wave, new_envelope, new_amplitude)


def noise_stream(
    shape: tuple,
    noise_std_tecu: float,
    seed: int,
    trial_chunks: int = 16,
) -> xr.DataArray:
    """Draw a lazily evaluated Gaussian noise array with a fixed RNG layout.

    The draws are made once against the full logical trial/sample axes with
    a fixed dask chunk layout independent of operational batching. Batches
    select from this array, so sample noise is identical when generation or
    estimation batch sizes or output chunks change and adjacent batches do
    not repeat draws.

    Args:
        shape: Full logical noise shape, e.g.
            ``(n_trials, n_time, n_y, n_x)``.
        noise_std_tecu: Absolute Gaussian standard deviation in TECU.
        seed: Seed for the measurement-noise random stream (separate from
            parameter-sampling and geometry streams).
        trial_chunks: Fixed trial chunk size for the RNG layout; independent
            of operational batching.

    Returns:
        Lazily evaluated noise DataArray with dims
        ``(trial, time, y, x)`` matching ``shape``.
    """
    rng = da.random.default_rng(seed)
    trial_chunks = min(trial_chunks, shape[0]) if shape else 1
    chunks = (trial_chunks, *tuple(shape[1:]))
    noise = rng.normal(scale=noise_std_tecu, size=shape, chunks=chunks)
    return xr.DataArray(noise, dims=("trial", "time", "y", "x")[: len(shape)])


def add_noise(
    field: xr.DataArray,
    noise_std_tecu: float | None,
    seed: int,
) -> xr.DataArray:
    """Add Gaussian measurement noise once after source summation.

    Noise is drawn against the full logical trial/sample axes with a fixed
    dask RNG layout independent of operational batching, then rechunked to
    match ``field``. For batched generation, draw the full-axis noise once
    with :func:`noise_stream` and select per batch instead of calling this
    per batch.

    Args:
        field: Noise-free dTEC in TECU (e.g. ``(trial, time, y, x)`` from
            :class:`AnalyticalTECForwardModel`).
        noise_std_tecu: Absolute Gaussian standard deviation in TECU, or
            ``None``/0 to return the field unchanged.
        seed: Seed for the measurement-noise random stream (separate from
            parameter-sampling and geometry streams).

    Returns:
        Noisy dTEC DataArray with the same dims, coords, and chunk layout
        as ``field``.
    """
    if not noise_std_tecu:
        return field
    shape = tuple(field.sizes[d] for d in ("trial", "time", "y", "x"))
    noise = noise_stream(shape, noise_std_tecu, seed)
    noise = noise.assign_coords(field.coords)
    if field.chunks:
        chunks = dict(zip(field.dims, field.chunks, strict=True))
        noise = noise.chunk(chunks)
    return field + noise


def build_grid_dataset(
    samples: GridSamples,
    field: xr.DataArray,
    scenario: Scenario,
    *,
    density_constant: int = 20,
    smooth_pixels: int = 0,
    output_dtype: str = "float32",
) -> xr.Dataset:
    """Build the canonical grid dataset directly from evaluated values.

    Converts the evaluated dTEC to ``image(trial, time, y, x)`` without
    interpolation. ``density`` is a constant compatibility field (a
    placeholder, not a real GNSS sample count). An optional rolling-mean
    spatial smoothing (the legacy 3x3 behavior at ``smooth_pixels=3``) is
    applied when requested; new physical-control experiments should leave
    it off.

    Args:
        samples: Grid sample geometry providing the coordinate arrays.
        field: Evaluated (noisy) dTEC with dims ``(trial, time, y, x)``.
        scenario: Scenario owning the generating parameters; its per-source
            truth fields are attached as coordinates indexed by
            ``(trial, source)``.
        density_constant: Constant density value (placeholder).
        smooth_pixels: Rolling-mean window side in pixels for optional
            spatial smoothing; 0 disables it.
        output_dtype: dtype for the persisted ``image`` variable.

    Returns:
        Dataset with ``image(trial, time, y, x)`` (dTEC in TECU, cast to
        ``output_dtype``), ``density(trial, time, y, x)`` (constant), and
        per-source truth coordinates (``lam``, ``tau``, ``dir``, ``snr``
        single-source aliases where applicable) on ``(trial, source)``.
        Attributes record the TEC convention and grid provenance.
    """
    image = field.astype(output_dtype)
    density = xr.ones_like(image) * density_constant
    data = xr.Dataset(
        {"image": image, "density": density},
        coords={
            "x": samples.x.values,
            "y": samples.y.values,
            "time": samples.time.values,
        },
    )
    data = _attach_truth(data, scenario)
    data["image"].attrs.update(
        {
            "units": "TECU",
            "long_name": "shell-equivalent dTEC, no slant mapping",
            "smoothing": f"rolling mean {smooth_pixels}x{smooth_pixels}"
            if smooth_pixels
            else "none",
        }
    )
    data["density"].attrs["comment"] = (
        "constant compatibility placeholder, not a real GNSS sample count"
    )
    if smooth_pixels:
        image_smooth = (
            data["image"]
            .rolling(x=smooth_pixels, y=smooth_pixels, center=True, min_periods=1)
            .mean()
        )
        data["image"] = image_smooth.astype(output_dtype)
    return data


def _attach_truth(data: xr.Dataset, scenario: Scenario) -> xr.Dataset:
    """Attach generating-parameter truth fields to a dataset.

    Args:
        data: Dataset to attach coordinates to.
        scenario: Scenario owning the generating parameters.

    Returns:
        Dataset with per-source truth coordinates on ``(trial, source)``:
        wave type, wavelength (km), period (minutes), phase (rad),
        direction (rad, planar), center (km), amplitude (TECU), envelope
        widths (km), speed ratio, initial ring radius (km), and spreading
        time (s). Single-source scenes also get ``lam``, ``tau``, ``dir``
        aliases on ``trial`` for legacy compatibility. Each variable's
        units are recorded in attrs.
    """
    if not scenario.sources:
        return data
    trials = data["trial"].values
    n_sources = len(scenario.sources)

    def to_trial_indexed(values):
        """Broadcast scalar values along trial, or select to trial labels.

        Args:
            values: Scalar or trial-indexed xarray/NumPy values.

        Returns:
            1-D array aligned to ``trials``.
        """
        if isinstance(values, xr.DataArray):
            return values.sel(trial=trials).values if "trial" in values.dims else values
        return np.full(len(trials), values, dtype=float)

    fields: dict[str, list] = {}
    for _i, source in enumerate(scenario.sources):
        wave = source.wave
        fields.setdefault("wave_type", []).append(type(wave).__name__)
        fields.setdefault("wavelength_km", []).append(
            to_trial_indexed(wave.wavelength_km)
        )
        fields.setdefault("period_s", []).append(to_trial_indexed(wave.period_s))
        fields.setdefault("phase_rad", []).append(to_trial_indexed(wave.phase_rad))
        fields.setdefault("center_x_km", []).append(to_trial_indexed(wave.center_km[0]))
        fields.setdefault("center_y_km", []).append(to_trial_indexed(wave.center_km[1]))
        if isinstance(wave, PlanarWave):
            fields.setdefault("direction_rad", []).append(
                to_trial_indexed(wave.direction_rad)
            )
        else:
            fields.setdefault("direction_rad", []).append([np.nan] * len(trials))
        env = source.envelope
        if env is not None:
            fields.setdefault("sigma_u_km", []).append(to_trial_indexed(env.sigma_u_km))
            fields.setdefault("sigma_v_km", []).append(
                to_trial_indexed(env.sigma_v_km)
                if env.sigma_v_km is not None
                else [np.nan] * len(trials)
            )
            fields.setdefault("speed_ratio", []).append(to_trial_indexed(env.speed_ratio))
            fields.setdefault("center_u_km", []).append(to_trial_indexed(env.center_u_km))
            fields.setdefault("spreading_time_s", []).append(
                to_trial_indexed(env.spreading_time_s)
                if env.spreading_time_s is not None
                else [np.nan] * len(trials)
            )
        else:
            for name in (
                "sigma_u_km",
                "sigma_v_km",
                "speed_ratio",
                "center_u_km",
                "spreading_time_s",
            ):
                fields.setdefault(name, []).append([np.nan] * len(trials))
        fields.setdefault("amplitude_tecu", []).append(
            to_trial_indexed(source.amplitude_tecu)
        )

    units = {
        "wavelength_km": "km",
        "period_s": "s",
        "phase_rad": "rad",
        "direction_rad": "rad",
        "center_x_km": "km",
        "center_y_km": "km",
        "sigma_u_km": "km",
        "sigma_v_km": "km",
        "center_u_km": "km",
        "spreading_time_s": "s",
        "amplitude_tecu": "TECU",
        "speed_ratio": "1",
    }
    for name, values in fields.items():
        arr = np.asarray(values)
        if arr.shape == (n_sources,) and arr.dtype.kind in "US":
            arr = np.stack([np.full(len(trials), v) for v in arr.ravel()])
        if arr.shape == (n_sources, len(trials)):
            arr = arr.T
        coord = xr.DataArray(
            arr.astype(float) if arr.dtype.kind == "f" else arr,
            coords={"trial": trials, "source": np.arange(n_sources)},
            name=name,
        )
        if name in units:
            coord.attrs["units"] = units[name]
        data = data.assign_coords({name: coord})

    if n_sources == 1:
        first = scenario.sources[0]
        lam = np.asarray(fields["wavelength_km"][0], dtype=float)
        tau = np.asarray(fields["period_s"][0], dtype=float) / 60
        data = data.assign_coords(
            lam=("trial", lam),
            tau=("trial", tau),
        )
        data["lam"].attrs["units"] = "km"
        data["tau"].attrs["units"] = "minutes"
        if isinstance(first.wave, PlanarWave):
            d = np.asarray(fields["direction_rad"][0], dtype=float)
            data = data.assign_coords(dir=("trial", d))
            data["dir"].attrs["units"] = "rad"
    return data


# ---------------------------------------------------------------------------
# GNSS image builder and gap masking
# ---------------------------------------------------------------------------


def build_gnss_dataset(
    samples: GNSSSamples,
    field: xr.DataArray,
    scenario: Scenario,
    image_maker,
    *,
    window: int = 1,
    step: int = 1,
    schedule: np.ndarray | None = None,
    output_dtype: str = "float32",
) -> xr.Dataset:
    """Reconstruct images from epoch-level GNSS dTEC with the image maker.

    Aligns the forward-model dTEC to the retained observations, builds
    window membership after geometry/availability filtering against the
    independently requested master schedule, and calls the existing image
    maker for each window and trial. A wholly missing window has NaN
    ``image`` and zero ``density``; unexpected interpolation exceptions
    propagate rather than being swallowed as empty windows.

    Args:
        samples: :class:`GNSSSamples` from a GNSS sampler.
        field: Noisy dTEC with dims ``(trial, obs)``, aligned to
            ``samples.obs`` rows.
        scenario: Scenario owning the generating parameters; truth fields
            are attached to the output.
        image_maker: Initialized :class:`gnss_tid.image.ImageMakerBase`
            (constructed with the same ``Local2D`` projection; use
            ``hp_freq=None`` for controlled experiments).
        window: Positive count of master raw-epoch slots per window
            (single-epoch windows are the default).
        step: Positive count of slots between window starts.
        schedule: Optional master requested raw-epoch axis; when supplied,
            windows keep empty slots and logical labels from this schedule
            instead of surviving unique timestamps.
        output_dtype: dtype for the persisted ``image`` variable.

    Returns:
        Dataset with ``image(trial, time, y, x)`` and
        ``density(trial, time, y, x)`` on the full requested image-time
        axis (window start times), plus per-source truth coordinates. A
        wholly missing window has NaN image and zero density.

    Raises:
        ValueError: If the window or step is not a positive count.
        RuntimeError: If every window is missing.
    """
    from .pointdata import TimeWindow

    if window <= 0 or step <= 0:
        raise ValueError("window and step must be positive counts of master slots")
    obs = samples.obs
    times = pd.to_datetime(obs["time"]).to_numpy()
    unique_times, inverse = np.unique(times, return_inverse=True)

    # master schedule: retain empty windows and logical start/end labels;
    # surviving unique timestamps do not define the requested schedule
    if schedule is not None:
        epochs = np.unique(pd.to_datetime(np.asarray(schedule)).to_numpy())
        times = times.astype(epochs.dtype)
        unique_times = unique_times.astype(epochs.dtype)
        # map surviving timestamps onto master slot indices for membership
        slot_of_time = np.searchsorted(epochs, unique_times)
        obs_slot = slot_of_time[inverse]
        frame_slots = np.arange(0, len(epochs), step)
    else:
        epochs = unique_times
        obs_slot = inverse
        frame_slots = np.arange(0, len(epochs), step)

    windows = []
    for slot_start in frame_slots:
        slot_stop = slot_start + window
        members = np.where((obs_slot >= slot_start) & (obs_slot < slot_stop))[0]
        if slot_stop <= len(epochs):
            windows.append(TimeWindow(epochs[slot_start], epochs[slot_stop - 1], members))
        else:
            break

    trial_values = field["trial"].values
    frames = []
    for w in windows:
        members = w.row_indices
        if len(members) == 0:
            frames.append(None)
            continue
        x = obs["x"].to_numpy()[members]
        y = obs["y"].to_numpy()[members]
        per_trial = []
        for i in range(len(trial_values)):
            values = field.isel(trial=i, obs=members).values
            result = image_maker(x, y, values)
            per_trial.append(result.expand_dims(time=[w.start_time]))
        frames.append(xr.concat(per_trial, dim="trial"))

    valid = [f for f in frames if f is not None]
    if not valid:
        raise RuntimeError("GNSS image generation produced no valid windows")

    first = valid[0]
    template_times = [w.start_time for w in windows]
    image_shape = first["image"].squeeze("time").shape[1:]
    density_shape = first["density"].squeeze("time").shape[1:]
    image = np.full(
        (len(trial_values), len(template_times), *image_shape),
        np.nan,
        dtype=output_dtype,
    )
    density = np.zeros((len(trial_values), len(template_times), *density_shape))
    for t_idx, frame in enumerate(frames):
        if frame is None:
            continue
        image[:, t_idx] = frame["image"].squeeze("time").values
        density[:, t_idx] = frame["density"].squeeze("time").values
    data = xr.Dataset(
        {
            "image": (("trial", "time", "y", "x"), image),
            "density": (("trial", "time", "y", "x"), density),
        },
        coords={
            "trial": trial_values,
            "time": np.asarray(template_times),
            "y": first["y"].values,
            "x": first["x"].values,
        },
    )
    data["image"].attrs.update(
        {
            "units": "TECU",
            "long_name": "shell-equivalent dTEC reconstructed at IPPs, no slant mapping",
            "reconstruction_height_km": samples.true_height_km,
        }
    )
    data = _attach_truth(data, scenario)
    return data


def apply_gap_mask(
    params: xr.Dataset,
    data: xr.Dataset,
    smooth_win: int = 9,
) -> xr.Dataset:
    """Mark estimates affected by missing frames as NaN.

    A wholly missing frame is identifiable from all-zero density or an
    entirely nonfinite image; an otherwise supported zero signal is never
    classified as missing. The detector turns zero-density frames into zero
    FFTs, so phase products crossing them have angle zero and bias
    neighboring smoothed estimates. The mask derives from the exact
    forward-difference and ``_rolling_freq_stats()`` support for
    ``smooth_win``; affected ``period``, ``wavelength``, and
    ``phase_velocity`` values are marked NaN after estimation.

    Args:
        params: Estimator output with dims ``(trial, time, py, px)``.
        data: Image dataset with ``density(trial, time, y, x)``.
        smooth_win: Odd rolling-window length used by the estimator's
            frequency smoothing.

    Returns:
        Dataset with gap-affected ``period``, ``wavelength``, and
        ``phase_velocity`` values set to NaN; other variables unchanged.
    """
    if "density" not in data:
        return params
    density = data["density"]
    # a wholly missing frame has zero density everywhere (no available
    # simulated measurements); sparse images legitimately contain zero
    # pixels at poorly supported grid points, so "any pixel is zero" must
    # not classify a frame as missing
    missing = (density <= 0).all(dim=("y", "x"))
    missing = missing | data["image"].isnull().all(dim=("y", "x"))
    if not bool(missing.any()):
        return params

    # forward-difference support: freq[t] uses frames t and t-1
    diff_affected = missing | missing.shift(time=1, fill_value=False)
    # centered rolling mean/std support for smooth_win
    roll_affected = missing.rolling(time=smooth_win, center=True, min_periods=1).max()
    affected = np.asarray(diff_affected, dtype=bool) | (np.asarray(roll_affected) > 0)
    out = params.copy()
    for name in ("period", "wavelength", "phase_velocity"):
        if name in out:
            values = out[name].values.copy()
            values[affected] = np.nan
            out[name] = (out[name].dims, values)
    return out
