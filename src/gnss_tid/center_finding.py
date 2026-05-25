import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn

logger = logging.getLogger(__name__)


PHASE_BINS_CORRELATION = 25
PHASE_BINS_STATIONARY = 50
CENTER_INIT_RANGE = 500
WAVELENGTH_INIT_NOISE = 50
WAVELENGTH_PERTURB_RANGE = 10
WAVELENGTH_PERTURB_OFFSET = 30


class CorrelationModel(nn.Module):
    """Wave model returning the best phase-correlation metric over phase offsets."""

    def __init__(self, c0: np.ndarray, w0: float) -> None:
        super().__init__()
        self.center = nn.Parameter(torch.tensor(c0))
        self.wavelength = nn.Parameter(torch.tensor(w0))

    def forward(
        self, xy: torch.Tensor, tid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dist = torch.linalg.vector_norm(xy - self.center, dim=1, keepdim=True)
        phase = 2.0 * torch.pi * dist / self.wavelength
        offsets = torch.linspace(0.0, 2.0 * torch.pi, PHASE_BINS_CORRELATION)
        vals = torch.mean(torch.cos(phase + offsets[None, :]) * tid[:, None], dim=0)
        metric = torch.max(vals)
        best_phase = offsets[torch.argmax(vals)]
        return metric, best_phase


class CenterModel(nn.Module):
    """Stationary-wave model with per-time-step wavelengths."""

    def __init__(self, c0: np.ndarray, w0: np.ndarray) -> None:
        super().__init__()
        self.center = nn.Parameter(torch.tensor(c0))
        self.wavelength = nn.Parameter(torch.tensor(w0))

    def forward(
        self, xy: torch.Tensor, tid: torch.Tensor, n_fin: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dist = torch.linalg.vector_norm(xy - self.center, dim=1)
        phase = 2.0 * torch.pi * dist[:, None] / self.wavelength[None, :]
        offsets = torch.linspace(0.0, 2.0 * torch.pi, PHASE_BINS_STATIONARY)
        vals = (
            torch.sum(
                torch.cos(phase[:, :, None] + offsets[None, None, :])
                * tid[:, :, None],
                dim=0,
            )
            / n_fin[:, None]
        )
        m, idx = torch.max(vals, dim=1)
        metric = torch.sum(m)
        best_phase = offsets[idx]
        return metric, best_phase


class EarlyStopping:
    """Stops when ``val`` hasn't improved by ``improvement`` for ``period`` epochs."""

    def __init__(self, improvement: float, period: int) -> None:
        self.best: float | None = None
        self.best_epoch = 0
        self.improvement = improvement
        self.period = period

    def should_stop(self, val: float, epoch: int) -> bool:
        if self.best is None:
            self.best = val
            self.best_epoch = epoch
            return False

        check_val = 2 * (val - self.best) / abs(val + self.best)
        if check_val > self.improvement:
            self.best_epoch = epoch
            self.best = val

        return epoch - self.best_epoch > self.period


def _empty_history() -> dict[str, list]:
    return {"center": [], "wavelength": [], "metric": [], "phase": []}


def _correlation_result(history: dict[str, list], epoch: int) -> dict[str, Any]:
    wavelength = history["wavelength"][epoch]
    offset = history["phase"][epoch] * wavelength / (2 * np.pi)
    return {
        "center": history["center"][epoch],
        "wavelength": wavelength,
        "metric": history["metric"][epoch],
        "offset": offset,
        "history": history,
    }


def _stationary_result(
    history: dict[str, list], epoch: int | None = None
) -> dict[str, Any]:
    if epoch is None:
        epoch = int(np.argmax(history["metric"]))
    wavelength = history["wavelength"][epoch]
    phase = history["phase"][epoch]
    offset = np.unwrap(phase) * wavelength / (2 * np.pi)
    return {
        "center": history["center"][epoch],
        "wavelength": wavelength,
        "metric": history["metric"][epoch],
        "offset": offset,
        "phase": phase,
        "history": history,
    }


def run_correlation_center_finder(
    c0: np.ndarray,
    w: float,
    x: np.ndarray,
    y: np.ndarray,
    tec: np.ndarray,
    *,
    max_iter: int,
    learning_rate: float,
    improvement: float,
    steps: int,
) -> dict[str, Any]:
    """Fit centre + wavelength against ``tec`` using phase-correlation maximization.

    Args:
        c0: ``(2,)`` initial centre coordinates.
        w: Initial wavelength scalar.
        x, y: ``(N,)`` point coordinates.
        tec: ``(N,)`` TEC values.
        max_iter: Maximum optimization steps.
        learning_rate: Adam learning rate.
        improvement: Relative-improvement threshold for early stopping.
        steps: Patience (epochs) for early stopping.
    """
    model = CorrelationModel(c0, w)
    r = torch.tensor(np.column_stack([x, y]))
    tid = torch.tensor(tec)
    optimizer = torch.optim.Adam(model.parameters(), learning_rate, maximize=True)
    stopper = EarlyStopping(improvement, steps)
    history = _empty_history()

    for epoch in range(max_iter):
        model.zero_grad()
        metric, phase = model(r, tid)
        history["center"].append(model.center.clone().detach().numpy())
        history["wavelength"].append(model.wavelength.item())
        history["metric"].append(metric.item())
        history["phase"].append(phase.item())
        metric.backward()
        optimizer.step()
        if stopper.should_stop(metric.item(), epoch):
            break

    return _correlation_result(history, stopper.best_epoch)


def run_stationary_center_finder(
    c0: np.ndarray,
    w0: float,
    x: np.ndarray,
    y: np.ndarray,
    tec: np.ndarray,
    *,
    max_iter: int,
    learning_rate: float,
    num_starts: int,
    reg_strength: float,
) -> dict[str, Any]:
    """Multi-start fit of centre and per-time-step wavelength against ``tec``.

    Args:
        c0: ``(2,)`` initial centre coordinates.
        w0: Initial wavelength scalar (broadcast across time inside perturbation).
        x, y: ``(N,)`` point coordinates.
        tec: ``(N, T)`` TEC values; ``T`` is the number of time samples.
        max_iter: Optimization steps per restart.
        learning_rate: Adam learning rate.
        num_starts: Number of random restarts; best metric wins.
        reg_strength: Weight on the 2nd-difference wavelength smoothness penalty.
    """
    # Local import to break the center_finding <-> plotting module cycle.
    from .plotting import plot_center_finder_fit

    logger.info("initialization | center: %s | wavelength: %s", str(c0), str(w0))
    xy = torch.tensor(np.column_stack((x, y)))
    tid = torch.tensor(tec)
    mask = torch.isfinite(tid)
    n_fin = torch.sum(mask, dim=0)
    torch.nan_to_num_(tid, 0)

    result_list = []
    for iteration in range(num_starts):
        center_perturbation = (
            np.random.rand(2) * CENTER_INIT_RANGE - CENTER_INIT_RANGE / 2
        )
        wavelength_perturbation = (
            np.random.rand() * WAVELENGTH_INIT_NOISE
            + np.random.rand(tec.shape[1]) * WAVELENGTH_PERTURB_RANGE
            - WAVELENGTH_PERTURB_OFFSET
        )
        model = CenterModel(c0 + center_perturbation, w0 + wavelength_perturbation)
        logger.info("iteration: %2d / %2d", iteration + 1, num_starts)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        history = _empty_history()

        for _step in range(max_iter):
            model.zero_grad()
            metric, phase = model(xy, tid, n_fin)
            history["center"].append(model.center.clone().detach().numpy())
            history["wavelength"].append(model.wavelength.clone().detach().numpy())
            history["metric"].append(metric.item())
            history["phase"].append(phase.clone().detach().numpy())
            reg_loss = torch.mean(torch.diff(model.wavelength, 2) ** 2)
            loss = -metric + reg_strength * reg_loss
            loss.backward()
            optimizer.step()

        result_list.append(_stationary_result(history))
        logger.info("metric: %f", result_list[-1]["metric"])
        logger.info(
            "center: [%f, %f]",
            result_list[-1]["center"][0],
            result_list[-1]["center"][1],
        )

    best_iteration = int(np.argmax([r["metric"] for r in result_list]))
    result = result_list[best_iteration]

    fig, _ = plot_center_finder_fit(
        result_list,
        c0=c0,
        cbox=[
            c0[0] - CENTER_INIT_RANGE / 2,
            c0[1] - CENTER_INIT_RANGE / 2,
            CENTER_INIT_RANGE,
            CENTER_INIT_RANGE,
        ],
    )
    fig.savefig("plots/center_finder_fit.png")
    plt.close(fig)

    return result


def find_center(
    pts: np.ndarray, vectors: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Least-squares estimate of the wave centre from spectral patch vectors.

    Pure-numpy helper distinct from the torch-based finders; used to initialize
    their searches.

    Args:
        pts: ``(N, 2)`` patch centre coordinates.
        vectors: ``(N, 2)`` per-patch peak wavevector components.
        weights: ``(N,)`` non-negative weights (e.g. spectral peak power).

    Returns:
        ``(2,)`` estimated centre coordinates.
    """
    vec_norm = np.linalg.norm(vectors, axis=1)
    mask = (
        (vec_norm > 0)
        & np.isfinite(vec_norm)
        & np.isfinite(weights)
        & (weights > 0)
    )
    if not np.any(mask):
        return np.array([np.nan, np.nan])
    w = np.sqrt(weights[mask]) / vec_norm[mask]
    A = np.column_stack([vectors[mask, 1], -vectors[mask, 0]]) * w[:, None]
    b = np.sum(A * pts[mask], axis=1)
    center, *_ = np.linalg.lstsq(A, b)
    return center


def run_smoothed_center_finder(
    px: np.ndarray,
    py: np.ndarray,
    F: np.ndarray,
    Fx: np.ndarray,
    Fy: np.ndarray,
    sparse_x: np.ndarray,
    sparse_y: np.ndarray,
    sparse_image: np.ndarray,
    center_finder: Callable,
) -> dict[str, Any]:
    """Initialize from a spectral patch field and run a torch center finder.

    Uses :func:`find_center` on the supplied patch wavevectors to seed the
    centre, takes ``1 / max(|k|)`` as the wavelength seed, and dispatches to
    ``center_finder`` to refine against the sparse image stack.

    Args:
        px: ``(Px,)`` patch-centre x coordinates.
        py: ``(Py,)`` patch-centre y coordinates.
        F: Peak spectral power on the ``(px, py)`` grid, used as weighting
            when seeding the centre.
        Fx: Peak wavenumber x-component on the ``(px, py)`` grid.
        Fy: Peak wavenumber y-component on the ``(px, py)`` grid.
        sparse_x: ``(N,)`` x coordinates of the sparse image pixels.
        sparse_y: ``(N,)`` y coordinates of the sparse image pixels.
        sparse_image: ``(N, T)`` sparse image values across ``T`` time steps.
        center_finder: Callable ``(c0, w0, x, y, image) -> dict`` (typically
            :func:`run_stationary_center_finder`, partial-bound via Hydra).

    Returns:
        The result dict produced by ``center_finder``.
    """
    logger.info("finding center")
    X, Y = np.meshgrid(px, py)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    vectors = np.column_stack([Fx.ravel(), Fy.ravel()])
    weights = F.ravel()
    k = np.hypot(vectors[:, 0], vectors[:, 1])
    c0 = find_center(pts, vectors, weights)
    w0 = 1 / k.max()
    return center_finder(c0, w0, sparse_x, sparse_y, sparse_image)
