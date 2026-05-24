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


class _BaseWaveModel(nn.Module):
    def __init__(self, c0: np.ndarray, w0: float | np.ndarray) -> None:
        super().__init__()
        self.center = nn.Parameter(torch.tensor(c0))
        self.wavelength = nn.Parameter(torch.tensor(w0))
        self.history: dict[str, list] = {
            "center": [],
            "wavelength": [],
            "metric": [],
            "phase": [],
        }


class CorrelationModel(_BaseWaveModel):
    def get_result(self, epoch):
        wavelength = self.history["wavelength"][epoch]
        offset = self.history["phase"][epoch] * wavelength / (2 * np.pi)
        return {
            "center": self.history["center"][epoch],
            "wavelength": wavelength,
            "metric": self.history["metric"][epoch],
            "offset": offset,
            "history": self.history,
        }

    def forward(self, xy, tid):
        dist = torch.linalg.vector_norm(xy - self.center, dim=1, keepdim=True)
        phase = 2.0 * torch.pi * dist / self.wavelength
        phase_offset_vals = torch.linspace(0.0, 2.0 * torch.pi, PHASE_BINS_CORRELATION)
        model_out = torch.cos(phase + phase_offset_vals[None, :])

        vals = torch.mean(model_out * tid[:, None], dim=0)
        metric = torch.max(vals)
        phase = phase_offset_vals[torch.argmax(vals)]

        self.history["center"].append(self.center.clone().detach().numpy())
        self.history["wavelength"].append(self.wavelength.item())
        self.history["metric"].append(metric.item())
        self.history["phase"].append(phase.item())

        return metric


class EarlyStopping:
    def __init__(self, improvement, period) -> None:
        self.best = None
        self.best_epoch = 0
        self.improvement = improvement
        self.period = period

    def should_stop(self, val, epoch):
        if self.best is None:
            self.best = val
            self.best_epoch = epoch
            return False

        check_val = 2 * (val - self.best) / abs(val + self.best)
        if check_val > self.improvement:
            self.best_epoch = epoch
            self.best = val

        return epoch - self.best_epoch > self.period


class CenterFinder:
    def __init__(self, max_iter, learning_rate, improvement, steps):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.stopper = EarlyStopping(improvement, steps)

    def __call__(self, c0, w, x, y, tec):
        # c0: (2,) ndarray; w: scalar; tec: (N,) array
        model = CorrelationModel(c0, w)
        r = torch.tensor(np.column_stack([x, y]))
        tid = torch.tensor(tec)

        optimizer = torch.optim.Adam(
            model.parameters(), self.learning_rate, maximize=True
        )
        for epoch in range(self.max_iter):
            model.zero_grad()
            metric = model(r, tid)
            metric.backward()
            optimizer.step()
            if self.stopper.should_stop(metric.item(), epoch):
                break

        return model.get_result(self.stopper.best_epoch)


class CenterModel(_BaseWaveModel):
    def get_result(self, epoch=None):
        if epoch is None:
            epoch = np.argmax(self.history["metric"])
        wavelength = self.history["wavelength"][epoch]
        phase = self.history["phase"][epoch]
        offset = np.unwrap(phase) * wavelength / (2 * np.pi)
        return {
            "center": self.history["center"][epoch],
            "wavelength": wavelength,
            "metric": self.history["metric"][epoch],
            "offset": offset,
            "phase": phase,
            "history": self.history,
        }

    def forward(self, xy, tid, n_fin):
        dist = torch.linalg.vector_norm(xy - self.center, dim=1)
        phase = 2.0 * torch.pi * dist[:, None] / self.wavelength[None, :]
        phase_offset_vals = torch.linspace(0.0, 2.0 * torch.pi, PHASE_BINS_STATIONARY)
        model_out = torch.cos(phase[:, :, None] + phase_offset_vals[None, None, :])
        vals = torch.sum(model_out * tid[:, :, None], dim=0) / n_fin[:, None]
        m, idx = torch.max(vals, dim=1)
        metric = torch.sum(m)
        phase = phase_offset_vals[idx]

        self.history["center"].append(self.center.clone().detach().numpy())
        self.history["wavelength"].append(self.wavelength.clone().detach().numpy())
        self.history["metric"].append(metric.item())
        self.history["phase"].append(phase.clone().detach().numpy())

        return -1 * metric


class StationaryCenterFinder:
    def __init__(self, max_iter, learning_rate, num_starts, reg_strength):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.num_starts = num_starts
        self.reg_strength = reg_strength

    def __call__(self, c0, w0, x, y, tec):
        # c0: (2,) ndarray; w0: scalar (broadcast to (T,) inside the perturbation);
        # tec: (N, T) array — T is the number of time samples.
        # Local import to break the center_finding <-> plotting module cycle.
        from .plotting import plot_center_finder_fit

        logger.info("initialization | center: %s | wavelength: %s", str(c0), str(w0))
        xy = torch.tensor(np.column_stack((x, y)))
        tid = torch.tensor(tec)
        mask = torch.isfinite(tid)
        n_fin = torch.sum(mask, dim=0)
        torch.nan_to_num_(tid, 0)

        result_list = []
        for iteration in range(self.num_starts):
            center_perturbation = (
                np.random.rand(2) * CENTER_INIT_RANGE - CENTER_INIT_RANGE / 2
            )
            wavelength_perturbation = (
                np.random.rand() * WAVELENGTH_INIT_NOISE
                + np.random.rand(tec.shape[1]) * WAVELENGTH_PERTURB_RANGE
                - WAVELENGTH_PERTURB_OFFSET
            )
            model = CenterModel(c0 + center_perturbation, w0 + wavelength_perturbation)
            logger.info("iteration: %2d / %2d", iteration + 1, self.num_starts)
            optimizer = torch.optim.Adam(model.parameters(), self.learning_rate)
            for _step in range(self.max_iter):
                model.zero_grad()
                data_loss = model(xy, tid, n_fin)
                reg_loss = torch.mean(torch.diff(model.wavelength, 2) ** 2)
                loss = data_loss + self.reg_strength * reg_loss
                loss.backward()
                optimizer.step()
            result_list.append(model.get_result())
            logger.info("metric: %f", result_list[-1]["metric"])
            logger.info(
                "center: [%f, %f]",
                result_list[-1]["center"][0],
                result_list[-1]["center"][1],
            )
        best_iteration = np.argmax([r["metric"] for r in result_list])
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

    Pure-numpy helper distinct from the torch-based :class:`CenterFinder` and
    :class:`StationaryCenterFinder`; used to initialize their searches.

    Args:
        pts: ``(N, 2)`` patch centre coordinates.
        vectors: ``(N, 2)`` per-patch peak wavevector components.
        weights: ``(N,)`` non-negative weights (e.g. spectral peak power).

    Returns:
        ``(2,)`` estimated centre coordinates.
    """
    vec_norm = np.linalg.norm(vectors, axis=1)
    mask = vec_norm > 0
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
            :class:`StationaryCenterFinder`).

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
