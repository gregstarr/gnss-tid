import logging

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import pandas
import tqdm
import hydra

from pointdata import PointData
from plotting import plot_points, plot_radial, plot_radial2


class BasicMseModel(nn.Module):
    def __init__(self, cbounds, wbounds) -> None:
        super().__init__()
        self.history = {
            "center": [],
            "wavelength": [],
            "metric": [],
            "phase": [],
            "amplitude": [],
        }
        self.center = nn.Parameter(torch.empty(2))
        self.wavelength = nn.Parameter(torch.empty(1))
        self.amplitude = nn.Parameter(torch.tensor(0.1), requires_grad=False)
        with torch.no_grad():
            nn.init.uniform_(self.center, *cbounds)
            nn.init.uniform_(self.wavelength, *wbounds)
    
    def get_result(self, epoch):
        wavelength = self.history["wavelength"][epoch]
        offset = self.history["phase"][epoch] * wavelength / (2 * np.pi)
        return {
            "center": self.history["center"][epoch],
            "amplitude": self.history["amplitude"][epoch],
            "wavelength": wavelength,
            "metric": self.history["metric"][epoch],
            "offset": offset,
            "history": self.history,
        }

    def forward(self, xy, tid):
        dist = torch.linalg.vector_norm(xy - self.center * 100.0, dim=1, keepdim=True)
        phase = 2.0 * torch.pi * dist / (self.wavelength * 100.0)
        phase_offset_vals = torch.linspace(0.0, 2.0 * torch.pi, 25)
        model_out = self.amplitude * torch.cos(phase + phase_offset_vals[None, :])
        
        err = model_out - tid[:, None]
        vals = torch.mean(err ** 2, dim=0)
        metric = -1.0 * torch.min(vals)
        phase = phase_offset_vals[torch.argmin(vals)]

        self.history["center"].append(self.center.clone().detach().numpy() * 100)
        self.history["wavelength"].append(self.wavelength.item() * 100)
        self.history["metric"].append(metric.item())
        self.history["phase"].append(phase.item())
        self.history["amplitude"].append(self.amplitude.item())

        return metric


class BasicCorrelationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.history = {
            "center": [],
            "wavelength": [],
            "metric": [],
            "phase": [],
        }
        self.center = nn.Parameter(torch.empty(2))
        self.wavelength = nn.Parameter(torch.empty(1))
        with torch.no_grad():
            nn.init.uniform_(self.center, 0.0, 200.0)
            nn.init.uniform_(self.wavelength, 150.0, 250.0)
    
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
        phase_offset_vals = torch.linspace(0.0, 2.0 * torch.pi, 25)
        model_out = torch.cos(phase + phase_offset_vals[None, :])
        
        vals = torch.mean(model_out * tid[:, None], dim=0)
        metric = torch.max(vals)
        phase = phase_offset_vals[torch.argmax(vals)]

        self.history["center"].append(self.center.clone().detach().numpy())
        self.history["wavelength"].append(self.wavelength.item())
        self.history["metric"].append(metric.item())
        self.history["phase"].append(phase.item())

        return metric
    

class QuadtraticCorrelationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.history = {
            "cx": [],
            "cy": [],
            "wavelength": [],
            "wavelength2": [],
            "metric": [],
            "phase": [],
        }
        self.center = nn.Parameter(torch.empty(2))
        self.wavelength = nn.Parameter(torch.empty(1))
        self.wavelength2 = nn.Parameter(torch.empty(1))
        with torch.no_grad():
            nn.init.uniform_(self.center, 20.0, 200.0)
            nn.init.uniform_(self.wavelength, 150.0, 250.0)
            nn.init.uniform_(self.wavelength2, -12.0, -5.0)
    
    def get_result(self, epoch):
        
        return {
            "cx": self.history["cx"][epoch],
            "cy": self.history["cy"][epoch],
            "wavelength": self.history["wavelength"][epoch],
            "wavelength2": self.history["wavelength2"][epoch],
            "metric": self.history["metric"][epoch],
            "offset": self.history["phase"][epoch],
            "history": self.history,
        }

    def forward(self, xy, tid):
        dist = torch.linalg.vector_norm(xy - self.center, dim=1, keepdim=True)
        offset_vals = torch.linspace(0.0, 300.0, 30)
        dist = dist + offset_vals[None, :]
        w2 = torch.exp(self.wavelength2)
        phase = 2 * torch.pi * torch.atan(dist * torch.sqrt(w2/self.wavelength)) / torch.sqrt(self.wavelength * w2)
        model_out = torch.cos(phase)
        
        vals = torch.mean(model_out * tid[:, None], dim=0)
        metric = torch.max(vals)
        phase = offset_vals[torch.argmax(vals)]

        self.history["cx"].append(self.center[0].item())
        self.history["cy"].append(self.center[1].item())
        self.history["wavelength"].append(self.wavelength.item())
        self.history["wavelength2"].append(w2.item())
        self.history["metric"].append(metric.item())
        self.history["phase"].append(phase.item())

        return metric


def optimize(r, tid, cfg, height):
    best_metric = -np.inf
    for start in tqdm.tqdm(range(cfg.opt.n_inits), f"{height=}"):
        model = hydra.utils.instantiate(cfg.model)
        stopper = hydra.utils.instantiate(cfg.stopper)
        optimizer = torch.optim.NAdam(
            model.parameters(), cfg.opt.learning_rate, maximize=True
        )
        for epoch in range(cfg.opt.max_iter):
            model.zero_grad()

            metric = model(r, tid)
            metric.backward()
            optimizer.step()


            if stopper.should_stop(metric.item(), epoch):
                break
        
        result = model.get_result(stopper.best_epoch)
        if result["metric"] > best_metric:
            best_metric = result["metric"]
            best_result = result
    return best_result


def run_for_height(points, height, time_slice, cfg):
    data = points.get_data_at_height(height)
    data = data.isel(time=time_slice)
    r = np.column_stack([data.x.values.flatten(), data.y.values.flatten()])
    tid = data.tid.values.flatten()
    mask = np.isnan(r).any(axis=1) | np.isnan(tid)
    r = torch.tensor(r[~mask])
    tid = torch.tensor(tid[~mask])

    if r.shape[0] == 0:
        return None

    result = optimize(r, tid, cfg, height)
    return result


def run_time_index(points, time_slice, cfg):
    logging.info("running times: %s - %s", points.time[time_slice.start], points.time[time_slice.stop])
    heights = np.arange(100, 400 + cfg.run.height_res, cfg.run.height_res)
    results = []
    for height in heights:
        result = run_for_height(points, height, time_slice, cfg)
        if result is None:
            continue
        result["height"] = height
        results.append(result)
    logging.info("%s / %s results for this time", len(results), len(heights))
    if len(results) == 0:
        return
    plot_results(points, results, time_slice)


def plot_results(points: PointData, results: list, time_slice: slice):
    df = pandas.DataFrame(results)
    best = df.iloc[df.metric.argmax()]
    data = points.get_data_at_height(best.height)
    data = data.isel(time=time_slice)
    metric_history = np.array(best.history["metric"])
    print(best)
    fig, ax = plt.subplots(len(best.history), 1, tight_layout=True, figsize=(4, len(best.history)*3), sharex=True)
    for i, (key, value) in enumerate(best.history.items()):
        ax[i].plot(value)
        ax[i].set_ylabel(key)
        ax[i].grid(True)
        fig.savefig(f"fit_{time_slice.start:02d}.png")
        plt.close(fig)

    fig, ax = plt.subplots(1, 3, tight_layout=True, figsize=(12, 5))
    plot_points(ax[0], data)
    plot_radial2(ax[0], (best.cx, best.cy), best.wavelength, best.wavelength2, best.offset)
    ax[0].set_title(f"center=({best.cx}, {best.cy})")

    ax[1].axvline(best.height, color="k", linestyle='--', alpha=.5)
    ax[1].plot(df.height, df.metric, '.-')
    ax[1].set_xlabel("height (km)")
    ax[1].set_ylabel("metric")
    ax[1].grid(True)
    ax[1].set_title(f"wavelength={best.wavelength}")

    ax[2].plot(metric_history)
    ax[2].set_xlabel("step")
    ax[2].set_ylabel("metric")
    ax[2].grid(True)
    ax[2].set_title(f"time={points.time[time_slice.start]}")

    fig.savefig(f"radial_{time_slice.start:02d}.png")
    plt.close(fig)


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg):
    points = hydra.utils.instantiate(cfg.pointdata)
    for i in range(0, len(points.time) - cfg.run.time_step, cfg.run.time_step):
        time_slice = slice(i, i + cfg.run.time_step)
        run_time_index(points, time_slice, cfg)
    

if __name__ == "__main__":
    main()
