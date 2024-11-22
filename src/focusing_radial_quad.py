import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import pandas
import tqdm
import hydra
from sklearn.preprocessing import StandardScaler

from pointdata import PointData
from gnss_tid.plotting import plot_points, plot_radial2


class QuadtraticCorrelationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.history = {
            "cx": [],
            "cy": [],
            "w0": [],
            "w2": [],
            "u": [],
            "v": [],
            "metric": [],
            "offset": [],
        }
        self.center = nn.Parameter(torch.randn(2) / 2.0)
        self.u = nn.Parameter(torch.randn(1))
        self.v = nn.Parameter(torch.randn(1))
        self.offset_vals = torch.linspace(0.0, 1.0, 20)
    
    def forward(self, xy, tid):
        dist = torch.linalg.vector_norm(xy - self.center, dim=1, keepdim=True)
        cycles = self.u * torch.atan(dist * self.v**2)
        model_out = torch.cos(2.0 * torch.pi * (cycles + self.offset_vals[None, :]))
        
        vals = torch.mean(model_out * tid[:, None], dim=0)
        metric = torch.max(vals)
        offset = self.offset_vals[torch.argmax(vals)]

        self.update_history(metric, offset)

        return metric, offset
    
    def update_history(self, metric, offset):
        v2 = self.v**2
        w0 = torch.abs(1 / (self.u * v2))
        w2 = torch.abs(v2 / self.u)

        self.history["cx"].append(self.center[0].item())
        self.history["cy"].append(self.center[1].item())
        self.history["w0"].append(w0.item())
        self.history["w2"].append(w2.item())
        self.history["u"].append(self.u.item())
        self.history["v"].append(v2.item())
        self.history["metric"].append(metric.item())
        self.history["offset"].append(offset.item())
    
    def get_result(self, epoch):
        return {
            "cx": self.history["cx"][epoch],
            "cy": self.history["cy"][epoch],
            "w0": self.history["w0"][epoch],
            "w2": self.history["w2"][epoch],
            "u": self.history["u"][epoch],
            "v": self.history["v"][epoch],
            "metric": self.history["metric"][epoch],
            "offset": self.history["offset"][epoch],
            "history": self.history,
        }


def optimize(r: torch.Tensor, tid: torch.Tensor, cfg, height):
    rmean = torch.mean(r, dim=0, keepdim=True)
    r.sub_(rmean)
    rstd = r.std()
    r.div_(rstd)
    tidmean = torch.mean(tid)
    tid.sub_(tidmean)
    tidstd = tid.std()
    tid.div_(tidstd)

    best_metric = -np.inf
    for start in tqdm.tqdm(range(cfg.opt.n_inits), f"{height=}"):
        model = QuadtraticCorrelationModel()
        stopper = hydra.utils.instantiate(cfg.stopper)
        optimizer = torch.optim.NAdam(
            model.parameters(), cfg.opt.learning_rate, maximize=True
        )
        for epoch in range(cfg.opt.max_iter):
            model.zero_grad()

            metric, offset = model(r, tid)
            metric.backward()
            optimizer.step()

            if stopper.should_stop(metric.item(), epoch):
                break
        
        result = model.get_result(stopper.best_epoch)
        if result["metric"] > best_metric:
            best_metric = result["metric"]
            best_result = result
    best_result["rmean"] = rmean.numpy()
    best_result["rstd"] = rstd.numpy()
    best_result["tidmean"] = tidmean.numpy()
    best_result["tidstd"] = tidstd.numpy()
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
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

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
        fig.savefig(output_dir / f"fit_{time_slice.start:02d}.png")
        plt.close(fig)

    data["x"] = (data.x - best.rmean[0, 0]) / best.rstd
    data["y"] = (data.y - best.rmean[0, 1]) / best.rstd
    fig, ax = plt.subplots(1, 3, tight_layout=True, figsize=(12, 5))
    plot_points(ax[0], data)
    plot_radial2(ax[0], (best.cx, best.cy), best.w0, best.w2, best.offset)
    ax[0].set_title(f"center=({best.cx}, {best.cy})")

    ax[1].axvline(best.height, color="k", linestyle='--', alpha=.5)
    ax[1].plot(df.height, df.metric, '.-')
    ax[1].set_xlabel("height (km)")
    ax[1].set_ylabel("metric")
    ax[1].grid(True)
    ax[1].set_title(f"wavelength={best.w0:.2f} + {best.w2:.2f}R")

    ax[2].plot(metric_history)
    ax[2].set_xlabel("step")
    ax[2].set_ylabel("metric")
    ax[2].grid(True)
    ax[2].set_title(f"time={points.time[time_slice.start]}")

    fig.savefig(output_dir / f"radial_{time_slice.start:02d}.png")
    plt.close(fig)


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg):
    points = hydra.utils.instantiate(cfg.pointdata)
    for i in range(0, len(points.time) - cfg.run.time_step, cfg.run.time_step):
        time_slice = slice(i, i + cfg.run.time_step)
        run_time_index(points, time_slice, cfg)
    

if __name__ == "__main__":
    main()
