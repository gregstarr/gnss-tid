import logging

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas
from tqdm import tqdm
import hydra

from pointdata import PointData


def compute_radial_metric(data, center, wavelength):
    dx = data.x - center[0]
    dy = data.y - center[1]
    phase = 2 * np.pi * np.hypot(dx.values.flatten(), dy.values.flatten()) / wavelength
    phase_offset = np.linspace(0, 2 * np.pi, 25)
    templates = np.cos(phase[:, None] + phase_offset[None, :])
    vals = np.nanmean(templates * data.tid.values.flatten()[:, None], axis=0)
    metric = np.max(vals)
    offset = phase_offset[np.argmax(vals)] * wavelength / (2 * np.pi)
    return metric, offset


def plot_radial(ax, center, wavelength, offset):
    for radius in np.arange(1, 1200, wavelength):
        circle = plt.Circle(center, radius + offset, facecolor='none', edgecolor=(0, 0, 0), linewidth=1, alpha=0.5)
        ax.add_patch(circle)


def plot_points(ax, data, clim=(-.3, .3)):
    x = np.nanmean(data.x, axis=0)
    y = np.nanmean(data.y, axis=0)
    tid = np.nanmean(data.tid, axis=0)
    mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(tid))
    x = x[mask]
    y = y[mask]
    tid = tid[mask]
    ax.scatter(x, y, c=tid, s=6, vmin=clim[0], vmax=clim[1], cmap='bwr')
    ax.grid(True)


def optimize(r, tid, cfg, c0=(100.0, 50.0), w0=200.0):
    history = {
        "center": [],
        "wavelength": [],
        "metric": [],
    }
    center = tf.Variable(tf.cast(c0, tf.float32))
    wavelength = tf.Variable(tf.cast(w0, tf.float32))
    vc = tf.Variable([0.0, 0.0])
    vw = tf.Variable(0.0)
    best = 0
    best_epoch = 0
    for epoch in range(cfg.opt.max_iter):
        with tf.GradientTape() as tape:
            metric = objective(r, tid, center, wavelength)
        grad_c, grad_w = tape.gradient(metric, [center, wavelength])
        vc.assign(cfg.opt.momentum * vc + cfg.opt.learning_rate * grad_c)
        center.assign_add(vc)
        vw.assign(cfg.opt.momentum * vw + cfg.opt.learning_rate * grad_w)
        wavelength.assign_add(vw)
        history["center"].append(center.numpy())
        history["wavelength"].append(wavelength.numpy())
        history["metric"].append(metric.numpy())

        if metric - best > cfg.opt.end.improvement:
            best_epoch = epoch
            best = metric
        
        if epoch - best_epoch > cfg.opt.end.steps:
            break

    return history
    

def objective(r, tid, center, wavelength):
    dr = r - center
    phase = 2.0 * np.pi * tf.norm(dr, axis=1) / wavelength
    phase_offset = tf.linspace(0.0, 2.0 * np.pi, 25)
    templates = tf.cos(tf.expand_dims(phase, 1) + tf.expand_dims(phase_offset, 0))
    vals = tf.math.reduce_mean(templates * tf.expand_dims(tid, 1), axis=0)
    metric = tf.math.reduce_max(vals)
    # idx = tf.math.argmax(vals)
    # offset = phase_offset[idx] * wavelength / (2.0 * np.pi)
    return metric


def plot_results(points: PointData, results: list, time_slice: slice):
    df = pandas.DataFrame(results)
    best = df.iloc[df.metric.argmax()]
    data = points.get_data_at_height(best.height)
    data = data.isel(time=time_slice)
    metric_history = best.history["metric"]

    fig, ax = plt.subplots(1, 3, tight_layout=True, figsize=(12, 5))
    plot_points(ax[0], data)
    plot_radial(ax[0], best.center, best.wavelength, best.offset)
    ax[0].set_title(f"center={best.center}")

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


def run_for_height(points, height, time_slice, cfg):
    data = points.get_data_at_height(height)
    data = data.isel(time=time_slice)
    r = np.column_stack([data.x.values.flatten(), data.y.values.flatten()])
    tid = data.tid.values.flatten()
    mask = np.isnan(r).any(axis=1) | np.isnan(tid)
    r = tf.cast(r[~mask], tf.float32)
    tid = tf.cast(tid[~mask], tf.float32)

    if r.shape[0] == 0:
        return None

    centers = np.array([[100, 50]]) + np.random.rand(cfg.opt.n_inits, 2) * 100
    wavelengths = np.array([160]) + np.random.rand(cfg.opt.n_inits) * 80
    best_metric = 0
    for center, wavelength in tqdm(zip(centers, wavelengths), f"{height=}", cfg.opt.n_inits):
        result = {}
        history = optimize(r, tid, cfg, c0=center, w0=wavelength)
        i = np.argmax(history["metric"])
        metric, offset = compute_radial_metric(
            data, history["center"][i], history["wavelength"][i]
        )
        result = {
            "center": center,
            "metric": metric,
            "offset": offset,
            "wavelength": history["wavelength"][i],
            "history": history,
        }
        if metric > best_metric:
            best_metric = metric
            best_result = result
    return best_result


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


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg):
    points = hydra.utils.instantiate(cfg.pointdata)
    for i in range(0, len(points.time) - cfg.run.time_step, cfg.run.time_step):
        time_slice = slice(i, i + cfg.run.time_step)
        run_time_index(points, time_slice, cfg)
    

if __name__ == "__main__":
    main()
