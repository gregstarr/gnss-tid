from pathlib import Path
import numpy as np
import pandas as pd
import hydra
import matplotlib.pyplot as plt
import matplotlib.animation as animation



@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg):
    points = hydra.utils.instantiate(cfg.pointdata)
    data = points.get_data_at_height(200)
    data_slice = data.isel(time=slice(0, cfg.run.time_step))

    fig, ax = plt.subplots(figsize=(5,5))
    x = np.nanmean(data_slice.x.values, axis=0)
    y = np.nanmean(data_slice.y.values, axis=0)
    tid = np.nanmean(data_slice.tid.values, axis=0)
    scat = ax.scatter(x, y, c=tid, s=3, vmin=-.3, vmax=.3, cmap='bwr')
    ax.grid(True)
    ax.set_xlim(-1500, 1500)
    ax.set_ylim(-1500, 1500)
    ax.set_title(data_slice.time.values)

    def update(frame):
        data_slice = data.isel(time=slice(frame * cfg.run.time_step, (frame + 1) * cfg.run.time_step))
        x = np.nanmean(data_slice.x.values, axis=0)
        y = np.nanmean(data_slice.y.values, axis=0)
        tid = np.nanmean(data_slice.tid.values, axis=0)
        xy = np.column_stack((x, y))
        scat.set_offsets(xy)
        scat.set_array(tid)
        ax.set_title(data_slice.time.values)
        return scat
    
    ani = animation.FuncAnimation(fig=fig, func=update, frames=data.time.shape[0]//cfg.run.time_step, interval=150)
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    ani.save(filename=output_dir / "pillow_example.gif", writer="pillow")


if __name__ == "__main__":
    main()
