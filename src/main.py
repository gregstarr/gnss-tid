import hydra
from omegaconf import OmegaConf


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg):
    points = hydra.utils.instantiate(cfg.event.pointdata)
    process = hydra.utils.instantiate(cfg.focus)
    focus_data = process.run(points, cfg.sample.window, cfg.sample.step)
    focus_data = focus_data.assign_attrs(
        center=points.get_coord_center(),
        cfg=OmegaConf.to_container(cfg, resolve=True),
    )
    focus_data.to_netcdf(cfg.output_fn)


if __name__ == "__main__":
    main()
