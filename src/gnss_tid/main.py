import hydra


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg):
    points = hydra.utils.instantiate(cfg.event.pointdata)
    process = hydra.utils.instantiate(cfg.focus)
    focus_data = process.run(points, cfg.sample.window, cfg.sample.step)
    focus_data.to_netcdf(cfg.output_fn)


if __name__ == "__main__":
    main()
