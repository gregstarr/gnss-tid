import xarray
import gnss_tid.parameter


def save_data(data_fn):
    print("#" * 80)
    print("DATA")
    print("#" * 80)
    print()
    data = gnss_tid.synthetic.constant_model2(
        center=xarray.DataArray([0, 0], dims=["ci"]),
        snr=da.linspace(-6, 6, 30).persist(),
        wavelength=da.linspace(150, 400, 6).persist(),
        tau=da.linspace(10, 50, 6).persist(),
        xlim=(-1500, 1500),
        ylim=(-1500, 1500),
        hres=20,
    )
    
    compressor = BloscCodec(cname=BloscCname.lz4, clevel=5, shuffle=BloscShuffle.shuffle)
    encoding = {var: {"compressors": compressor} for var in data.data_vars}
    data.to_zarr(data_fn, mode="w", encoding=encoding)


def get_template(data, block_size, step_size):
    edges = block_size // (2 * step_size)
    patches = (
        data.image.isel(time=[0], tau=[0], lam=[0], snr=[0])
        .rolling(y=block_size, x=block_size, center=True)
        .construct(x="kx", y="ky", stride=step_size)
        .isel(x=slice(edges, -edges), y=slice(edges, -edges))
        .rename({"x": "px", "y": "py"})
    )
    s = (
        patches.sizes["px"],
        patches.sizes["py"],
        data.sizes["lam"],
        data.sizes["tau"],
        data.sizes["time"],
        data.sizes["snr"],
    )
    coords={
        "px": patches.px,
        "py": patches.py,
        "time": data.time,
        "lam": data.lam,
        "tau": data.tau,
        "snr": data.snr,
    }
    temp_arr = xarray.DataArray(
        da.zeros(s),
        coords=coords,
        dims=("px", "py", "lam", "tau", "time", "snr"),
    )
    var_names = ["phase_speed", "wavelength", "period", "vx", "vy", "R", "coherence"]
    temp = xarray.Dataset({name: temp_arr.rename(name) for name in var_names})
    return temp.chunk(px=-1, py=-1, lam=1, tau=1, time=-1, snr=1)


if __name__ == "__main__":
    file = "outputs/2024-12-06/22-20-41/autofocus.h5"
    data = xarray.open_dataset(file)
    
    BLOCK_SIZE = 32
    STEP_SIZE = 8
    NFFT = 128

    params = gnss_tid.parameter.estimate_parameters_block_v4(data, NFFT, BLOCK_SIZE, STEP_SIZE)
    print(params)
