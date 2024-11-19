import hydra
import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import fft2


def plot_row(ax, data, img, blocks, step):
    # scatter, image, spatial, max    
    ax[0].scatter(data.x.values, data.y.values, c=data.tec.values, s=5, vmin=-.3, vmax=.3, cmap='bwr', alpha=.5)
    ax[0].set_aspect(1)
    
    # image
    ax[1].imshow(img, vmin=-.3, vmax=.3, cmap="bwr", origin="lower")
    
    # spatial
    mag = np.max(blocks, axis=(-2, -1))
    freq_idx = np.argwhere(blocks == np.max(blocks, axis=(-2, -1), keepdims=True))
    uidx = np.unique(freq_idx[:, :2], axis=0, return_index=True)[1]
    freqs = np.fft.fftfreq(blocks.shape[-1])[freq_idx[uidx, 2:]]
    fm = np.hypot(freqs[:, 0], freqs[:, 1])
    fnorm = freqs / np.where(fm > 0, fm, 1)[:, None]
    U = np.reshape(fnorm[:, 1], blocks.shape[:2]) * mag
    V = np.reshape(fnorm[:, 0], blocks.shape[:2]) * mag
    C = np.reshape(fm, blocks.shape[:2])
    X, Y = np.meshgrid(
        np.arange(blocks.shape[1]) * step + blocks.shape[3] // 2,
        np.arange(blocks.shape[0]) * step + blocks.shape[2] // 2,
    )
    p = ax[1].quiver(
        X, Y, U, V, C,
        headwidth=0,
        headlength=0,
        headaxislength=0,
        pivot="mid",
        angles="xy",
        scale_units="xy",
        scale=50,
        units="xy",
        width=1.5,
        cmap="bone",
        clim=(0, .25),
    )
    plt.colorbar(p, label="freq", fraction=.1)
    # focus
    ax[2].imshow(np.fft.fftshift(blocks.max(axis=(0, 1))), origin="lower")
    ax[2].set_ylabel(f"max = {blocks.max():.2f}")


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg):
    points = hydra.utils.instantiate(cfg.event.pointdata)
    focuser = hydra.utils.instantiate(cfg.focus)

    nrows = len(focuser.heights) // 4
    fig, ax = plt.subplots(nrows, 3, figsize=(20, 4 * nrows), tight_layout=True)
    title1 = "|".join([f"{k}={v}" for k, v in focuser.image_maker.kwargs.items()])
    title2 = f"blk_size={cfg.focus.block_size}|blk_step={cfg.focus.block_step}"
    ax[0, 0].set_title(title1)
    ax[0, 1].set_title(title2)

    B = []
    F = []
    for i, height in enumerate(focuser.heights):
        row = i // 4
        data = points.get_data(slice(4), height)
        print(f"height = {height}")
        img = focuser.image_maker(data.x.values, data.y.values, data.tec.values)
        F.append(abs(fft2(img)) ** 2)
        blocks = focuser.get_fft_blocks(img)
        B.append(blocks)
        if (i % 4) == 0:
            ax[row, 0].set_ylabel(f"height={height}")
            plot_row(ax[row], data, img, blocks, focuser.block_step)
    fig.savefig("spectral.png")
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8), tight_layout=True)
    B = np.stack(B, axis=0)
    F = np.stack(F, axis=0)
    baseline = F.max(axis=(1, 2))
    height_max_all = B.max(axis=(1, 2, 3, 4))
    space_max_heightfreq = B.max(axis=(0, 3, 4), keepdims=True)
    idx = np.argwhere(B == space_max_heightfreq)
    uidx = np.unique(idx[:, [1,2]], axis=0, return_index=True)[1]
    idx = idx[uidx, 1:]
    heightspace_maxi_freq = B[:, *tuple(idx.T)]
    height_sum_space_maxi_freq = heightspace_maxi_freq.sum(axis=1)
    space_max_height_maxi_freq = np.max(heightspace_maxi_freq, axis=0)
    mask = space_max_height_maxi_freq > np.quantile(space_max_height_maxi_freq, .75)
    height_sum_topq_space_maxi_freq = heightspace_maxi_freq[:, mask].sum(axis=1)

    ax.grid(True)
    ax.plot(focuser.heights, height_max_all / height_max_all.max(), label="max all")
    ax.plot(focuser.heights, baseline / baseline.max(), label="single fft max")
    ax.plot(focuser.heights, height_sum_space_maxi_freq / height_sum_space_maxi_freq.max(), label="spatial sum")
    ax.plot(focuser.heights, height_sum_topq_space_maxi_freq / height_sum_topq_space_maxi_freq.max(), label="topq spatial sum")
    ax.set_title(title1 + "|" + title2)
    ax.legend()
    fig.savefig("focus.png")
    

if __name__ == "__main__":
    main()
