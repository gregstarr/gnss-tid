from typing import Protocol

from matplotlib import pyplot as plt
import numpy as np
from metpy.interpolate import interpolate_to_grid
from skimage import filters


class ImageMaker(Protocol):
    def __call__(self, x: np.ndarray, y: np.ndarray, tec: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...


class MetpyImageMaker:
    def __init__(self, hp_freq=.05, **kwargs):
        self.kwargs = kwargs
        self.hp_freq = hp_freq
    
    def __call__(self, x: np.ndarray, y: np.ndarray, tec: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_grid, y_grid, img = interpolate_to_grid(x, y, tec, **self.kwargs)
        img[np.isnan(img)] = 0
        img = filters.butterworth(img, self.hp_freq, high_pass=True)
        return x_grid, y_grid, img


class ImageTestProcess:

    def __init__(self, image_maker: ImageMaker, height):
        self.image_maker = image_maker
        self.height = height
        self.title = "|".join([f"{k}={v}" for k, v in self.image_maker.kwargs.items()])

    def run(self, points, window: int, step: int):
        for t in points.get_time_slices(window, step):
            data = points.get_data(t, self.height)
            x_grid, y_grid, img = self.image_maker(data.x.values, data.y.values, data.tec.values)
            FFT = np.fft.fftshift(np.fft.fft2(img))
            power = abs(FFT) ** 2
            edges = filters.scharr(img)
            efft = np.fft.fftshift(np.fft.fft2(edges))
            epower = 10*np.log(abs(efft) ** 2)
            g = filters.gabor(img, .1, np.pi/4)
            g = np.hypot(*g)
            
            fig, ax = plt.subplots(3, 2, figsize=(18, 10), tight_layout=True)
            fig.suptitle(self.title)
            ax[0, 0].scatter(data.x.values, data.y.values, c=data.tec.values, s=6, vmin=-.3, vmax=.3, cmap='bwr')
            extent = [x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()]
            ax[0, 1].imshow(img, vmin=-.3, vmax=.3, cmap="bwr", origin="lower", extent=extent)
            ax[1, 0].imshow(power, origin="lower")
            ax[1, 1].imshow(edges, origin="lower", extent=extent)
            ax[2, 0].imshow(epower, origin="lower")
            ax[2, 1].imshow(g, origin="lower", extent=extent)
            fig.savefig("TEST.png")
            return
