import numpy as np
import pandas
import matplotlib.pyplot as plt
from skimage import measure

from pointdata import PointData
from image import ImageMaker


class SpectralFocusing:

    def __init__(self, image_maker: ImageMaker, hmin, hmax, hres):
        self.image_maker = image_maker
        self.heights = np.arange(hmin, hmax, hres)

    def run(self, points: PointData, window: int, step: int):
        for t in points.get_time_slices(window, step):
            for h in self.heights:
                data = points.get_data(t, h)
                self.focus(data)

    def focus(self, data):
        x_grid, y_grid, img = self.image_maker(data.x.values, data.y.values, data.tec.values)
        FFT = np.fft.fftshift(np.fft.fft2(img))
        power = 10 * np.log10(abs(FFT) ** 2)
        half_power_level = np.nanmax(power) - 3
        label_img = measure.label(power > half_power_level)
        props = (
            "label", "centroid", "coords", "intensity_max", "intensity_mean", 
            "intensity_std", "num_pixels",
        )
        regions = measure.regionprops_table(label_img, power, properties=props)
        regions = (
            pandas.DataFrame(regions)
            .query("num_pixels >= 2")
        )
