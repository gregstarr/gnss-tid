__all__ = [
    "autofocus_smoothed_spectral",
    "autofocus_spectral_blocks",
    "estimate_parameters_block",
    "estimate_parameters_dask",
    "interpolate_points_to_image",
]


def __getattr__(name):
    if name in {"autofocus_smoothed_spectral", "autofocus_spectral_blocks"}:
        from . import autofocus

        return getattr(autofocus, name)
    if name in {"estimate_parameters_block", "estimate_parameters_dask"}:
        from . import parameter

        return getattr(parameter, name)
    if name == "interpolate_points_to_image":
        from .image import interpolate_points_to_image

        return interpolate_points_to_image
    raise AttributeError(name)
