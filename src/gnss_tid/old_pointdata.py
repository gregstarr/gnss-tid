from __future__ import annotations

import logging

from .pointdata import (
    PointData as _PointData,
    get_data,
    get_time_slices,
    load_observations,
)

LOGGER = logging.getLogger(__name__)


class PointData(_PointData):
    """Compatibility wrapper for legacy imports.

    The legacy module now delegates to the unified functional observation loader.
    """

    def __init__(
        self,
        files,
        latitude_limits,
        longitude_limits,
        time_limits,
        missing_data_threshold: float = 0.9,
        flatten_method: str = "mean",
    ):
        del flatten_method
        LOGGER.warning(
            "gnss_tid.old_pointdata.PointData is deprecated; use "
            "gnss_tid.pointdata.PointData"
        )
        super().__init__(
            files,
            latitude_limits,
            longitude_limits,
            time_limits,
            q_thresh=missing_data_threshold,
        )


__all__ = ["PointData", "get_data", "get_time_slices", "load_observations"]
