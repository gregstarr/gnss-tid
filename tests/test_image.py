from contextlib import nullcontext
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import xarray

from gnss_tid import image, parallel_logging
from gnss_tid.pointdata import TimeWindow


@pytest.fixture
def cube_inputs(monkeypatch):
    """Return cube arguments and mocked retrieval for a small spatial grid.

    Args:
        monkeypatch: Pytest fixture used to replace data retrieval.

    Returns:
        Arguments with two windows and heights, plus the retrieval mock.
        The image-maker returns image and density on (y, x) with grid coords.
    """
    times = np.array(["2020-01-01T00:01", "2020-01-01T00:00"], dtype="datetime64[m]")
    windows = [
        TimeWindow(t, t + np.timedelta64(1, "m"), np.array([i]))
        for i, t in enumerate(times)
    ]

    def fetch(obs, rx, ts, height, lat_limits, lon_limits, proj):
        """Return projected points encoding the input window and height.

        Args:
            obs, rx: Unused observation and receiver tables.
            ts: Window whose row index identifies the test time.
            height: Height to encode in the TEC values.
            lat_limits, lon_limits, proj: Unused retrieval bounds and projection.

        Returns:
            DataFrame with x, y, and tec columns for interpolation.
        """
        return pd.DataFrame({"x": [0.0], "y": [0.0], "tec": [height + ts.row_indices[0]]})

    def make_image(x, y, tec):
        """Return image and density on a fixed (y, x) grid.

        Args:
            x, y: Unused projected point coordinates.
            tec: TEC values used to fill the image.

        Returns:
            Dataset with image and density on (y, x), with matching coordinates.
        """
        return xarray.Dataset(
            {
                "image": (("y", "x"), np.full((2, 3), tec[0])),
                "density": (("y", "x"), np.ones((2, 3), dtype=int)),
            },
            coords={"y": [0.0, 1.0], "x": [0.0, 1.0, 2.0]},
        )

    retrieval = Mock(side_effect=fetch)
    monkeypatch.setattr(image, "get_data", retrieval)
    return {
        "obs": pd.DataFrame(),
        "rx": pd.DataFrame(),
        "image_maker": make_image,
        "time_windows": windows,
        "heights": [350.0, 250.0],
        "tec_name": "tec",
        "lat_limits": (0.0, 1.0),
        "lon_limits": (0.0, 1.0),
        "proj": None,
        "n_jobs": 1,
    }, retrieval


@pytest.mark.parametrize("n_times, n_heights", [(1, 1), (1, 2), (2, 1), (2, 2)])
def test_generate_image_cube_builds_all_combinations(cube_inputs, n_times, n_heights):
    """Verify every pair is generated and input axis order is preserved.

    Args:
        cube_inputs: Cube arguments and retrieval mock.
        n_times: Number of requested windows.
        n_heights: Number of requested heights.

    Returns:
        None; asserts cube coordinates, dimensions, values, and task count.
    """
    kwargs, retrieval = cube_inputs
    kwargs["time_windows"] = kwargs["time_windows"][:n_times]
    kwargs["heights"] = kwargs["heights"][:n_heights]
    cube = image.generate_image_cube(**kwargs)

    assert retrieval.call_count == n_times * n_heights
    assert cube.image.dims == cube.density.dims == ("time", "height", "y", "x")
    assert dict(cube.sizes) == {"time": n_times, "height": n_heights, "y": 2, "x": 3}
    np.testing.assert_array_equal(
        cube.time, [ts.start_time for ts in kwargs["time_windows"]]
    )
    np.testing.assert_array_equal(cube.height, kwargs["heights"])
    expected = np.array([[350, 250], [351, 251]])[:n_times, :n_heights]
    np.testing.assert_array_equal(cube.image[:, :, 0, 0], expected)
    assert (cube.density == 1).all()


@pytest.mark.parametrize("valid_indices", [[0, 1, 2], [0, 1], [0, 2], [0]])
def test_generate_image_cube_preserves_failed_slices(cube_inputs, valid_indices):
    """Verify missing pairs, entire times, and entire heights remain as NaN.

    Args:
        cube_inputs: Cube arguments and retrieval mock.
        valid_indices: Indices of successful time-height pairs.

    Returns:
        None; asserts valid data and NaN values at missing pairs.
    """
    kwargs, retrieval = cube_inputs
    fetch = retrieval.side_effect
    retrieval.side_effect = [
        fetch(None, None, ts, height, None, None, None) if i in valid_indices else None
        for i, (ts, height) in enumerate(
            (ts, h) for ts in kwargs["time_windows"] for h in kwargs["heights"]
        )
    ]

    cube = image.generate_image_cube(**kwargs)

    assert cube.image.shape == (2, 2, 2, 3)
    for i in range(4):
        selected = cube.isel(time=i // 2, height=i % 2)
        if i in valid_indices:
            assert selected.image.notnull().all()
            assert (selected.density == 1).all()
        else:
            assert selected.image.isnull().all()
            assert selected.density.isnull().all()


def test_generate_image_cube_raises_when_all_slices_fail(cube_inputs):
    """Verify an entirely failed cube raises instead of returning an empty grid.

    Args:
        cube_inputs: Cube arguments and retrieval mock.

    Returns:
        None; asserts the all-failed error.
    """
    kwargs, retrieval = cube_inputs
    retrieval.side_effect = [None] * 4
    with pytest.raises(RuntimeError, match="no valid slices"):
        image.generate_image_cube(**kwargs)


@pytest.mark.parametrize(
    "heights", [[], 350.0, [[250.0]], [250.0, 250.0], [np.nan], [np.inf]]
)
def test_generate_image_cube_rejects_invalid_heights(cube_inputs, heights):
    """Verify invalid height axes are rejected before scheduling work.

    Args:
        cube_inputs: Cube arguments and retrieval mock.
        heights: Invalid height input to validate.

    Returns:
        None; asserts validation fails without fetching data.
    """
    kwargs, retrieval = cube_inputs
    kwargs["heights"] = heights
    with pytest.raises(ValueError, match="heights"):
        image.generate_image_cube(**kwargs)
    retrieval.assert_not_called()


@pytest.mark.parametrize("duplicate", [False, True])
def test_generate_image_cube_rejects_invalid_windows(cube_inputs, duplicate):
    """Verify empty and duplicate time axes are rejected before scheduling work.

    Args:
        cube_inputs: Cube arguments and retrieval mock.
        duplicate: Whether to test duplicate windows rather than an empty list.

    Returns:
        None; asserts validation fails without fetching data.
    """
    kwargs, retrieval = cube_inputs
    kwargs["time_windows"] = [kwargs["time_windows"][0]] * 2 if duplicate else []
    with pytest.raises(ValueError, match="time_windows"):
        image.generate_image_cube(**kwargs)
    retrieval.assert_not_called()


def test_generate_image_stack_keeps_paired_height_semantics(cube_inputs):
    """Verify the existing stack still generates one paired height per window.

    Args:
        cube_inputs: Stack-compatible arguments and retrieval mock.

    Returns:
        None; asserts paired task count, dimensions, and image values.
    """
    kwargs, retrieval = cube_inputs
    stack = image.generate_image_stack(**kwargs)
    assert retrieval.call_count == 2
    assert stack.image.dims == ("time", "y", "x")
    assert stack.height.dims == ("time",)
    np.testing.assert_array_equal(stack.image[:, 0, 0], [350, 251])


@pytest.mark.parametrize("kind, total", [("stack", 2), ("cube", 4)])
@pytest.mark.parametrize("pbar", [False, True])
@pytest.mark.parametrize("failures", ["none", "some", "all"])
def test_image_progress(cube_inputs, monkeypatch, kind, total, pbar, failures):
    """Verify optional progress counts completed work and closes on failure.

    Args:
        cube_inputs: Image generation arguments and retrieval mock.
        monkeypatch: Pytest fixture replacing the tqdm display.
        kind: Stack or cube API to exercise.
        total: Expected number of scheduled slices.
        pbar: Whether to enable the display.
        failures: Whether no, some, or all retrievals return None.

    Returns:
        None; asserts bar construction, updates, closure, and output equivalence.
    """
    kwargs, retrieval = cube_inputs
    fetch = retrieval.side_effect
    if failures != "none":
        pairs = (
            list(zip(kwargs["time_windows"], kwargs["heights"], strict=True))
            if kind == "stack"
            else [(ts, h) for ts in kwargs["time_windows"] for h in kwargs["heights"]]
        )
        retrieval.side_effect = [
            None
            if failures == "all" or i == 0
            else fetch(None, None, ts, height, None, None, None)
            for i, (ts, height) in enumerate(pairs)
        ]

    tqdm = Mock()
    monkeypatch.setattr(parallel_logging, "tqdm", tqdm)
    generate = getattr(image, f"generate_image_{kind}")
    if failures == "all":
        with pytest.raises(RuntimeError, match="no valid slices"):
            generate(**kwargs, pbar=pbar)
    else:
        result = generate(**kwargs, pbar=pbar)
        retrieval.reset_mock()
        retrieval.side_effect = fetch
        expected = generate(**kwargs)
        if failures == "some":
            if kind == "stack":
                expected = expected.isel(time=slice(1, None))
            else:
                expected = expected.astype(float)
                expected["image"][0, 0] = np.nan
                expected["density"][0, 0] = np.nan
        xarray.testing.assert_equal(result, expected)

    if pbar:
        tqdm.assert_called_once_with(total=total, desc=f"Generating image {kind}")
        bar = tqdm.return_value
        assert sum(call.args[0] for call in bar.update.call_args_list) == total
        bar.close.assert_called_once_with()
    else:
        tqdm.assert_not_called()


@pytest.mark.parametrize("pbar", [None, False, True])
def test_worker_channels_optional_progress(monkeypatch, pbar):
    """Verify logging always stays enabled and progress defaults to enabled.

    Args:
        monkeypatch: Pytest fixture replacing queue listeners.
        pbar: Explicit progress setting, or None to use the default.

    Returns:
        None; asserts queue selection and disabled progress reports are no-ops.
    """
    log_queue, progress_queue = Mock(), Mock()
    logging = Mock(return_value=nullcontext(log_queue))
    progress = Mock(return_value=nullcontext(progress_queue))
    monkeypatch.setattr(parallel_logging, "log_queue_listener", logging)
    monkeypatch.setattr(parallel_logging, "progress_queue_listener", progress)
    kwargs = {} if pbar is None else {"pbar": pbar}

    with parallel_logging.worker_channels(2, "test", **kwargs) as ch:
        assert ch.log_queue is log_queue
        assert ch.progress_queue is (None if pbar is False else progress_queue)
        ch.report()

    logging.assert_called_once_with()
    if pbar is False:
        progress.assert_not_called()
        progress_queue.put.assert_not_called()
    else:
        progress.assert_called_once_with(2, "test")
        progress_queue.put.assert_called_once_with(1)
