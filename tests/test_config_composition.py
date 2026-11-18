"""Config composition tests for the focus group under event.

Compose the real configs in ``apps/conf`` with Hydra and verify that the
``focus`` config group now lives at ``cfg.event.focus``: each event config
loads the ``timesmooth`` focus preset by default, the preset's nested
``image_maker`` group resolves, and body/CLI overrides merge over the preset.
"""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

CONF_DIR = Path(__file__).resolve().parents[1] / "apps" / "conf"


def _compose(*overrides: str) -> DictConfig:
    """Compose ``config.yaml`` from ``apps/conf`` with optional overrides.

    Args:
        *overrides: Hydra command-line-style override strings, e.g.
            ``"event=20150325_full"`` or ``"event.focus.time_window=5"``.

    Returns:
        The composed :class:`~omegaconf.DictConfig`.
    """
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base=None):
        return compose(config_name="config", overrides=list(overrides))


def test_default_composes_event_focus():
    """Default config: timesmooth preset composes at event.focus.

    Verifies ``cfg.event.focus`` holds the timesmooth preset keys, the nested
    ``image_maker`` group resolved inside it, and no top-level ``focus`` key
    remains.
    """
    cfg = _compose()
    assert cfg.event.focus.time_window == 15
    assert cfg.event.focus.normalization == "image"
    assert cfg.event.focus.n_jobs == 12
    assert cfg.event.focus.image_maker._target_ == ("gnss_tid.image.ScipyRbfImageMaker")
    assert "focus" not in cfg


def test_focus_inherited_through_event_default():
    """``20150325_*`` events inherit the focus preset via ``event/default.yaml``."""
    cfg = _compose("event=20150325_full")
    assert cfg.event.focus.time_window == 15
    assert cfg.event.focus.image_maker is not None
    assert "focus" not in cfg


def test_cli_override_wins():
    """A command-line ``event.focus.*`` override beats the preset value."""
    cfg = _compose("event.focus.time_window=5")
    assert cfg.event.focus.time_window == 5


def test_event_body_override_wins(tmp_path):
    """An event yaml body ``focus:`` key beats the preset value."""
    conf_dir = tmp_path / "conf_root"
    (conf_dir / "event").mkdir(parents=True)
    (conf_dir / "event" / "focus").mkdir()
    for rel in [
        "config.yaml",
        "event/default.yaml",
        "event/focus/timesmooth.yaml",
        "event/focus/image_maker/rbf.yaml",
    ]:
        src = CONF_DIR / rel
        dst = conf_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    (conf_dir / "event" / "test_override.yaml").write_text(
        "defaults:\n"
        "  - focus: timesmooth\n"
        "  - _self_\n"
        "time_limits:\n"
        '  - "20210101_140000"\n'
        '  - "20210101_150000"\n'
        "focus:\n"
        "  time_window: 20\n",
        encoding="utf-8",
    )
    with initialize_config_dir(config_dir=str(conf_dir), version_base=None):
        cfg = compose(config_name="config", overrides=["event=test_override"])
    assert cfg.event.focus.time_window == 20
    # Non-overridden preset keys still come through.
    assert cfg.event.focus.normalization == "image"


@pytest.mark.parametrize("preset", ["spectral"])
def test_preset_switch(preset):
    """Switching the preset via ``event/focus=...`` picks up its keys."""
    cfg = _compose("event/focus=spectral")
    assert cfg.event.focus.time_window == 1
    assert cfg.event.focus.normalization is None
    # spectral preset also nests a center_finder group.
    assert cfg.event.focus.center_finder is not None
