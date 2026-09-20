"""Configuration loading utilities for the EquityLens pipeline.

The single source of truth for all pipeline parameters is ``config.yaml``.
This module exposes a thin, typed wrapper so that every downstream module
receives a consistent configuration object.
"""

from __future__ import annotations

import pathlib
from types import SimpleNamespace
from typing import Any, Dict

import yaml

# Project root = two levels above this file (src/ -> repo root)
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def _to_namespace(obj: Any) -> Any:
    """Recursively convert dicts into attribute-accessible namespaces."""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    return obj


def load_config(config_path: str | pathlib.Path = CONFIG_PATH) -> SimpleNamespace:
    """Load the YAML configuration file into a namespace object.

    Parameters
    ----------
    config_path:
        Path to the YAML configuration file. Defaults to the repo-level
        ``config.yaml``.

    Returns
    -------
    SimpleNamespace
        Nested configuration object with attribute access, e.g.
        ``cfg.data.start_date``.
    """
    config_path = pathlib.Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}. "
            "The pipeline cannot run without it."
        )
    with open(config_path, "r", encoding="utf-8") as fh:
        raw: Dict[str, Any] = yaml.safe_load(fh)
    return _to_namespace(raw)


def resolve_path(relative_path: str) -> pathlib.Path:
    """Resolve a repo-relative path (as used in config.yaml) to an absolute one.

    Parameters
    ----------
    relative_path:
        Path string as stored in the configuration file, e.g.
        ``"outputs/figures"``.

    Returns
    -------
    pathlib.Path
        Absolute path anchored at the project root.
    """
    path = pathlib.Path(relative_path)
    return path if path.is_absolute() else PROJECT_ROOT / path
