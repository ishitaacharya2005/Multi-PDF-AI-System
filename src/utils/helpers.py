from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config_file() -> dict[str, Any]:
    config_path = project_root() / "config.yaml"
    if not config_path.is_file():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data


def ensure_directory(path: str | Path) -> str:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return str(directory)