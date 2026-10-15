"""Lectura de config.yaml. Las rutas relativas se resuelven desde la raíz del repo."""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config.yaml"


def load_config(path=CONFIG_PATH):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    for key in ("DATA_DIR", "OUTPUT_DIR"):
        p = Path(cfg[key])
        cfg[key] = p if p.is_absolute() else REPO_ROOT / p

    return cfg
