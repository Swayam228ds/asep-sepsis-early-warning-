"""Shared helpers for the training scripts."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
RAW = ROOT / "data" / "raw"


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_jsonable))


def _jsonable(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(type(x))


def curve_points(xs, ys, n=100):
    """Downsample a curve to at most n points for the dashboard."""
    idx = np.linspace(0, len(xs) - 1, min(n, len(xs))).astype(int)
    return [[round(float(xs[i]), 4), round(float(ys[i]), 4)] for i in idx]
