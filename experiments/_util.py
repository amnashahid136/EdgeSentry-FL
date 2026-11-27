import json
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import RESULTS_DIR  # noqa: E402


def load_json(name, default):
    p = os.path.join(RESULTS_DIR, name)
    return json.load(open(p)) if os.path.exists(p) else default


def save_json(name, obj):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, name), "w") as f:
        json.dump(obj, f, indent=1)


def ms(vals):
    v = np.asarray(vals, dtype=float)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "n": int(len(v))}
