"""
Fast sanity tests (no training). Run:  python -m pytest tests -q     or     python tests/test_smoke.py
They guard the properties the project's claims rest on:
  * no train/test leakage in the dataset
  * the privacy accountant behaves (monotone, tighter than basic composition)
  * FedAvg arithmetic is right
  * model size/MAC bookkeeping matches the real network
  * the website serves every page and the command channel validates input
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def test_windows_are_leak_free():
    """Run the REAL window sampler on a signal whose every sample is unique
    (a ramp) and check that no sample value appears in both a training and a
    test window."""
    from data.prepare_dataset import _sample_windows
    ramp = np.arange(480_000, dtype=np.float64)
    xtr, _, xte, _ = _sample_windows(ramp, 1, n_train=400, n_test=100, rng=np.random.default_rng(0))
    train_vals, test_vals = set(xtr.astype(np.int64).ravel()), set(xte.astype(np.int64).ravel())
    assert len(xtr) == 400 and len(xte) == 100
    assert train_vals.isdisjoint(test_vals), "train and test windows share samples -> leakage"


def test_dataset_files_have_expected_shape():
    from common import MACHINES, WINDOW_LEN
    for m in MACHINES:
        tr = np.load(os.path.join(ROOT, "data", m, "train.npz"))
        assert tr["X"].shape[1:] == (WINDOW_LEN, 1)
        assert set(np.unique(tr["y"])) == {0, 1, 2, 3}
        # per-window z-normalisation really happened
        assert abs(float(tr["X"][0].mean())) < 1e-3 and abs(float(tr["X"][0].std()) - 1) < 1e-2


def test_rdp_accountant():
    from federated.dp_utils import rdp_epsilon, composed_epsilon
    assert rdp_epsilon(0, 10) == float("inf")
    assert rdp_epsilon(1.0, 5) < rdp_epsilon(1.0, 10) < rdp_epsilon(1.0, 20)       # more rounds -> more eps
    assert rdp_epsilon(2.0, 10) < rdp_epsilon(1.0, 10)                              # more noise -> less eps
    assert rdp_epsilon(1.0, 50) < composed_epsilon(1.0, 50)                         # tighter than linear composition


def test_fedavg_weighted_mean():
    from federated.fl_core import federated_average
    w = [[np.array([0.6])], [np.array([0.4])], [np.array([0.8])]]
    out = federated_average(w, [1600, 800, 2400])[0][0]
    assert abs(out - (1600 * 0.6 + 800 * 0.4 + 2400 * 0.8) / 4800) < 1e-6


def test_model_bookkeeping_matches_network():
    from model.architecture import build_model, BASE_CONFIG, count_macs
    m = build_model()
    assert m.count_params() == 11972
    small = {k: v // 2 for k, v in BASE_CONFIG.items()}
    assert count_macs(small) < count_macs() / 2


def test_pruned_model_is_smaller_and_runs():
    from model.architecture import build_model
    from model.prune_model import prune_model
    pm, cfg = prune_model(build_model(), 0.5)
    assert pm.count_params() < 4000 and cfg["conv3"] == 32
    assert pm.predict(np.zeros((2, 1024, 1), np.float32), verbose=0).shape == (2, 4)


def test_xai_methods_shapes():
    from model.architecture import build_model
    from model.explainability import METHODS, attribute, deletion_curve
    m = build_model()
    x = np.random.default_rng(0).normal(size=(1024, 1)).astype(np.float32)
    for meth in METHODS:
        a = attribute(m, x, meth, 1)
        assert a.shape == (1024,) and np.all(np.isfinite(a)) and a.min() >= 0
    fr, g, r = deletion_curve(m, x, 1, a)
    assert len(fr) == len(g) == len(r)


def test_website_pages_and_api():
    from cloud_aggregator.dashboard_app import app
    c = app.test_client()
    for p in ["/", "/explain", "/learning", "/compression", "/privacy", "/about", "/healthz", "/api/status"]:
        assert c.get(p).status_code == 200, p
    # command channel validates its input
    assert c.post("/api/command", json={"machine_id": "nope"}).status_code == 400
    assert c.post("/api/command", json={"machine_id": "edge01", "scenario": "bogus"}).status_code == 400
    assert c.post("/api/command", json={"machine_id": "edge01", "model": "huge"}).status_code == 400
    assert c.post("/api/command", json={"machine_id": "edge01", "scenario": "ball_fault"}).status_code == 200
    assert c.get("/api/command/edge01").get_json()["command"]["scenario"] == "ball_fault"
    assert c.get("/api/command/edge01").get_json()["command"] is None            # consumed once
    assert c.get("/api/results/../../etc/passwd").status_code == 404


def test_alarm_is_debounced():
    from cloud_aggregator import dashboard_app as d
    d.STATE.clear(); d.HISTORY.clear()
    def msg(i, status):
        return {"machine_id": "edge02", "timestamp": 1000.0 + i, "status": status, "prediction": "ball_fault" if status == "FAULT" else "normal",
                "class_id": 3 if status == "FAULT" else 0, "confidence": .99, "latency_ms": .1, "true_label": "normal"}
    for i, s in enumerate(["OK", "FAULT", "OK", "FAULT"]):          # 2 faults of 4 -> no alarm yet
        d.record_update(msg(i, s))
    assert not d.STATE["edge02"]["alarm"]
    d.record_update(msg(5, "FAULT"))                                # 3 of last 5 -> alarm
    assert d.STATE["edge02"]["alarm"]
    for i in range(6, 10):
        d.record_update(msg(i, "OK"))                                # recovers -> alarm clears
    assert not d.STATE["edge02"]["alarm"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f(); print("ok  ", f.__name__)
    print(f"\n{len(fns)} tests passed")
