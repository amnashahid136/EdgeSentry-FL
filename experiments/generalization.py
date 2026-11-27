"""
generalization.py
-----------------
Leave-one-severity-out: train on two machines, test on the THIRD machine whose
defect severity the model has never seen. This is the harder, more honest test
of "will the model work on a new machine?" -- the standard split (test windows
from the same recordings/severities as training, just later in time) is
easier. Compares FedAvg and centralized training on the two seen machines.
Output: results/exp_generalization.json
"""
import argparse
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments._util import load_json, save_json, ms  # noqa: E402
from common import MACHINES, SEVERITY_BY_MACHINE  # noqa: E402
from federated.fl_core import train_federated, load_clients, load_split, evaluate, set_seed  # noqa: E402
from model.architecture import build_model  # noqa: E402

OUT = "exp_generalization.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--rounds", type=int, default=15)
    a = ap.parse_args()
    data = load_json(OUT, {"runs": {}})
    for s in a.seeds:
        if str(s) in data["runs"]:
            continue
        res = {}
        for held in MACHINES:
            seen = [c for c in load_clients(3) if c["machine"] != held]
            Xte, yte = load_split(held, "test")
            fed = train_federated(build_model, seen, rounds=a.rounds, local_epochs=2, seed=s,
                                  lr_start=1e-3, lr_end=5e-5, verbose=False)
            set_seed(s)
            cen = build_model()
            cen.fit(np.concatenate([c["X"] for c in seen]), np.concatenate([c["y"] for c in seen]),
                    epochs=20, batch_size=32, shuffle=True, verbose=0)
            res[held] = {"fedavg": evaluate(fed["model"], Xte, yte)["acc"], "centralized": evaluate(cen, Xte, yte)["acc"]}
            print(f"seed {s} held-out {held} ({SEVERITY_BY_MACHINE[held]}): {res[held]}", flush=True)
        data["runs"][str(s)] = res
        data["summary"] = {h: {k: ms([data["runs"][x][h][k] for x in data["runs"]]) for k in ("fedavg", "centralized")}
                           for h in MACHINES}
        save_json(OUT, data)


if __name__ == "__main__":
    main()
