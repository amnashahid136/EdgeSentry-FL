"""
fl_comparison.py
----------------
Multi-seed comparison that answers "what does federated learning actually buy
us?":

    local-only   each machine trains alone                (private, but poor on the rest of the fleet)
    FedAvg       federated learning, 3 nodes              (private, shared knowledge)
    centralized  all data pooled on one server            (best accuracy, NO privacy)

Every number is the mean +/- std over several random seeds (default 5), so a
single lucky run can't be mistaken for a result. Resumable: finished seeds
are skipped on re-run.   Output: results/exp_fl_comparison.json
"""
import argparse
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments._util import load_json, save_json, ms  # noqa: E402
from common import MODEL_DIR, MACHINES  # noqa: E402
from federated.fl_core import (train_federated, train_centralized, train_local_only,  # noqa: E402
                               load_clients, load_global_test, load_node_tests)
from model.architecture import build_model  # noqa: E402

OUT = "exp_fl_comparison.json"


def summarize(runs):
    seeds = sorted(runs, key=int)
    def col(f):
        return [f(runs[s]) for s in seeds]
    nodes = {m: ms(col(lambda r, m=m: r["local_only"][m]["acc"])) for m in MACHINES}
    own = {m: ms(col(lambda r, m=m: r["local_only"][m]["own_acc"])) for m in MACHINES}
    return {
        "centralized": {"acc": ms(col(lambda r: r["centralized"]["acc"])), "f1": ms(col(lambda r: r["centralized"]["f1"]))},
        "fedavg": {"acc": ms(col(lambda r: r["fedavg"]["acc"])), "f1": ms(col(lambda r: r["fedavg"]["f1"]))},
        "local_only": {
            "acc": ms(col(lambda r: sum(r["local_only"][m]["acc"] for m in MACHINES) / 3)),
            "f1": ms(col(lambda r: sum(r["local_only"][m]["f1"] for m in MACHINES) / 3)),
            "acc_per_node_on_fleet": nodes, "acc_on_own_data": own},
        "fedavg_per_node": {m: ms(col(lambda r, m=m: r["fedavg"]["per_node"][m])) for m in MACHINES},
        "n_seeds": len(seeds),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--local_epochs", type=int, default=2)
    a = ap.parse_args()

    data = load_json(OUT, {"runs": {}})
    ev, nt, clients = load_global_test(), load_node_tests(), load_clients(3)
    os.makedirs(os.path.join(MODEL_DIR, "seeds"), exist_ok=True)
    for s in a.seeds:
        if str(s) in data["runs"]:
            print(f"seed {s}: already done, skipping")
            continue
        print(f"=== seed {s} ===", flush=True)
        _, cen = train_centralized(build_model, epochs=20, seed=s, eval_set=ev)
        print(f"  centralized acc={cen['acc']:.4f}", flush=True)
        loc = train_local_only(build_model, epochs=20, seed=s, eval_set=ev)
        print("  local-only acc on fleet:", {m: round(v["acc"], 3) for m, v in loc.items()}, flush=True)
        fed = train_federated(build_model, clients, rounds=a.rounds, local_epochs=a.local_epochs, seed=s,
                              eval_set=ev, node_tests=nt, lr_start=1e-3, lr_end=5e-5, tag=f"FedAvg s{s}", verbose=False)
        h = fed["history"]
        fed["model"].save(os.path.join(MODEL_DIR, "seeds", f"fedavg_seed{s}.keras"))
        print(f"  fedavg acc={h[-1]['acc']:.4f}", flush=True)
        data["runs"][str(s)] = {
            "centralized": {"acc": cen["acc"], "f1": cen["f1"]},
            "local_only": {m: {k: v[k] for k in ("acc", "f1", "own_acc")} for m, v in loc.items()},
            "fedavg": {"acc": h[-1]["acc"], "f1": h[-1]["f1"], "per_node": h[-1]["per_node"],
                       "curve": [x["acc"] for x in h], "curve_f1": [x["f1"] for x in h]},
        }
        data["summary"] = summarize(data["runs"])
        data["config"] = {"rounds": a.rounds, "local_epochs": a.local_epochs, "centralized_epochs": 20,
                          "local_only_epochs": 20, "lr": "1e-3 -> 5e-5 cosine (FedAvg)"}
        save_json(OUT, data)
    print("done:", data["summary"]["fedavg"], data["summary"]["centralized"])


if __name__ == "__main__":
    main()
