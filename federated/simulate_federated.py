"""
simulate_federated.py
----------------------
Official federated-training run (FedAvg, or DP-FedAvg with --dp) across the
3 simulated edge nodes. Produces the global model that gets compressed and
deployed to the edge nodes.

Each round: server sends global weights -> every node trains on ITS OWN
data only (non-IID: one defect severity per node) -> nodes return weights
(never data) -> server averages them (weighted by local sample count).

Run:
    python federated/simulate_federated.py --rounds 20 --local_epochs 2 --seed 0
    python federated/simulate_federated.py --dp --dp_clip_norm 1.0 --dp_noise_multiplier 0.1
Outputs:
    saved_models/global_model.keras
    results/federated_log.csv           (per-round accuracy, F1, per-node accuracy)
    results/federated_training_curve.png
    results/official_metrics.json       (confusion matrix + per-class precision/recall)
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import MODEL_DIR, RESULTS_DIR, CLASSES, GLOBAL_MODEL_FILE, MACHINES  # noqa: E402
from federated.fl_core import (train_federated, load_clients, load_global_test,  # noqa: E402
                               load_node_tests, evaluate)
from model.architecture import build_model  # noqa: E402


def per_class_report(cm):
    cm = np.array(cm)
    rep = {}
    for i, c in enumerate(CLASSES):
        tp = cm[i, i]
        rep[c] = {"precision": float(tp / max(cm[:, i].sum(), 1)),
                  "recall": float(tp / max(cm[i, :].sum(), 1)),
                  "support": int(cm[i, :].sum())}
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--local_epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr_start", type=float, default=1e-3, help="client LR in round 1 (cosine-decayed)")
    ap.add_argument("--lr_end", type=float, default=5e-5, help="client LR in the last round")
    ap.add_argument("--dp", action="store_true", help="Enable DP-FedAvg (clip + Gaussian noise).")
    ap.add_argument("--dp_clip_norm", type=float, default=1.0)
    ap.add_argument("--dp_noise_multiplier", type=float, default=0.1)
    ap.add_argument("--dp_delta", type=float, default=1e-5)
    ap.add_argument("--n_clients", type=int, default=3, help="3 (one per machine) or a multiple of 3")
    a = ap.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    clients = load_clients(a.n_clients, seed=a.seed)
    ev = load_global_test()
    dp = {"clip": a.dp_clip_norm, "sigma": a.dp_noise_multiplier, "delta": a.dp_delta} if a.dp else None
    mode = f"DP-FedAvg(clip={a.dp_clip_norm}, sigma={a.dp_noise_multiplier})" if a.dp else "FedAvg"
    print(f"Starting federated training [{mode}]: {len(clients)} clients, {a.rounds} rounds")

    res = train_federated(build_model, clients, rounds=a.rounds, local_epochs=a.local_epochs,
                          batch_size=a.batch_size, seed=a.seed, eval_set=ev,
                          node_tests=load_node_tests(), dp=dp, tag=mode,
                          lr_start=a.lr_start, lr_end=a.lr_end)
    model, hist = res["model"], res["history"]

    if a.dp:   # DP run: do not overwrite the official non-private model
        out_model, log_name, plot_name = "dp_demo_model.keras", "dp_demo_log.csv", "dp_demo_training_curve.png"
    else:
        out_model, log_name, plot_name = GLOBAL_MODEL_FILE, "federated_log.csv", "federated_training_curve.png"
    model.save(os.path.join(MODEL_DIR, out_model))
    print(f"Saved model -> saved_models/{out_model}")

    with open(os.path.join(RESULTS_DIR, log_name), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["round", "global_acc", "global_macro_f1"] + [f"acc_{m}" for m in MACHINES]
                   + (["mean_update_norm", "epsilon_rdp"] if a.dp else []))
        for h in hist:
            w.writerow([h["round"], f"{h['acc']:.4f}", f"{h['f1']:.4f}"]
                       + [f"{h['per_node'][m]:.4f}" for m in MACHINES]
                       + ([f"{h['update_norm']:.4f}", f"{h['epsilon']:.2f}"] if a.dp else []))

    if not a.dp:
        final = evaluate(model, *ev)
        metrics = {"seed": a.seed, "rounds": a.rounds, "local_epochs": a.local_epochs,
                   "acc": final["acc"], "macro_f1": final["f1"], "confusion_matrix": final["cm"],
                   "classes": CLASSES, "per_class": per_class_report(final["cm"]),
                   "per_node_acc": hist[-1]["per_node"], "history": hist}
        with open(os.path.join(RESULTS_DIR, "official_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=1)
        print(f"Final global test accuracy {final['acc']:.4f}  macro-F1 {final['f1']:.4f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x = [h["round"] for h in hist]
        plt.figure(figsize=(6.5, 4))
        plt.plot(x, [h["acc"] for h in hist], marker="o", lw=2, label="global model")
        for m in MACHINES:
            plt.plot(x, [h["per_node"][m] for h in hist], lw=1, alpha=.6, label=f"on {m} data")
        plt.xlabel("Federated round"); plt.ylabel("Test accuracy"); plt.ylim(0, 1.02)
        plt.title(f"EdgeSentry federated convergence ({mode})"); plt.grid(alpha=.3); plt.legend()
        plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, plot_name), dpi=150)
    except Exception as e:  # plotting is optional
        print("(skipped plot:", e, ")")


if __name__ == "__main__":
    main()
