"""
compare_dp.py
--------------
Differential-privacy study for DP-FedAvg (CLO3: security & privacy at the edge).

Question: what does client-level (epsilon, delta)-DP cost in accuracy, and how
does that cost change with the NUMBER OF CLIENTS?

Why the client count matters: DP noise is added to the SUM of client updates
and the result is divided by K, so the noise on the averaged update shrinks as
1/K while the (clipped) signal does not. With ~12K trainable weights, usable
accuracy needs  sigma  <~  K / sqrt(12000); a small fleet therefore cannot buy
meaningful privacy, a large one can. This script measures exactly that by
splitting the same data into K = 3, 12, 24 simulated clients.

Privacy accounting: Renyi-DP accountant for the Gaussian mechanism (full
participation, client-level add/remove adjacency) -- see federated/dp_utils.py.
The clip norm C is calibrated per K from the clients' own non-private update
norms in a short pilot run.

Run:   python federated/compare_dp.py --seeds 0 1
Output: results/exp_dp.json, results/dp_utility_tradeoff.{csv,md,png}   (resumable)
"""
import argparse
import csv
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from common import RESULTS_DIR  # noqa: E402
from experiments._util import load_json, save_json  # noqa: E402
from federated.fl_core import train_federated, load_clients, load_global_test  # noqa: E402
from federated.dp_utils import rdp_epsilon  # noqa: E402
from model.architecture import build_model  # noqa: E402

OUT = "exp_dp.json"
DELTA = 1e-5


def calibrate_clip(clients, seed):
    """Median per-client update norm over a short non-private pilot (3 rounds)."""
    r = train_federated(build_model, clients, rounds=3, local_epochs=2, seed=seed,
                        dp={"clip": 1e9, "sigma": 0.0, "delta": DELTA}, verbose=False)
    return float(np.median([h["update_norm"] for h in r["history"]]))


def write_outputs(data):
    rows = data["rows"]
    with open(os.path.join(RESULTS_DIR, "dp_utility_tradeoff.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clients", "noise_multiplier", "clip_norm", "epsilon_rdp", "acc_mean", "acc_std", "n_seeds"])
        for r in rows:
            a = np.array(r["accs"])
            w.writerow([r["K"], r["sigma"], f"{r['clip']:.3f}", "inf" if r["eps"] is None else f"{r['eps']:.1f}",
                        f"{a.mean():.4f}", f"{a.std(ddof=1) if len(a) > 1 else 0:.4f}", len(a)])
    with open(os.path.join(RESULTS_DIR, "dp_utility_tradeoff.md"), "w") as f:
        f.write("# EdgeSentry DP-FedAvg: privacy / utility trade-off\n\n")
        f.write(f"Rounds = {data['config']['rounds']}, delta = {DELTA}, RDP accountant (client-level). "
                "sigma = 0 means clipping only, no noise (epsilon = infinity).\n\n")
        f.write("| Clients | sigma | clip C | epsilon | accuracy (mean +/- std) |\n|---:|---:|---:|---:|---:|\n")
        for r in rows:
            a = np.array(r["accs"])
            eps = "inf" if r["eps"] is None else f"{r['eps']:.1f}"
            f.write(f"| {r['K']} | {r['sigma']} | {r['clip']:.2f} | {eps} | {a.mean():.3f} +/- "
                    f"{(a.std(ddof=1) if len(a) > 1 else 0):.3f} |\n")
        f.write("\nMore clients -> the same epsilon costs far less accuracy, because the noise on the "
                "averaged update shrinks as 1/K. With only 3 clients there is no useful privacy/utility "
                "operating point; this is a property of DP-FedAvg, not a bug.\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 4.5))
        for K in sorted({r["K"] for r in rows}):
            rr = sorted([r for r in rows if r["K"] == K and r["eps"] is not None], key=lambda r: r["eps"])
            plt.plot([r["eps"] for r in rr], [np.mean(r["accs"]) for r in rr], marker="o", label=f"{K} clients")
            base = [r for r in rows if r["K"] == K and r["eps"] is None]
            if base:
                plt.axhline(np.mean(base[0]["accs"]), ls=":", lw=1, color="gray")
        plt.xscale("log"); plt.xlabel("epsilon (RDP accountant, lower = stronger privacy)")
        plt.ylabel("global test accuracy"); plt.ylim(0, 1)
        plt.title("DP-FedAvg: privacy vs utility, by number of clients")
        plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "dp_utility_tradeoff.png"), dpi=150)
    except Exception as e:
        print("(plot skipped:", e, ")")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--clients", type=int, nargs="+", default=[3, 12, 24])
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 0.03, 0.1, 0.3, 1.0])
    ap.add_argument("--rounds", type=int, default=8)
    a = ap.parse_args()

    data = load_json(OUT, {"rows": [], "clips": {}})
    data["config"] = {"rounds": a.rounds, "delta": DELTA, "local_epochs": 2}
    ev = load_global_test()
    index = {(r["K"], r["sigma"]): r for r in data["rows"]}
    done = {(K, sg, s) for (K, sg), r in index.items() for s in r.get("seeds", [])}

    for s in a.seeds:
        for K in a.clients:
            clients = load_clients(K, seed=s)
            ck = f"{K}"
            if ck not in data["clips"]:
                data["clips"][ck] = calibrate_clip(clients, s)
                print(f"K={K}: calibrated clip C = {data['clips'][ck]:.3f}", flush=True)
                save_json(OUT, data)
            clip = data["clips"][ck]
            for sg in a.sigmas:
                if (K, sg, s) in done:
                    continue
                r = train_federated(build_model, clients, rounds=a.rounds, local_epochs=2, seed=s, eval_set=ev,
                                    dp={"clip": clip, "sigma": sg, "delta": DELTA}, verbose=False,
                                    lr_start=1e-3, lr_end=5e-5)
                acc, f1 = r["history"][-1]["acc"], r["history"][-1]["f1"]
                eps = rdp_epsilon(sg, a.rounds, DELTA) if sg > 0 else None
                row = index.get((K, sg))
                if row is None:
                    row = {"K": K, "sigma": sg, "clip": clip, "eps": eps, "accs": [], "f1s": [], "seeds": []}
                    data["rows"].append(row)
                    index[(K, sg)] = row
                row["accs"].append(acc); row["f1s"].append(f1); row["seeds"].append(s)
                print(f"seed {s} K={K:2d} sigma={sg:<5} eps={'inf' if eps is None else f'{eps:.1f}':>7}  acc={acc:.3f}", flush=True)
                data["rows"].sort(key=lambda r: (r["K"], r["sigma"]))
                save_json(OUT, data)
                write_outputs(data)
    write_outputs(data)
    print("done")


if __name__ == "__main__":
    main()
