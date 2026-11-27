"""
compression_study.py
--------------------
Multi-seed, FAIR compression comparison (CLO2/CLO3). For each seed's FedAvg
model we build:

    FP32 baseline            the trained model
    FP32 + same FL fine-tune CONTROL: gets the same extra federated rounds the
                             pruned models get, so a "pruning improved
                             accuracy" claim can't just be extra training
    INT8                     post-training quantization of the baseline
    Pruned 25/50/75% (+FT)   structured pruning, federated fine-tune, FP32
    Pruned 25/50/75% + INT8  pruning and quantization stacked

and measure size, params, MACs, RAM estimate, accuracy, macro-F1 and accuracy
under sensor noise. Needs experiments/fl_comparison.py to have run first.
Output: results/exp_compression.json   (resumable per seed)
"""
import argparse
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments._util import load_json, save_json, ms  # noqa: E402
from common import MODEL_DIR, add_awgn  # noqa: E402
from edge_node.engine import EdgeInferenceEngine  # noqa: E402
from federated.fl_core import (train_federated, load_clients, load_global_test, confusion_matrix, macro_f1)  # noqa: E402
from model.architecture import build_model, count_macs, peak_activation_elems  # noqa: E402
from model.compress_model import convert  # noqa: E402
from model.prune_model import prune_model, FT_LR  # noqa: E402

OUT = "exp_compression.json"
SNRS = [20, 10, 5, 0]
RATIOS = [0.25, 0.5, 0.75]


def cfg_of(m):
    g = m.get_layer
    return {"conv1": g("conv1").filters, "conv2": g("conv2").filters, "conv3": g("conv3").filters, "dense1": g("dense1").units}


def evaluate_all(predict_probs, X, y, noisy):
    pred = np.argmax(predict_probs(X), axis=1)
    cm = confusion_matrix(y, pred)
    rob = {str(s): float(np.mean(np.argmax(predict_probs(noisy[s]), axis=1) == y)) for s in SNRS}
    return {"acc": float(np.mean(pred == y)), "f1": macro_f1(cm), "robustness": rob}


def keras_fn(m):
    return lambda Z: m(tf.convert_to_tensor(Z), training=False).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--ft_rounds", type=int, default=4)
    a = ap.parse_args()

    data = load_json(OUT, {"runs": {}})
    ev = load_global_test()
    X, y = ev
    noisy = {s: add_awgn(X, s, np.random.default_rng(1)) for s in SNRS}
    clients = load_clients(3)
    work = os.path.join(MODEL_DIR, "seeds")

    for s in a.seeds:
        if str(s) in data["runs"]:
            print(f"seed {s}: done, skipping")
            continue
        src_path = os.path.join(work, f"fedavg_seed{s}.keras")
        if not os.path.exists(src_path):
            print(f"seed {s}: {src_path} missing (run fl_comparison.py first)")
            continue
        print(f"=== seed {s} ===", flush=True)
        base = tf.keras.models.load_model(src_path)
        rows = {}
        fp32_p, int8_p = convert(model=base, out_dir=work, out_prefix=f"s{s}_base", quiet=True)
        bcfg = cfg_of(base)
        r = evaluate_all(keras_fn(base), X, y, noisy)
        rows["FP32 baseline"] = dict(r, size_kb=os.path.getsize(fp32_p) / 1024, params=base.count_params(),
                                     macs=count_macs(bcfg), ram_kb=peak_activation_elems(bcfg) * 4 / 1024)
        eng = EdgeInferenceEngine(int8_p)
        r = evaluate_all(eng.predict_batch, X, y, noisy)
        rows["INT8 (quantized)"] = dict(r, size_kb=os.path.getsize(int8_p) / 1024, params=base.count_params(),
                                        macs=count_macs(bcfg), ram_kb=peak_activation_elems(bcfg) / 1024)
        # control: same extra federated rounds as pruned models get
        ctl = train_federated(lambda: build_model(bcfg), clients, rounds=a.ft_rounds, local_epochs=2, seed=s,
                              init_weights=base.get_weights(), lr_start=FT_LR[0], lr_end=FT_LR[1], verbose=False)
        r = evaluate_all(keras_fn(ctl["model"]), X, y, noisy)
        rows["FP32 + FL fine-tune (control)"] = dict(r, size_kb=rows["FP32 baseline"]["size_kb"], params=base.count_params(),
                                                      macs=count_macs(bcfg), ram_kb=rows["FP32 baseline"]["ram_kb"])
        for ratio in RATIOS:
            pm, cfg = prune_model(base, ratio)
            ft = train_federated(lambda cfg=cfg: build_model(cfg), clients, rounds=a.ft_rounds, local_epochs=2, seed=s,
                                 init_weights=pm.get_weights(), lr_start=FT_LR[0], lr_end=FT_LR[1], verbose=False)
            m = ft["model"]
            f32, i8 = convert(model=m, out_dir=work, out_prefix=f"s{s}_p{int(ratio*100)}", quiet=True)
            tag = f"Pruned {int(ratio*100)}%"
            r = evaluate_all(keras_fn(m), X, y, noisy)
            rows[f"{tag} (FP32)"] = dict(r, size_kb=os.path.getsize(f32) / 1024, params=m.count_params(),
                                         macs=count_macs(cfg), ram_kb=peak_activation_elems(cfg) * 4 / 1024)
            r = evaluate_all(EdgeInferenceEngine(i8).predict_batch, X, y, noisy)
            rows[f"{tag} + INT8"] = dict(r, size_kb=os.path.getsize(i8) / 1024, params=m.count_params(),
                                         macs=count_macs(cfg), ram_kb=peak_activation_elems(cfg) / 1024)
            print(f"  {tag}: params {m.count_params()}  fp32 acc {rows[f'{tag} (FP32)']['acc']:.3f}  "
                  f"int8 acc {rows[f'{tag} + INT8']['acc']:.3f}", flush=True)
        data["runs"][str(s)] = rows
        names = list(rows)
        data["summary"] = {n: {
            "size_kb": float(np.mean([data["runs"][k][n]["size_kb"] for k in data["runs"]])),
            "params": data["runs"][next(iter(data["runs"]))][n]["params"],
            "macs": data["runs"][next(iter(data["runs"]))][n]["macs"],
            "ram_kb": data["runs"][next(iter(data["runs"]))][n]["ram_kb"],
            "acc": ms([data["runs"][k][n]["acc"] for k in data["runs"]]),
            "f1": ms([data["runs"][k][n]["f1"] for k in data["runs"]]),
            "robustness": {str(sn): ms([data["runs"][k][n]["robustness"][str(sn)] for k in data["runs"]]) for sn in SNRS},
        } for n in names}
        data["snrs"], data["ft_rounds"] = SNRS, a.ft_rounds
        save_json(OUT, data)
    print("done")


if __name__ == "__main__":
    main()
