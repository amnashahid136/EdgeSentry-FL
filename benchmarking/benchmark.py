"""
benchmark.py
-------------
Edge-AI benchmark (CLO3): every deployable variant of the model compared on
the SAME held-out test set, with the metrics that actually matter on a
constrained device:

    size (KB) | parameters | MACs/inference | est. peak activation RAM (KB)
    latency (mean / p95 ms, single thread) | accuracy | macro-F1
    accuracy under sensor noise (SNR 20 / 10 / 5 / 0 dB)

Variants: Keras FP32 (cloud baseline), TFLite FP32, TFLite INT8, Pruned Keras,
Pruned + INT8. Latency is measured on the machine running this script (single
thread) -- it ranks the variants against each other but is NOT a Raspberry Pi
number; see README "Honest limitations".

Outputs: results/benchmark_report.{csv,md,json}
"""
import json
import os
import sys
import time

import numpy as np
import tensorflow as tf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from common import MODEL_DIR, RESULTS_DIR, DATA_DIR, add_awgn  # noqa: E402
from edge_node.engine import EdgeInferenceEngine  # noqa: E402
from federated.fl_core import confusion_matrix, macro_f1  # noqa: E402
from model.architecture import BASE_CONFIG, count_macs, peak_activation_elems  # noqa: E402

SNRS = [None, 20, 10, 5, 0]


def cfg_of(keras_model):
    g = keras_model.get_layer
    return {"conv1": g("conv1").filters, "conv2": g("conv2").filters,
            "conv3": g("conv3").filters, "dense1": g("dense1").units}


def metrics(pred, y):
    cm = confusion_matrix(y, pred)
    return float(np.mean(pred == y)), macro_f1(cm)


def time_calls(fn, X, n=300, warmup=20):
    for i in range(warmup):
        fn(X[i % len(X)])
    ts = []
    for i in range(n):
        t0 = time.perf_counter()
        fn(X[i % len(X)])
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(ts)), float(np.percentile(ts, 50)), float(np.percentile(ts, 95))


def main():
    npz = np.load(os.path.join(DATA_DIR, "global_test.npz"))
    X, y = npz["X"].astype(np.float32), npz["y"]
    rng = np.random.default_rng(0)
    noisy = {s: add_awgn(X, s, np.random.default_rng(1)) for s in SNRS}

    base = tf.keras.models.load_model(os.path.join(MODEL_DIR, "global_model.keras"))
    pruned = tf.keras.models.load_model(os.path.join(MODEL_DIR, "pruned_model.keras"))
    variants = [
        ("Keras FP32 (cloud baseline)", "keras", base, "global_model.keras", base, 4),
        ("TFLite FP32", "tflite", None, "edge_model_fp32.tflite", base, 4),
        ("TFLite INT8 (quantized)", "tflite", None, "edge_model_int8.tflite", base, 1),
        ("Pruned Keras FP32", "keras", pruned, "pruned_model.keras", pruned, 4),
        ("Pruned + INT8 (both)", "tflite", None, "pruned_model_int8.tflite", pruned, 1),
    ]
    rows = []
    for name, kind, kmodel, fname, src, bytes_per in variants:
        path = os.path.join(MODEL_DIR, fname)
        if kind == "keras":
            fn_probs = lambda Z, m=kmodel: m(tf.convert_to_tensor(Z), training=False).numpy()
            one = lambda x, m=kmodel: m(tf.convert_to_tensor(x[None]), training=False).numpy()
        else:
            eng = EdgeInferenceEngine(path)
            fn_probs = lambda Z, e=eng: e.predict_batch(Z)
            one = lambda x, e=eng: e.probs(x)
        acc, f1 = metrics(np.argmax(fn_probs(X), axis=1), y)
        mean_ms, p50, p95 = time_calls(one, X)
        rob = {}
        for s in SNRS[1:]:
            rob[str(s)] = metrics(np.argmax(fn_probs(noisy[s]), axis=1), y)[0]
        cfg = cfg_of(src)
        rows.append({
            "name": name, "file": fname, "size_kb": os.path.getsize(path) / 1024,
            "params": int(src.count_params()), "macs": count_macs(cfg),
            "ram_kb": peak_activation_elems(cfg) * bytes_per / 1024,
            "latency_ms": mean_ms, "latency_p50_ms": p50, "latency_p95_ms": p95,
            "acc": acc, "f1": f1, "robustness": rob,
        })
        print(f"{name:30s} {rows[-1]['size_kb']:7.1f} KB  {rows[-1]['params']:6d} params  "
              f"{mean_ms:8.3f} ms  acc={acc:.3f}  f1={f1:.3f}  noise@5dB={rob['5']:.3f}", flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "benchmark_report.json"), "w") as f:
        json.dump({"rows": rows, "snrs": SNRS[1:], "n_test": int(len(y)),
                   "note": "Latency measured on the host CPU, 1 thread; ranks variants, not a Raspberry Pi number."},
                  f, indent=1)
    with open(os.path.join(RESULTS_DIR, "benchmark_report.csv"), "w") as f:
        f.write("model,size_kb,params,macs,ram_kb,latency_ms,latency_p95_ms,accuracy,macro_f1,acc_snr20,acc_snr10,acc_snr5,acc_snr0\n")
        for r in rows:
            f.write(f"{r['name']},{r['size_kb']:.2f},{r['params']},{r['macs']},{r['ram_kb']:.1f},{r['latency_ms']:.4f},"
                    f"{r['latency_p95_ms']:.4f},{r['acc']:.4f},{r['f1']:.4f},"
                    + ",".join(f"{r['robustness'][str(s)]:.4f}" for s in SNRS[1:]) + "\n")
    with open(os.path.join(RESULTS_DIR, "benchmark_report.md"), "w") as f:
        f.write("# EdgeSentry Benchmark Report (single run, official model)\n\n")
        f.write("| Model | Size KB | Params | MACs | Peak RAM KB | Latency ms (mean / p95) | Accuracy | Macro-F1 | Acc @5 dB noise |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(f"| {r['name']} | {r['size_kb']:.1f} | {r['params']} | {r['macs']/1e3:.0f}K | {r['ram_kb']:.1f} | "
                    f"{r['latency_ms']:.3f} / {r['latency_p95_ms']:.3f} | {r['acc']:.3f} | {r['f1']:.3f} | {r['robustness']['5']:.3f} |\n")
        b, e = rows[0], rows[-1]
        f.write(f"\nSize {b['size_kb']:.1f} KB -> {e['size_kb']:.1f} KB "
                f"({(1 - e['size_kb'] / b['size_kb']) * 100:.0f}% smaller); accuracy {b['acc']:.3f} -> {e['acc']:.3f}.\n")
        f.write("\nSingle-run numbers; see results/exp_compression.json for the 5-seed mean +/- std. "
                "Latency was measured on the host CPU (1 thread), not on Raspberry Pi hardware.\n")


if __name__ == "__main__":
    main()
