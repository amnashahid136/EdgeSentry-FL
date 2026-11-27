"""
prune_model.py
---------------
Structured (channel/filter) pruning -- the SECOND compression technique next
to INT8 quantization. Quantization lowers the precision of every weight;
structured pruning physically removes whole filters/units, giving a smaller
DENSE model that needs no sparse kernels (the right choice for edge CPUs/MCUs).

Pipeline:
  1. Rank each layer's output channels by L1 weight magnitude.
  2. Build a narrower model (same `build_model`, smaller cfg) and transplant
     the strongest channels, propagating dependencies (pruning conv_i's
     outputs also prunes conv_{i+1}'s inputs and the BatchNorm params).
  3. Fine-tune to recover accuracy -- FEDERATEDLY (a few FedAvg rounds on the
     clients' own data), so the privacy story is not broken by pooling data
     centrally. (The previous version fine-tuned on pooled data, which also
     gave the pruned model extra training the baseline never got -- an unfair
     comparison. `fair_baseline_finetune` in experiments/ fixes that.)
  4. Export fp32 + INT8 TFLite.

Run:  python model/prune_model.py --prune_ratio 0.5 --finetune_rounds 4
"""
import argparse
import os
import sys

import numpy as np
import tensorflow as tf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from common import MODEL_DIR, RESULTS_DIR, GLOBAL_MODEL_FILE  # noqa: E402
from model.architecture import build_model, BASE_CONFIG  # noqa: E402
from model.compress_model import convert as quantize_convert  # noqa: E402


def pruned_config(prune_ratio):
    return {k: max(2, int(round(v * (1 - prune_ratio)))) for k, v in BASE_CONFIG.items()}


def _topk(scores, k):
    return np.sort(np.argsort(-scores)[:min(k, len(scores))])


def _l1(kernel, out_axis):
    return np.sum(np.abs(kernel), axis=tuple(a for a in range(kernel.ndim) if a != out_axis))


def prune_model(source_model, prune_ratio):
    """Return (pruned_model, cfg) warm-started from `source_model`."""
    cfg = pruned_config(prune_ratio)
    tgt = build_model(cfg, name=f"edgesentry_pruned_{int(prune_ratio * 100)}pct")
    g = source_model.get_layer

    k1, b1 = g("conv1").get_weights()
    i1 = _topk(_l1(k1, 2), cfg["conv1"])
    tgt.get_layer("conv1").set_weights([k1[:, :, i1], b1[i1]])
    tgt.get_layer("bn1").set_weights([w[i1] for w in g("bn1").get_weights()])

    k2, b2 = g("conv2").get_weights()
    k2 = k2[:, i1, :]
    i2 = _topk(_l1(k2, 2), cfg["conv2"])
    tgt.get_layer("conv2").set_weights([k2[:, :, i2], b2[i2]])
    tgt.get_layer("bn2").set_weights([w[i2] for w in g("bn2").get_weights()])

    k3, b3 = g("conv3").get_weights()
    k3 = k3[:, i2, :]
    i3 = _topk(_l1(k3, 2), cfg["conv3"])
    tgt.get_layer("conv3").set_weights([k3[:, :, i3], b3[i3]])
    tgt.get_layer("bn3").set_weights([w[i3] for w in g("bn3").get_weights()])

    d1, db1 = g("dense1").get_weights()
    d1 = d1[i3, :]
    id1 = _topk(np.sum(np.abs(d1), axis=0), cfg["dense1"])
    tgt.get_layer("dense1").set_weights([d1[:, id1], db1[id1]])

    do, dbo = g("out").get_weights()
    tgt.get_layer("out").set_weights([do[id1, :], dbo])
    return tgt, cfg


FT_LR = (3e-4, 5e-5)   # gentle client LR (start, end) for federated fine-tuning after pruning


def run(prune_ratio=0.5, finetune_rounds=4, local_epochs=2, model_path=None, out_dir=None,
        seed=0, quiet=False, save=True):
    from federated.fl_core import train_federated, load_clients, load_global_test, evaluate
    model_path = model_path or os.path.join(MODEL_DIR, GLOBAL_MODEL_FILE)
    out_dir = out_dir or MODEL_DIR
    src = tf.keras.models.load_model(model_path)
    ev = load_global_test()
    pre = evaluate(src, *ev)

    pruned, cfg = prune_model(src, prune_ratio)
    warm = evaluate(pruned, *ev)
    if not quiet:
        print(f"Pruning {int(prune_ratio*100)}% -> {cfg}; params {src.count_params()} -> {pruned.count_params()}")
        print(f"  accuracy: source {pre['acc']:.4f} | right after pruning {warm['acc']:.4f}")

    ft = train_federated(lambda: build_model(cfg, name="pruned_ft"), load_clients(3), rounds=finetune_rounds,
                         local_epochs=local_epochs, seed=seed, eval_set=ev,
                         init_weights=pruned.get_weights(), verbose=not quiet, tag="prune-FT",
                         lr_start=FT_LR[0], lr_end=FT_LR[1])
    final_model = ft["model"]
    post = evaluate(final_model, *ev)
    if not quiet:
        print(f"  accuracy after {finetune_rounds} federated fine-tune rounds: {post['acc']:.4f}")
    if not save:
        return final_model, cfg, {"pre": pre, "warm": warm, "post": post}

    pruned_path = os.path.join(out_dir, "pruned_model.keras")
    final_model.save(pruned_path)
    fp32_path, int8_path = quantize_convert(model_path=pruned_path, out_dir=out_dir,
                                            out_prefix="pruned_model", quiet=quiet)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "pruning_report.md"), "w") as f:
        f.write("# EdgeSentry Structured Pruning Report\n\n")
        f.write(f"- Prune ratio: {prune_ratio:.0%} fewer channels/units per layer\n")
        f.write(f"- Channels kept: {cfg} (base: {BASE_CONFIG})\n")
        f.write(f"- Parameters: {src.count_params()} -> {final_model.count_params()} "
                f"({(1 - final_model.count_params() / src.count_params()) * 100:.1f}% fewer)\n\n")
        f.write("| Stage | Accuracy |\n|---|---:|\n")
        f.write(f"| Source model (unpruned) | {pre['acc']:.4f} |\n")
        f.write(f"| Pruned, before fine-tuning | {warm['acc']:.4f} |\n")
        f.write(f"| Pruned, after {finetune_rounds} federated fine-tune rounds | {post['acc']:.4f} |\n\n")
        f.write("Fine-tuning is federated (no data pooling). For a *fair* comparison the unpruned "
                "baseline also receives the same extra federated rounds in experiments/compression_study.py.\n")
    return pruned_path, fp32_path, int8_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prune_ratio", type=float, default=0.5)
    ap.add_argument("--finetune_rounds", type=int, default=4)
    ap.add_argument("--local_epochs", type=int, default=2)
    a = ap.parse_args()
    run(a.prune_ratio, a.finetune_rounds, a.local_epochs)
