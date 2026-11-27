"""
explainability.py
-------------------
Explainable-AI (XAI) toolkit for EdgeSentry (CLO3: "explainability & trust in
edge AI, interpretability-latency trade-offs"). Everything is built on the
FP32 Keras model: gradient-based XAI needs a differentiable model, so the
realistic design is "decide on the edge with the INT8 model, explain in the
cloud on demand with the FP32 model".

Four attribution methods, cheapest to most expensive:
    saliency     |d p_class / d x|                        1 backward pass
    smoothgrad   saliency averaged over noisy copies     ~20 passes
    ig           Integrated Gradients (Sundararajan '17)  ~32 passes
    occlusion    drop in p_class when a window is zeroed  ~63 forward passes

Because explanations can look convincing and still be wrong, we also MEASURE
them with a deletion test: remove the segments an explanation calls most
important and see how fast the model's confidence collapses. A faithful
method makes confidence fall much faster than removing random segments.

Run:  python model/explainability.py        (writes results/xai_report.json + 2 PNGs)
"""
import json
import os
import sys
import time

import numpy as np
import tensorflow as tf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from common import CLASSES, CLASS_LABELS, MODEL_DIR, RESULTS_DIR, DATA_DIR, GLOBAL_MODEL_FILE, WINDOW_LEN  # noqa: E402

SEG = 32                       # segment length (samples) used for deletion/aggregation
METHODS = ["saliency", "smoothgrad", "ig", "occlusion"]
METHOD_LABELS = {"saliency": "Gradient saliency", "smoothgrad": "SmoothGrad",
                 "ig": "Integrated Gradients", "occlusion": "Occlusion"}


def _probs(model, X):
    return model(tf.convert_to_tensor(X, dtype=tf.float32), training=False).numpy()


def _grad_of_class(model, X, cls):
    X = tf.convert_to_tensor(X, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(X)
        p = model(X, training=False)[:, cls]
    return tape.gradient(p, X).numpy()[..., 0]          # (N, L)


def saliency(model, x, cls):
    return np.abs(_grad_of_class(model, x[None], cls)[0])


def smoothgrad(model, x, cls, n=20, sigma=0.15, seed=0):
    rng = np.random.default_rng(seed)
    noisy = x[None] + rng.normal(0, sigma, size=(n,) + x.shape).astype(np.float32)
    return np.abs(_grad_of_class(model, noisy, cls)).mean(axis=0)


def integrated_gradients(model, x, cls, steps=32):
    alphas = np.linspace(0, 1, steps, dtype=np.float32)[:, None, None]
    path = alphas * x[None]                              # baseline = 0 (the signal mean)
    g = _grad_of_class(model, path, cls).mean(axis=0)
    return np.abs(g * x[:, 0])


def occlusion(model, x, cls, win=SEG, stride=SEG // 2):
    L = len(x)
    starts = list(range(0, L - win + 1, stride))
    batch = np.repeat(x[None], len(starts), axis=0)
    for i, s in enumerate(starts):
        batch[i, s:s + win, 0] = 0.0
    base = _probs(model, x[None])[0, cls]
    drops = np.maximum(base - _probs(model, batch)[:, cls], 0)
    attr, cnt = np.zeros(L), np.zeros(L)
    for d, s in zip(drops, starts):
        attr[s:s + win] += d
        cnt[s:s + win] += 1
    return attr / np.maximum(cnt, 1)


_FN = {"saliency": saliency, "smoothgrad": smoothgrad, "ig": integrated_gradients, "occlusion": occlusion}


def attribute(model, x, method, cls):
    return _FN[method](model, x, cls)


def segment_scores(attr, seg=SEG):
    return attr[: (len(attr) // seg) * seg].reshape(-1, seg).sum(axis=1)


def deletion_curve(model, x, cls, attr, n_points=17, n_random=8, seed=0):
    """Confidence in `cls` after removing the top-f fraction of segments
    (ranked by `attr`) vs. removing random segments. Returns fractions and
    both curves; a faithful attribution drops much faster than random."""
    seg_sc = segment_scores(attr)
    n_seg = len(seg_sc)
    fracs = np.linspace(0, 1, n_points)
    rng = np.random.default_rng(seed)

    def curve(order):
        batch = np.repeat(x[None], len(fracs), axis=0)
        for i, f in enumerate(fracs):
            for s in order[: int(round(f * n_seg))]:
                batch[i, s * SEG:(s + 1) * SEG, 0] = 0.0
        return _probs(model, batch)[:, cls]

    guided = curve(np.argsort(-seg_sc))
    rand = np.mean([curve(rng.permutation(n_seg)) for _ in range(n_random)], axis=0)
    return fracs, guided, rand


def explain(model, x, method="ig", cls=None):
    """Full explanation payload for one window (used by the website API)."""
    probs = _probs(model, x[None])[0]
    cls = int(np.argmax(probs)) if cls is None else int(cls)
    t0 = time.perf_counter()
    attr = attribute(model, x, method, cls)
    ms = (time.perf_counter() - t0) * 1000
    fr, g, r = deletion_curve(model, x, cls, attr)
    return {"method": method, "pred": cls, "probs": probs.tolist(), "attribution": attr.tolist(),
            "ms": ms, "deletion": {"fractions": fr.tolist(), "guided": g.tolist(), "random": r.tolist()},
            "seg": SEG}


# ------------------------------------------------------------------ report
def _spearman(a, b):
    from scipy.stats import spearmanr
    r = spearmanr(a, b).correlation
    return 0.0 if np.isnan(r) else float(r)


def build_report(model, X, y, n_per_class=25, seed=0):
    rng = np.random.default_rng(seed)
    preds = np.argmax(_probs(model, X), axis=1)
    idx = []
    for c in range(len(CLASSES)):
        pool = np.where((y == c) & (preds == c))[0]          # only correctly-classified windows
        idx += list(rng.choice(pool, size=min(n_per_class, len(pool)), replace=False))
    report = {"n_windows": len(idx), "seg": SEG, "methods": {}, "impulse_correlation": {}}
    curves = {m: [] for m in METHODS}
    rand_curves, times = [], {m: [] for m in METHODS}
    corr = {c: [] for c in CLASSES}
    fr = None
    for k, i in enumerate(idx):
        x, c = X[i], int(y[i])
        env = np.abs(x[: (len(x) // SEG) * SEG, 0]).reshape(-1, SEG).max(axis=1)   # local impulse amplitude
        for m in METHODS:
            t0 = time.perf_counter()
            a = attribute(model, x, m, c)
            times[m].append((time.perf_counter() - t0) * 1000)
            fr, g, r = deletion_curve(model, x, c, a, n_random=4)
            curves[m].append(g)
            if m == METHODS[0]:
                rand_curves.append(r)
            if m == "ig":
                corr[CLASSES[c]].append(_spearman(segment_scores(a), env))
        if k % 20 == 0:
            print(f"  XAI report: {k}/{len(idx)} windows", flush=True)
    rand = np.mean(rand_curves, axis=0)
    report["fractions"] = fr.tolist()
    report["random_curve"] = rand.tolist()
    for m in METHODS:
        mc = np.mean(curves[m], axis=0)
        report["methods"][m] = {
            "label": METHOD_LABELS[m], "ms_mean": float(np.mean(times[m])),
            "curve": mc.tolist(),
            "deletion_auc": float(np.trapezoid(mc, fr)), "random_auc": float(np.trapezoid(rand, fr)),
            "conf_after_10pct_removed": float(np.interp(0.1, fr, mc)),
            "random_conf_after_10pct_removed": float(np.interp(0.1, fr, rand)),
        }
    report["impulse_correlation"] = {c: float(np.mean(v)) for c, v in corr.items() if v}
    return report, idx


def make_figures(model, X, y, report, idx):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(3)
    fig, axes = plt.subplots(4, 1, figsize=(9, 8.5), sharex=True)
    for c, ax in enumerate(axes):
        pool = [i for i in idx if y[i] == c]
        i = int(rng.choice(pool))
        x = X[i]
        a = attribute(model, x, "ig", c)
        seg = segment_scores(a)
        ax.plot(x[:, 0], color="#2b6cb0", lw=0.8)
        for s, v in enumerate(seg / (seg.max() + 1e-12)):
            ax.axvspan(s * SEG, (s + 1) * SEG, color="orange", alpha=0.65 * float(v), lw=0)
        ax.set_ylabel(CLASS_LABELS[c], fontsize=8)
    axes[-1].set_xlabel("sample index in the 1024-sample window (orange = Integrated-Gradients importance)")
    fig.suptitle("Where the model looks (correctly classified test windows)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "explainability_saliency.png"), dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    fr = report["fractions"]
    for m in METHODS:
        ax.plot(np.array(fr) * 100, report["methods"][m]["curve"], label=METHOD_LABELS[m], lw=2)
    ax.plot(np.array(fr) * 100, report["random_curve"], "k--", label="random removal")
    ax.set_xlabel("% of signal removed (most-important segments first)")
    ax.set_ylabel("model confidence in its prediction")
    ax.set_title("Explanation faithfulness: deletion test")
    ax.grid(alpha=.3); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "xai_faithfulness.png"), dpi=140)
    plt.close(fig)


def main():
    model = tf.keras.models.load_model(os.path.join(MODEL_DIR, GLOBAL_MODEL_FILE))
    npz = np.load(os.path.join(DATA_DIR, "global_test.npz"))
    X, y = npz["X"].astype(np.float32), npz["y"]
    report, idx = build_report(model, X, y)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "xai_report.json"), "w") as f:
        json.dump(report, f, indent=1)
    make_figures(model, X, y, report, idx)
    print(f"{'method':22s}{'ms':>9s}{'AUC':>8s}{'conf@10%':>10s}")
    for m in METHODS:
        r = report["methods"][m]
        print(f"{r['label']:22s}{r['ms_mean']:9.1f}{r['deletion_auc']:8.3f}{r['conf_after_10pct_removed']:10.3f}")
    print(f"{'random removal':22s}{'':9s}{report['methods']['ig']['random_auc']:8.3f}"
          f"{report['methods']['ig']['random_conf_after_10pct_removed']:10.3f}")
    print("impulse correlation (IG vs local amplitude):", report["impulse_correlation"])


if __name__ == "__main__":
    main()
