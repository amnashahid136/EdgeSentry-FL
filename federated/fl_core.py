"""
fl_core.py
----------
The one federated-learning engine used by every experiment in EdgeSentry
(official training run, multi-seed comparison, pruning fine-tune, DP sweep).

    train_federated(...)   FedAvg  or  DP-FedAvg (clip + Gaussian noise)
    train_centralized(...) pooled-data upper bound (breaks privacy; baseline only)
    train_local_only(...)  each node trains alone (what you get WITHOUT federation)

Privacy property being simulated: a "client" only ever hands the server its
updated WEIGHTS; its raw vibration windows are never read by the server or
by any other client. (Everything runs in one process here, but the data flow
is exactly the FL protocol -- see README "Honest limitations".)
"""
import os
import sys
import time

import numpy as np
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import DATA_DIR, MACHINES, NUM_CLASSES  # noqa: E402
from federated.dp_utils import clip_update, add_gaussian_noise, rdp_epsilon  # noqa: E402


# ----------------------------------------------------------------- metrics
def confusion_matrix(y_true, y_pred, k=NUM_CLASSES):
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm


def macro_f1(cm):
    f1s = []
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(f1s))


def evaluate(model, X, y):
    probs = model.predict(X, batch_size=256, verbose=0)
    pred = np.argmax(probs, axis=1)
    cm = confusion_matrix(y, pred)
    return {"acc": float(np.mean(pred == y)), "f1": macro_f1(cm), "cm": cm.tolist()}


# -------------------------------------------------------------------- data
def load_split(machine_id, split):
    npz = np.load(os.path.join(DATA_DIR, machine_id, f"{split}.npz"))
    return npz["X"].astype(np.float32), npz["y"]


def load_global_test():
    npz = np.load(os.path.join(DATA_DIR, "global_test.npz"))
    return npz["X"].astype(np.float32), npz["y"]


def load_node_tests():
    return {m: load_split(m, "test") for m in MACHINES}


def load_clients(n_clients=3, seed=0):
    """Return a list of client datasets. n_clients=3 -> one client per machine
    (the main setup). n_clients=3k -> each machine's data is split into k
    random shards, i.e. a larger simulated fleet (used to study DP scaling)."""
    assert n_clients % len(MACHINES) == 0, "n_clients must be a multiple of 3"
    k = n_clients // len(MACHINES)
    rng = np.random.default_rng(seed)
    clients = []
    for m in MACHINES:
        X, y = load_split(m, "train")
        if k == 1:
            clients.append({"name": m, "machine": m, "X": X, "y": y})
            continue
        idx = rng.permutation(len(X))
        for j, part in enumerate(np.array_split(idx, k)):
            clients.append({"name": f"{m}-{j}", "machine": m, "X": X[part], "y": y[part]})
    return clients


# ------------------------------------------------------------------ helpers
def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _reset_optimizer(model):
    """Fresh Adam state for every client (clients don't share optimizer state)."""
    for v in model.optimizer.variables:
        if "learning_rate" in v.path or "learning_rate" in v.name:
            continue
        v.assign(tf.zeros_like(v))


def federated_average(weight_list, sample_counts):
    total = float(sum(sample_counts))
    coeffs = np.array(sample_counts, dtype=np.float64) / total
    out = []
    for i in range(len(weight_list[0])):
        stack = np.stack([w[i] for w in weight_list])
        out.append(np.tensordot(coeffs, stack, axes=1).astype(stack.dtype))
    return out


def dp_federated_average(global_w, local_ws, sample_counts, clip, sigma, rng, trainable_mask):
    """DP-FedAvg (McMahan et al. 2018): clip each client's update to L2 norm
    `clip`, sum, add N(0, (sigma*clip)^2) noise, divide by #clients. Noise is
    applied only to trainable parameters; BatchNorm running statistics are
    averaged normally (they are batch statistics, not gradient-learned)."""
    tr = [i for i, t in enumerate(trainable_mask) if t]
    other = [i for i, t in enumerate(trainable_mask) if not t]
    clipped, norms = [], []
    for lw in local_ws:
        upd = [lw[i] - global_w[i] for i in tr]
        c, n = clip_update(upd, clip)
        clipped.append(c)
        norms.append(n)
    summed = [np.sum(np.stack([u[j] for u in clipped]), axis=0) for j in range(len(tr))]
    noised = add_gaussian_noise(summed, sigma, clip, rng)
    n = len(local_ws)
    new = list(global_w)
    for pos, i in enumerate(tr):
        new[i] = global_w[i] + noised[pos] / n
    if other:
        coeffs = np.array(sample_counts, dtype=np.float64) / float(sum(sample_counts))
        for i in other:
            stack = np.stack([lw[i] for lw in local_ws])
            new[i] = np.tensordot(coeffs, stack, axes=1).astype(stack.dtype)
    return new, norms


# ------------------------------------------------------------- FedAvg / DP
def train_federated(build_fn, clients, rounds=20, local_epochs=2, batch_size=32, seed=0,
                    eval_set=None, node_tests=None, init_weights=None, dp=None,
                    verbose=True, tag="FedAvg", lr_start=None, lr_end=None):
    """Run federated training and return {"weights", "history", "model"}.

    dp: None for plain FedAvg, or {"clip": C, "sigma": z, "delta": d}.
    lr_start/lr_end: optional cosine decay of the clients' learning rate across
    rounds (reduces client drift late in training, a standard FedAvg remedy).
    history rows: round, loss, acc, f1, per_node{machine: acc}, update_norm, epsilon.
    """
    set_seed(seed)
    rng = np.random.default_rng(seed)
    global_model = build_fn()
    local_model = build_fn()
    if init_weights is not None:
        global_model.set_weights(init_weights)
    global_w = global_model.get_weights()
    trainable_mask = [w.trainable for w in global_model.weights]

    history = []
    t0 = time.time()
    for rnd in range(1, rounds + 1):
        local_ws, counts = [], []
        if lr_start is not None:
            frac = (rnd - 1) / max(rounds - 1, 1)
            lr = (lr_end or lr_start) + 0.5 * (lr_start - (lr_end or lr_start)) * (1 + np.cos(np.pi * frac))
            local_model.optimizer.learning_rate.assign(lr)
        for c in clients:
            local_model.set_weights(global_w)          # 1. receive global model
            _reset_optimizer(local_model)
            local_model.fit(c["X"], c["y"], epochs=local_epochs, batch_size=batch_size,
                            shuffle=True, verbose=0)    # 2. train on LOCAL data only
            local_ws.append(local_model.get_weights())  # 3. send weights back (not data)
            counts.append(len(c["X"]))
        row = {"round": rnd}
        if dp:
            global_w, norms = dp_federated_average(global_w, local_ws, counts, dp["clip"],
                                                   dp["sigma"], rng, trainable_mask)
            row["update_norm"] = float(np.mean(norms))
            row["epsilon"] = rdp_epsilon(dp["sigma"], rnd, dp.get("delta", 1e-5))
        else:
            global_w = federated_average(local_ws, counts)  # 4. server aggregates
        global_model.set_weights(global_w)
        if eval_set is not None:
            ev = evaluate(global_model, *eval_set)
            row.update(acc=ev["acc"], f1=ev["f1"])
        if node_tests:
            row["per_node"] = {m: evaluate(global_model, *xy)["acc"] for m, xy in node_tests.items()}
        history.append(row)
        if verbose:
            extra = f"  eps={row['epsilon']:.1f}" if dp else ""
            print(f"  [{tag}] round {rnd:2d}/{rounds}  acc={row.get('acc', float('nan')):.4f}  "
                  f"f1={row.get('f1', float('nan')):.4f}{extra}  ({time.time() - t0:.0f}s)", flush=True)
    return {"weights": global_w, "history": history, "model": global_model}


# ---------------------------------------------------------------- baselines
def train_centralized(build_fn, epochs=20, batch_size=32, seed=0, eval_set=None):
    """Upper bound: pool every machine's training data on one server.
    NOT privacy-preserving -- exists only to measure the cost of federation."""
    set_seed(seed)
    Xs, ys = zip(*[load_split(m, "train") for m in MACHINES])
    model = build_fn()
    model.fit(np.concatenate(Xs), np.concatenate(ys), epochs=epochs, batch_size=batch_size,
              shuffle=True, verbose=0)
    return model, (evaluate(model, *eval_set) if eval_set else None)


def train_local_only(build_fn, epochs=20, batch_size=32, seed=0, eval_set=None):
    """Each machine trains alone on its own data (no federation). Returns the
    per-machine results on the GLOBAL test set (shows how badly a model that
    only ever saw one fault severity generalises to the rest of the fleet)."""
    out = {}
    for i, m in enumerate(MACHINES):
        set_seed(seed + 100 * i)
        X, y = load_split(m, "train")
        model = build_fn()
        model.fit(X, y, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=0)
        res = evaluate(model, *eval_set) if eval_set else {}
        res["own_acc"] = evaluate(model, *load_split(m, "test"))["acc"]   # on its OWN machine's test data
        out[m] = res
    return out
