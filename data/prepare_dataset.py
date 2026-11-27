"""
prepare_dataset.py
-------------------
Loads the REAL CWRU (Case Western Reserve University) bearing-fault
vibration dataset and reshapes it into per-edge-node train/test shards for
the EdgeSentry predictive-maintenance project.

Source data:
    archive/raw/*.mat  -- raw 48kHz drive-end (DE) accelerometer recordings
    from the CWRU Bearing Data Center, one .mat file per fault type and
    per defect ("severity") diameter:
        Time_Normal_1_098.mat  -> healthy baseline
        B0{07,14,21}_1_*.mat   -> ball fault,        0.007" / 0.014" / 0.021"
        IR0{07,14,21}_1_*.mat  -> inner-race fault,  0.007" / 0.014" / 0.021"
        OR0{07,14,21}_6_1_*.mat-> outer-race fault,  0.007" / 0.014" / 0.021"
    All recorded at ~1772-1774 RPM (motor load 1).

Classes (unchanged from the previous synthetic generator):
    0 = Normal
    1 = Inner Race Fault
    2 = Outer Race Fault
    3 = Ball Fault

Each simulated edge node is assigned one real defect severity (007 / 014 /
021 mils), so the resulting per-machine data is naturally non-IID -- the
same setting the previous synthetic generator approximated, now backed by
real sensor recordings instead of simulated impulses. The single shared
"Normal" recording is split into disjoint segments so no raw sample is
reused across nodes.

LEAK-FREE SPLIT (v2): every recording is cut in TIME -- the first 70% of the
signal is used only for training windows, the last 30% only for test
windows, with a one-window gap in between. Training windows overlap by 50%
(data augmentation); test windows never overlap. This guarantees that no
test window shares a single sample with any training window, which the
original random-window sampling did not (neighbouring windows from the same
recording could land on both sides of the split).

Output (unchanged):
    data/edge0{1,2,3}/train.npz, test.npz   (per edge node)
    data/global_test.npz                     (held-out set for the cloud
                                               to evaluate the global model)
"""
import os
import re
import sys

import numpy as np
from scipy.io import loadmat

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import WINDOW_LEN, SAMPLE_RATE, CLASSES  # noqa: E402,F401

ARCHIVE_RAW_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "archive", "raw"
)
NORMAL_FILE = "Time_Normal_1_098.mat"

# Per-machine (per edge-node) real defect severity -> creates realistic non-IID data
MACHINE_PROFILES = [
    dict(machine_id="edge01", severity="007", inner="IR007_1_110.mat",
         outer="OR007_6_1_136.mat", ball="B007_1_123.mat"),   # mildest defects
    dict(machine_id="edge02", severity="014", inner="IR014_1_175.mat",
         outer="OR014_6_1_202.mat", ball="B014_1_190.mat"),   # moderate defects
    dict(machine_id="edge03", severity="021", inner="IR021_1_214.mat",
         outer="OR021_6_1_239.mat", ball="B021_1_227.mat"),   # most severe defects
]


def _load_de_signal(mat_filename):
    """Load the drive-end (DE) accelerometer channel from a raw CWRU .mat file."""
    path = os.path.join(ARCHIVE_RAW_DIR, mat_filename)
    mat = loadmat(path)
    num_match = re.search(r"_(\d+)\.mat$", mat_filename)
    key = f"X{num_match.group(1)}_DE_time" if num_match else None
    if key is None or key not in mat:
        # fallback: some files carry more than one *_DE_time key; pick the
        # one matching the filename, otherwise the only DE_time key present
        candidates = [k for k in mat.keys() if k.endswith("DE_time")]
        key = candidates[0]
    return mat[key].astype(np.float64).ravel()


TRAIN_FRACTION = 0.70      # first 70% of each recording -> training region
TRAIN_STRIDE = WINDOW_LEN // 2   # 50% overlap between training windows


def _windows_from_signal(sig, window_len=WINDOW_LEN, stride=None):
    """Chop a 1D signal into fixed-length windows (stride defaults to
    window_len, i.e. non-overlapping)."""
    stride = stride or window_len
    n_windows = (len(sig) - window_len) // stride + 1
    idx = np.arange(window_len)[None, :] + stride * np.arange(n_windows)[:, None]
    return sig[idx]


def _sample_windows(sig, label, n_train, n_test, rng, train_stride=None):
    """Draw train/test windows of one class from a real signal using a
    TEMPORAL split: training windows come only from the first 70% of the
    recording, test windows only from the last 30% (plus a 1-window gap)."""
    cut = int(len(sig) * TRAIN_FRACTION)
    train_region = sig[:cut]
    test_region = sig[cut + WINDOW_LEN:]
    tr_windows = _windows_from_signal(train_region, stride=train_stride or TRAIN_STRIDE)
    te_windows = _windows_from_signal(test_region, stride=WINDOW_LEN)
    tr_idx = rng.permutation(len(tr_windows))[:n_train]
    te_idx = rng.permutation(len(te_windows))[:n_test]
    xtr = tr_windows[tr_idx].astype(np.float32)
    xte = te_windows[te_idx].astype(np.float32)
    ytr = np.full(len(xtr), label, dtype=np.int64)
    yte = np.full(len(xte), label, dtype=np.int64)
    return xtr, ytr, xte, yte


def _znorm(X):
    """z-normalize per-window (mimics an on-device streaming normalizer)."""
    X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-6)
    return X[..., np.newaxis]  # (N, WINDOW_LEN, 1) for Conv1D


def generate_machine_dataset(machine_id, severity, inner, outer, ball, normal_slice,
                              n_per_class_train=400, n_per_class_test=100, seed=0):
    rng = np.random.default_rng(seed)
    label_to_file = {1: inner, 2: outer, 3: ball}

    X_train_parts, y_train_parts, X_test_parts, y_test_parts = [], [], [], []
    for label, fname in label_to_file.items():
        sig = _load_de_signal(fname)
        xtr, ytr, xte, yte = _sample_windows(sig, label, n_per_class_train, n_per_class_test, rng)
        X_train_parts.append(xtr); y_train_parts.append(ytr)
        X_test_parts.append(xte); y_test_parts.append(yte)

    # normal class: this node's own disjoint slice of the shared healthy recording
    # The healthy recording is shared by 3 nodes, so each node only gets a
    # third of it -- use denser (75%) overlap in its TRAINING region so the
    # normal class is not under-represented. Test windows stay non-overlapping.
    xtr, ytr, xte, yte = _sample_windows(normal_slice, 0, n_per_class_train, n_per_class_test,
                                         rng, train_stride=WINDOW_LEN // 4)
    X_train_parts.append(xtr); y_train_parts.append(ytr)
    X_test_parts.append(xte); y_test_parts.append(yte)

    X_train = _znorm(np.concatenate(X_train_parts))
    y_train = np.concatenate(y_train_parts)
    X_test = _znorm(np.concatenate(X_test_parts))
    y_test = np.concatenate(y_test_parts)

    # shuffle
    tr_idx = rng.permutation(len(X_train))
    te_idx = rng.permutation(len(X_test))
    return (X_train[tr_idx], y_train[tr_idx]), (X_test[te_idx], y_test[te_idx])


def main(out_dir=None):
    out_dir = out_dir or os.path.dirname(os.path.abspath(__file__))
    all_test_X, all_test_y = [], []

    # split the single shared "Normal" recording into disjoint per-node segments
    normal_sig_full = _load_de_signal(NORMAL_FILE)
    n_nodes = len(MACHINE_PROFILES)
    seg_len = len(normal_sig_full) // n_nodes
    normal_segments = [normal_sig_full[i * seg_len:(i + 1) * seg_len] for i in range(n_nodes)]

    for i, profile in enumerate(MACHINE_PROFILES):
        (X_train, y_train), (X_test, y_test) = generate_machine_dataset(
            seed=100 + i, normal_slice=normal_segments[i], **profile
        )
        node_dir = os.path.join(out_dir, profile["machine_id"])
        os.makedirs(node_dir, exist_ok=True)
        np.savez(os.path.join(node_dir, "train.npz"), X=X_train, y=y_train)
        np.savez(os.path.join(node_dir, "test.npz"), X=X_test, y=y_test)
        all_test_X.append(X_test)
        all_test_y.append(y_test)
        print(f"[{profile['machine_id']}] train={X_train.shape} test={X_test.shape} "
              f"defect_severity={profile['severity']}mil (real CWRU data)")

    # combined held-out set used by the cloud aggregator to score the global model
    Xg = np.concatenate(all_test_X)
    yg = np.concatenate(all_test_y)
    np.savez(os.path.join(out_dir, "global_test.npz"), X=Xg, y=yg)
    print(f"[global_test] {Xg.shape}, classes={CLASSES}")


if __name__ == "__main__":
    main()
