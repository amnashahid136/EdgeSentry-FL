"""
architecture.py
----------------
Compact 1D-CNN for bearing-fault classification, deliberately sized to be
edge-deployable (~12K parameters, ~25 KB after INT8 quantization) so it
runs on a Raspberry Pi / Jetson Nano / microcontroller-class device.

The network ends in GlobalAveragePooling, so the parameter count does NOT
depend on the input window length -- that is why the window can be
lengthened (400 -> 1024 samples) to improve accuracy at zero model-size
cost.

`build_model(cfg)` takes a channel configuration, so the structurally
pruned model (model/prune_model.py) is built by the very same function
with fewer channels -- no duplicated topology code.
"""
import os
import sys

import tensorflow as tf
from tensorflow.keras import layers, models

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import WINDOW_LEN, NUM_CLASSES  # noqa: E402,F401

# Unpruned channel / unit counts. Layers are NAMED, so pruning and XAI code
# can address them by name instead of fragile positional indices.
BASE_CONFIG = {"conv1": 16, "conv2": 32, "conv3": 64, "dense1": 32}
KERNELS = {"conv1": 31, "conv2": 5, "conv3": 3}


def build_model(cfg=None, window_len=WINDOW_LEN, num_classes=NUM_CLASSES, lr=1e-3, name=None):
    cfg = cfg or BASE_CONFIG
    model = models.Sequential([
        layers.Input(shape=(window_len, 1)),
        layers.Conv1D(cfg["conv1"], KERNELS["conv1"], strides=2, activation="relu",
                      padding="same", name="conv1"),
        layers.BatchNormalization(name="bn1"),
        layers.MaxPooling1D(2, name="pool1"),
        layers.Conv1D(cfg["conv2"], KERNELS["conv2"], strides=2, activation="relu",
                      padding="same", name="conv2"),
        layers.BatchNormalization(name="bn2"),
        layers.MaxPooling1D(2, name="pool2"),
        layers.Conv1D(cfg["conv3"], KERNELS["conv3"], activation="relu",
                      padding="same", name="conv3"),
        layers.BatchNormalization(name="bn3"),
        layers.GlobalAveragePooling1D(name="gap"),
        layers.Dense(cfg["dense1"], activation="relu", name="dense1"),
        layers.Dropout(0.3, name="drop"),
        layers.Dense(num_classes, activation="softmax", name="out"),
    ], name=name or "edgesentry_1dcnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def count_macs(cfg=None, window_len=WINDOW_LEN, num_classes=NUM_CLASSES):
    """Multiply-accumulate operations for ONE inference (analytic count).
    MACs are the standard hardware-independent compute metric for edge AI."""
    cfg = cfg or BASE_CONFIG
    L = window_len
    macs = 0
    cin = 1
    for name, pool in (("conv1", True), ("conv2", True), ("conv3", False)):
        stride = 2 if name != "conv3" else 1
        L = -(-L // stride)                       # 'same' padding -> ceil(L/stride)
        macs += L * KERNELS[name] * cin * cfg[name]
        cin = cfg[name]
        if pool:
            L //= 2
    macs += cfg["conv3"] * cfg["dense1"] + cfg["dense1"] * num_classes
    return int(macs)


def peak_activation_elems(cfg=None, window_len=WINDOW_LEN):
    """Largest (input + output) activation footprint of any single layer, in
    elements -- a first-order estimate of the RAM 'tensor arena' an MCU
    would need. Multiply by 1 byte (INT8) or 4 bytes (FP32)."""
    cfg = cfg or BASE_CONFIG
    L, cin, peak = window_len, 1, 0
    for name, pool in (("conv1", True), ("conv2", True), ("conv3", False)):
        stride = 2 if name != "conv3" else 1
        Lout = -(-L // stride)
        peak = max(peak, L * cin + Lout * cfg[name])
        L, cin = Lout, cfg[name]
        if pool:
            peak = max(peak, L * cin + (L // 2) * cin)
            L //= 2
    return int(peak)


if __name__ == "__main__":
    m = build_model()
    m.summary()
    print("MACs/inference:", count_macs(), " peak act elems:", peak_activation_elems())
