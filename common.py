"""
common.py
---------
Single source of truth for constants shared by every part of EdgeSentry
(data prep, model, federated training, edge node, cloud website).
Keeping them here means the window length / class list can never drift
between the training code and the code running on the edge devices.
"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

WINDOW_LEN = 1024          # samples per vibration window (~21 ms at 48 kHz)
SAMPLE_RATE = 48000.0      # Hz, CWRU drive-end accelerometer
NUM_CLASSES = 4
CLASSES = ["normal", "inner_race_fault", "outer_race_fault", "ball_fault"]
CLASS_LABELS = ["Normal", "Inner-race fault", "Outer-race fault", "Ball fault"]
MACHINES = ["edge01", "edge02", "edge03"]
SEVERITY_BY_MACHINE = {"edge01": "0.007 in", "edge02": "0.014 in", "edge03": "0.021 in"}

DATA_DIR = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "saved_models")
RESULTS_DIR = os.path.join(ROOT, "results")

EDGE_MODEL_FILE = "edge_model_int8.tflite"      # what the edge nodes run
PRUNED_EDGE_MODEL_FILE = "pruned_model_int8.tflite"
GLOBAL_MODEL_FILE = "global_model.keras"        # FP32 model (cloud-side XAI)


# ----------------------------------------------------------- signal helpers
import numpy as np  # noqa: E402


def znorm_windows(X):
    """Per-window z-normalisation -- the same cheap streaming normaliser the
    edge device applies to every raw accelerometer window."""
    X = X[..., 0] if X.ndim == 3 else X
    X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-6)
    return X[..., np.newaxis].astype(np.float32)


def add_awgn(X, snr_db, rng):
    """Add white Gaussian sensor noise at a given signal-to-noise ratio (dB)
    to already z-normalised windows (signal power = 1), then re-normalise.
    snr_db=None -> unchanged. Used for the noise-robustness study."""
    if snr_db is None:
        return X
    sigma = 10 ** (-snr_db / 20.0)
    noisy = X + rng.normal(0, sigma, size=X.shape).astype(np.float32)
    return znorm_windows(noisy)


def new_mqtt_client(client_id=""):
    """paho-mqtt 1.x and 2.x have different constructors; this works with both."""
    import paho.mqtt.client as mqtt
    try:
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, client_id=client_id)   # paho >= 2.0
    except AttributeError:
        return mqtt.Client(client_id=client_id)                                    # paho 1.x
