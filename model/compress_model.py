"""
compress_model.py
-------------------
Compresses a trained Keras model to TensorFlow Lite: a float32 variant and a
full-integer INT8 post-training-quantized variant (CLO2).

    saved_models/edge_model_fp32.tflite
    saved_models/edge_model_int8.tflite   <- what the edge nodes load

The INT8 model uses calibration windows drawn from the nodes' own TRAINING
shards (never the test set).
"""
import os
import shutil
import sys
import tempfile

import numpy as np
import tensorflow as tf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from common import MODEL_DIR, DATA_DIR, MACHINES, GLOBAL_MODEL_FILE  # noqa: E402


def representative_dataset_gen(calib_samples=300):
    X = np.concatenate([np.load(os.path.join(DATA_DIR, m, "train.npz"))["X"] for m in MACHINES])
    idx = np.random.default_rng(0).choice(len(X), size=min(calib_samples, len(X)), replace=False)
    for i in idx:
        yield [X[i:i + 1].astype(np.float32)]


def convert(model_path=None, out_dir=None, out_prefix="edge_model", quiet=False, model=None):
    """Convert a Keras model (path or object) to fp32 + INT8 TFLite files."""
    out_dir = out_dir or MODEL_DIR
    os.makedirs(out_dir, exist_ok=True)
    if model is None:
        model_path = model_path or os.path.join(MODEL_DIR, GLOBAL_MODEL_FILE)
        model = tf.keras.models.load_model(model_path)

    # Convert via SavedModel (not from_keras_model): on Keras 3 the direct
    # helper has a fragile fallback path that can crash with
    # "TypeError: 'NoneType' object is not callable".
    export_dir = tempfile.mkdtemp(prefix="edgesentry_savedmodel_")
    try:
        model.export(export_dir)
        conv = tf.lite.TFLiteConverter.from_saved_model(export_dir)
        fp32_path = os.path.join(out_dir, f"{out_prefix}_fp32.tflite")
        with open(fp32_path, "wb") as f:
            f.write(conv.convert())

        conv = tf.lite.TFLiteConverter.from_saved_model(export_dir)
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = representative_dataset_gen
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.float32     # float I/O keeps the edge app simple
        conv.inference_output_type = tf.float32
        int8_path = os.path.join(out_dir, f"{out_prefix}_int8.tflite")
        with open(int8_path, "wb") as f:
            f.write(conv.convert())
    finally:
        shutil.rmtree(export_dir, ignore_errors=True)

    if not quiet:
        k = os.path.getsize(model_path) / 1024 if model_path else float("nan")
        print(f"Model compression ({out_prefix}): keras {k:.1f} KB | "
              f"fp32 tflite {os.path.getsize(fp32_path)/1024:.1f} KB | "
              f"INT8 tflite {os.path.getsize(int8_path)/1024:.1f} KB")
    return fp32_path, int8_path


if __name__ == "__main__":
    convert()
