"""
engine.py
---------
On-device inference engine: loads a TFLite model and classifies one
vibration window. Shared by the edge node app, the benchmark script and the
website's playground, so latency/accuracy numbers mean the same thing
everywhere.

On a real Raspberry Pi / Jetson, `pip install tflite-runtime` and this module
automatically uses it instead of full TensorFlow (much lighter install).
"""
import time

import numpy as np


def _interpreter_cls():
    try:
        from tflite_runtime.interpreter import Interpreter  # lightweight, for real edge hardware
        return Interpreter
    except ImportError:
        import tensorflow as tf
        return tf.lite.Interpreter


class EdgeInferenceEngine:
    def __init__(self, model_path, num_threads=1):
        # 1 thread = what a single small ARM core would give you.
        self.interpreter = _interpreter_cls()(model_path=model_path, num_threads=num_threads)
        self.interpreter.allocate_tensors()
        self.inp = self.interpreter.get_input_details()[0]
        self.out = self.interpreter.get_output_details()[0]
        self.model_path = model_path

    def probs(self, window):
        """window: (L,1) float32 -> (class probabilities, latency in ms)."""
        t0 = time.perf_counter()
        self.interpreter.set_tensor(self.inp["index"], window[np.newaxis, ...].astype(np.float32))
        self.interpreter.invoke()
        p = self.interpreter.get_tensor(self.out["index"])[0].copy()
        return p, (time.perf_counter() - t0) * 1000.0

    def predict(self, window):
        p, ms = self.probs(window)
        c = int(np.argmax(p))
        return c, float(p[c]), ms, p

    def predict_batch(self, X):
        return np.stack([self.probs(x)[0] for x in X])
