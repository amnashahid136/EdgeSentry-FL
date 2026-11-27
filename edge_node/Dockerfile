FROM python:3.11-slim
WORKDIR /app

# NOTE for real hardware (Raspberry Pi / Jetson): `pip install tflite-runtime`
# instead of tensorflow-cpu. edge_node/engine.py picks it up automatically
# (no code change) and the install shrinks from ~600 MB to a few MB.
# Full tensorflow-cpu is used here only so the same image also runs on a laptop.
RUN pip install --no-cache-dir numpy==1.26.4 paho-mqtt==1.6.1 tensorflow-cpu==2.21.0

COPY common.py /app/common.py
COPY edge_node/ /app/edge_node/
# only what a device needs: its local sensor shard (stand-in for a real accelerometer) + the INT8 models
COPY data/edge01/test.npz /app/data/edge01/test.npz
COPY data/edge02/test.npz /app/data/edge02/test.npz
COPY data/edge03/test.npz /app/data/edge03/test.npz
COPY saved_models/edge_model_int8.tflite /app/saved_models/edge_model_int8.tflite
COPY saved_models/pruned_model_int8.tflite /app/saved_models/pruned_model_int8.tflite

ENV PYTHONPATH=/app TF_CPP_MIN_LOG_LEVEL=3
ENTRYPOINT ["python3", "edge_node/edge_app.py"]
