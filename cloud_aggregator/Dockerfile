FROM python:3.11-slim
WORKDIR /app

# TensorFlow is needed ONLY for the explainability lab (FP32 model, gradients)
# and the INT8 playground; live monitoring itself needs just Flask + paho.
RUN pip install --no-cache-dir flask==3.0.3 numpy==1.26.4 paho-mqtt==1.6.1 tensorflow-cpu==2.21.0

COPY common.py /app/common.py
COPY cloud_aggregator/ /app/cloud_aggregator/
COPY edge_node/engine.py /app/edge_node/engine.py
COPY model/architecture.py model/explainability.py /app/model/
COPY data/edge01/test.npz /app/data/edge01/test.npz
COPY data/edge02/test.npz /app/data/edge02/test.npz
COPY data/edge03/test.npz /app/data/edge03/test.npz
COPY data/global_test.npz /app/data/global_test.npz
COPY saved_models/global_model.keras saved_models/edge_model_int8.tflite /app/saved_models/
COPY results/ /app/results/

ENV PYTHONPATH=/app TF_CPP_MIN_LOG_LEVEL=3
EXPOSE 5000
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s CMD python3 -c "import urllib.request;urllib.request.urlopen('http://localhost:5000/healthz')" || exit 1
CMD ["python3", "cloud_aggregator/dashboard_app.py"]
