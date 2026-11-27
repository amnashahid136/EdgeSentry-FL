# EdgeSentry — Semester Project Proposal
### Federated Edge-AI for Real-Time Industrial Predictive Maintenance
**Course:** Edge Computing and Intelligent Systems

---

## 1. Problem Statement

Industrial machines (motors, pumps, conveyor bearings) fail gradually, giving
off early vibration signatures before a breakdown. Sending raw, continuous
accelerometer data to the cloud for fault detection is expensive, adds
latency that is unacceptable for real-time alarms, and exposes proprietary
factory-floor data. **EdgeSentry** pushes the fault-detection model itself
onto the machine ("the edge"), so each machine detects its own faults
in real time, on-device, and only ever sends a small status message
("normal" / "inner race fault" / "outer race fault" / "ball fault") to a
central monitoring dashboard.

This is a direct instance of the edge-AI pattern the course centers on:
*model compression → on-device inference → federated learning across many
devices → benchmarking the resulting performance/privacy trade-offs.*

## 2. Objectives

1. Design a compact 1D-CNN fault-classification model small enough to run
   on constrained edge hardware (Raspberry Pi / Jetson Nano / MCU class).
2. Train the model **collaboratively but privately** across 3 simulated
   machines using **federated learning (FedAvg)** — no raw sensor data
   ever leaves a machine.
3. **Compress** the trained model via post-training INT8 quantization for
   deployment on resource-constrained edge devices.
4. Deploy the compressed model to simulated edge nodes (Docker containers,
   directly portable to real Raspberry Pi/Jetson hardware) that perform
   **real-time on-device inference** and publish results over **MQTT**.
5. Build a **cloud dashboard** that aggregates live status from all edge
   nodes — the "cloud" half of the edge-cloud continuum.
6. **Benchmark** the cloud vs. edge model on size, inference latency, and
   accuracy, and provide a lightweight **explainability (XAI)** view of
   model decisions.

## 3. Why This Project Is a Strong Fit (CLO / Topic Coverage)

| Course requirement | Where it's implemented |
|---|---|
| Edge/fog/cloud taxonomy & edge-cloud continuum (CLO1) | `docker-compose.yml` topology: edge nodes ↔ MQTT broker ↔ cloud dashboard |
| Edge hardware constraints (Pi/Jetson, compute/power/memory trade-offs) | Model sized to 12K params / 28 KB after INT8 (18 KB pruned+INT8); MACs, RAM estimate and an MCU fit calculator on the website's Compression page |
| Networking for the edge (MQTT) | `mqtt/mosquitto.conf`, `edge_node/edge_app.py` publish/subscribe |
| Model compression (quantization) (CLO2) | `model/compress_model.py` — INT8 post-training quantization |
| Model compression (structured pruning) (CLO2) | `model/prune_model.py` — dependency-aware channel pruning + fine-tune, benchmarked alongside quantization |
| On-device inference frameworks (TFLite) (CLO2) | `edge_node/edge_app.py` — TFLite Interpreter |
| Federated learning at the edge (FedAvg, non-IID) (CLO2) | `federated/simulate_federated.py` |
| Edge-AI applications — Industrial automation | Predictive maintenance / bearing-fault detection |
| Security & privacy at the edge (CLO3) | FL keeps raw data local by design; `federated/simulate_federated.py --dp` adds clip+noise DP-FedAvg on top, with a privacy/utility sweep in `federated/compare_dp.py`; see §6 |
| Explainability & trust in edge AI (CLO3) | `model/explainability.py` — gradient saliency maps |
| Performance benchmarking (CLO3) | `benchmarking/benchmark.py` — size/latency/accuracy report |
| Edge orchestration & containerization (Docker) | `docker-compose.yml`, per-service `Dockerfile`s |

## 4. System Architecture

```
                     ┌─────────────────────────┐
                     │      CLOUD LAYER         │
                     │  Flask Dashboard          │
                     │  (aggregated monitoring)  │
                     └────────────▲─────────────┘
                                  │  MQTT (status only,
                                  │   no raw sensor data)
                     ┌────────────┴─────────────┐
                     │     Mosquitto Broker       │
                     └───┬───────────┬───────────┘
             ┌───────────┘           │           └───────────┐
     ┌───────▼──────┐        ┌───────▼──────┐        ┌───────▼──────┐
     │  Edge Node 1  │        │  Edge Node 2  │        │  Edge Node 3  │
     │ (Raspberry Pi)│        │ (Raspberry Pi)│        │ (Raspberry Pi)│
     │ TFLite INT8   │        │ TFLite INT8   │        │ TFLite INT8   │
     │ on-device     │        │ on-device     │        │ on-device     │
     │ inference     │        │ inference     │        │ inference     │
     │ + local FL    │        │ + local FL    │        │ + local FL    │
     │   training    │        │   training    │        │   training    │
     └───────────────┘        └───────────────┘        └───────────────┘
           ▲ local vibration       ▲ local vibration        ▲ local vibration
           │ sensor data           │ sensor data             │ sensor data
           │ (never leaves         │ (never leaves           │ (never leaves
           │  the device)          │  the device)             │  the device)
```

**Offline (before deployment)**: `federated/simulate_federated.py` runs the
FedAvg training rounds across the three machines' local data shards and
produces one global model, which `model/compress_model.py` then quantizes.

**Online (deployed)**: each edge node loads the quantized model and runs
`edge_node/edge_app.py`, streaming live sensor windows through the model
and publishing only the classification result to MQTT.

## 5. Dataset

`data/prepare_dataset.py` loads **real bearing vibration recordings from
the CWRU (Case Western Reserve University) Bearing Data Center**
(`archive/raw/*.mat`, 48kHz drive-end accelerometer signals) for 4 classes
— Normal, Inner Race Fault, Outer Race Fault, Ball Fault — assigning each
simulated machine a different real defect severity (0.007" / 0.014" /
0.021"), producing genuine **non-IID data**, the exact condition federated
learning is designed to handle. Because the loader outputs the same
window shape and file layout as before, it plugs into the model, FL,
compression, and deployment code with no other changes required.

## 6. Security & Privacy Discussion

- **Data locality (built-in):** federated learning means raw vibration
  data never leaves a machine; only model weight updates (during training)
  and short status messages (during inference) are transmitted.
- **Differential privacy on top of FL (`--dp` flag):** FL alone protects
  raw *data*, but the published global model can still leak information
  about a client's data through its weights (e.g. membership inference).
  `federated/simulate_federated.py --dp` adds the standard DP-FedAvg
  defense (McMahan et al., 2018): each client's weight update is clipped
  to a fixed L2 norm, the clipped updates are summed, calibrated Gaussian
  noise is added to the sum, and the result is averaged — bounding how
  much any single client's data can influence the published model.
  `federated/compare_dp.py` sweeps the noise level and reports the
  accuracy trade-off against a simple, deliberately conservative epsilon
  bound (see `federated/dp_utils.py` for the exact accounting and its
  limitations — with only 3 simulated clients the achievable epsilon is
  large in absolute terms; the report should present this as a
  demonstration of the mechanism and trade-off shape, not a
  production-grade privacy guarantee at this scale).
- **Transport:** the demo runs MQTT unauthenticated for simplicity; the
  report will discuss adding TLS + broker authentication (`mosquitto`
  supports both) as the production hardening step.
- **On-device trust:** discuss secure boot / trusted execution as the
  hardware-level complement to the software-level privacy already provided
  by FL (per the course's "Security & privacy at the edge" topic).

## 7. Explainability

`model/explainability.py` computes a gradient-based saliency map showing
which parts of a vibration window most influenced a fault prediction.
This is intentionally computed on the FP32 model (offline / at
commissioning time) rather than on the INT8 edge model used for real-time
inference — a genuine **interpretability-vs-latency trade-off** worth
discussing in the final report, exactly as listed in the course topics.

## 8. Evaluation & Results (mean ± std over 3 seeds; reproduce with `./run_pipeline.sh`)

**Federated learning vs. alternatives** (test accuracy, held-out time segments of every recording; no train/test sample overlap):

| Setup | Accuracy | Raw data stays local |
|---|---:|:--:|
| Local-only (each machine alone) | 68.6 ± 0.2% | yes |
| **FedAvg, 3 nodes** | **84.9 ± 0.4%** | yes |
| Centralized (pooled) | 95.1 ± 1.0% | no |

**Compression** (TFLite sizes; pruned variants fine-tuned *federatedly*; a control gets the same extra rounds):

| Variant | Size KB | Params | MACs | Accuracy | Acc @ 5 dB noise |
|---|---:|---:|---:|---:|---:|
| FP32 baseline | 56.7 | 11972 | 977K | 84.9 ± 0.4% | 48% |
| INT8 (quantized) | 27.8 | 11972 | 977K | 84.4 ± 0.3% | 48% |
| FP32 + FL fine-tune (control) | 56.7 | 11972 | 977K | 85.3 ± 0.6% | 49% |
| Pruned 25% (FP32) | 37.4 | 6964 | 597K | 85.1 ± 0.8% | 58% |
| Pruned 25% + INT8 | 22.1 | 6964 | 597K | 85.0 ± 0.9% | 59% |
| Pruned 50% (FP32) | 23.3 | 3300 | 308K | 84.6 ± 0.6% | 54% |
| Pruned 50% + INT8 | 17.6 | 3300 | 308K | 84.3 ± 0.8% | 53% |
| Pruned 75% (FP32) | 14.4 | 980 | 109K | 77.3 ± 7.2% | 54% |
| Pruned 75% + INT8 | 14.4 | 980 | 109K | 77.4 ± 7.3% | 54% |

**Explainability** (deletion test on 100 correctly-classified windows):

| Method | Cost | Deletion AUC (lower = better) | Verdict |
|---|---:|---:|---|
| Gradient saliency | 20 ms | 0.746 | not better than random |
| SmoothGrad | 30 ms | 0.747 | not better than random |
| Integrated Gradients | 34 ms | 0.663 | faithful |
| Occlusion | 28 ms | 0.600 | faithful |

**Takeaways.** INT8 quantization is nearly free; structured pruning of 50% of channels is accuracy-neutral (it does *not* improve accuracy once compared with the control).
FedAvg recovers most of the gap between local-only and pooled training while keeping data local. Plain gradient saliency fails the faithfulness test, so the project
explains decisions with Integrated Gradients / occlusion in the cloud while the INT8 model decides on the edge. Latency was measured on a host CPU
(single thread), not on Raspberry Pi hardware. Known weaknesses (reported on the website): low accuracy on an unseen defect size, sensitivity to sensor
noise, and DP only being meaningful with many clients.

## 8b. Edge-computing features implemented
On-device INT8 inference · on-device debounced alarm · store-and-forward during uplink outage · report-on-change with heartbeat · runtime model hot-swap
with SHA-256 integrity · cloud→edge control channel (MQTT/REST) · kurtosis/crest-factor condition indicators · low-confidence flag · sensor-noise injection ·
resource-limited containers (0.5 CPU, 512 MB) · MCU/SBC flash+RAM fit calculator · live bandwidth-saving KPI.

## 9. Tools & Technologies

Python, TensorFlow/Keras, TensorFlow Lite, Docker & docker-compose,
Eclipse Mosquitto (MQTT), Flask, NumPy, Matplotlib. All open-source, all
runnable on a laptop for development and directly portable to Raspberry
Pi / Jetson hardware for a real deployment (stretch goal, see §10).

## 10. 16-Week Timeline

| Weeks | Milestone |
|---|---|
| 1–2 | Literature review of edge-AI / TinyML / FL papers; finalize dataset & problem scope |
| 3–4 | Build & train baseline (non-federated) 1D-CNN; establish accuracy baseline |
| 5–6 | Implement federated learning (FedAvg) across simulated edge nodes; convergence analysis |
| 7–8 | Model compression (quantization, optionally pruning); compression benchmark report |
| 9–10 | Build edge node app (TFLite inference) + MQTT + cloud dashboard; container orchestration |
| 11 | **Sessional-II checkpoint**: working end-to-end demo |
| 12–13 | Explainability module; security/privacy hardening (MQTT auth/TLS) |
| 14 | Stretch goal: deploy to a real Raspberry Pi / Jetson Nano if hardware is available |
| 15 | Full benchmarking (latency/energy/accuracy), write final report |
| 16 | Final presentation & demo |

## 11. Stretch Goals (optional, if time allows)

- Deploy one edge node to real Raspberry Pi hardware for an authentic
  latency/energy measurement instead of a container simulation.
- Differential-privacy noise added to the federated weight updates.
- Structured pruning in addition to quantization, for a second
  compression comparison point.
