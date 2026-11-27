# EdgeSentry
### Federated Edge-AI for Real-Time Industrial Predictive Maintenance — with a full web dashboard

EdgeSentry detects bearing faults (**normal / inner-race / outer-race / ball**) from vibration data
**directly on the machine**, using a compressed INT8 model trained collaboratively across several machines
with **federated learning** — no raw sensor data ever leaves a device. A Flask website monitors the fleet
live, lets an operator *orchestrate* the edge devices, explains every decision, and presents all
experimental results.


**capabilities**

* **Website** (6 pages): live fleet · explainability lab · federated learning · edge compression · privacy · about
* **Explainability**: 4 attribution methods (gradient, SmoothGrad, Integrated Gradients, occlusion) **plus a deletion test that measures whether each explanation is faithful** — plain gradient saliency fails it
* **Edge orchestration** (dashboard → device, over MQTT or REST): inject faults, hot-swap the model, cut/restore the uplink, add sensor noise, change reporting mode
* **Edge autonomy**: on-device debounced alarm, store-and-forward buffer, report-on-change, uncertainty flag, kurtosis/crest-factor features, model SHA-256
* **Baselines that explain the value of FL**: local-only vs FedAvg vs centralized; leave-one-severity-out generalization
* **Fair compression study** with MACs, peak-RAM estimate, noise robustness and an MCU/SBC "will it fit?" calculator
* **Docker**: resource-limited edge containers (0.5 CPU / 512 MB), health check, MQTT reconnect logic (edge nodes no longer silently fall back if the broker starts late)
* **Tests**: `python tests/test_smoke.py`

---

## Quick start

All results are **already generated** in `results/` and the models are in `saved_models/`, so the website shows real numbers immediately.

### A — Local demo, no Docker (REST transport)
```bash
pip install -r requirements.txt
./run_local_demo.sh            # dashboard + 3 edge nodes  ->  http://localhost:5050
```

### B — Full architecture with a real MQTT broker (Docker)
```bash
docker compose up --build      # mosquitto + dashboard + 3 resource-limited edge nodes
# open http://localhost:5050
```

### C — Regenerate every result
```bash
./run_pipeline.sh              # full (~1 h on a laptop CPU)
QUICK=1 ./run_pipeline.sh      # 1 seed, trimmed DP sweep
```
Needs the raw CWRU files in `archive/raw/` (included).

### D — Tests
```bash
python tests/test_smoke.py     # 9 fast checks: no data leakage, DP accountant, FedAvg maths, website, alarm logic
```

---

## 5-minute live demo script (for the presentation)

1. **Live Fleet** — three machines report; each runs *on-device* INT8 inference. Point at the KPI "uplink traffic saved" (≈99% less than streaming the raw sensor).
2. Click **Inject inner-race fault** (all machines) → cards turn red, the **ALARM** banner appears.
3. On one card press **✂ Cut uplink**: the node keeps classifying and raises its **on-device alarm**; press **Restore** → the buffered backlog is flushed ("flushed N").
4. Switch a node to **Model: pruned + INT8** → the SHA-256 changes (OTA-style swap), accuracy barely moves.
5. Set **Sensor noise: 5 dB** → predictions degrade (honest limitation).
6. Click **🔍 Explain** → Explainability Lab: compare the four methods and show the faithfulness curve.
7. **Federated Learning** page → local-only vs FedAvg vs centralized; **Privacy** page → ε calculator.

---

## Results (mean ± std over 3 seeds; all produced by `./run_pipeline.sh`)

### Federated learning: what does federation buy?
| Training setup | Test accuracy (mean ± std, 3 seeds) | Macro-F1 | Raw data stays local? |
|---|---:|---:|:--:|
| Local-only (each machine alone) | 68.6 ± 0.2% | 71.5 ± 0.3% | yes |
| **FedAvg (3 nodes)** | **84.9 ± 0.4%** | 86.9 ± 0.4% | yes |
| Centralized (pooled data) | 95.1 ± 1.0% | 95.7 ± 0.9% | **no** |

FedAvg closes 62% of the gap between training alone and pooling all data, while keeping data local.
The remaining gap is non-IID *client drift* (each node sees one defect size); lower LR, fewer local epochs and a different
normalisation layer did not close it (learning-rate decay stabilises convergence).

### Generalization to an unseen defect size (leave-one-severity-out)
| Held-out machine (unseen defect size) | FedAvg | Centralized |
|---|---:|---:|
| edge01 (0.007 in) | 60.7 ± 0.0% | 72.0 ± 0.0% |
| edge02 (0.014 in) | 59.8 ± 0.0% | 41.9 ± 0.0% |
| edge03 (0.021 in) | 65.6 ± 0.0% | 44.2 ± 0.0% |

### Compression (fair comparison; sizes are TFLite files)
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

* Quantization is essentially free (-0.5 pts, 51% smaller).
* Pruned 50% + INT8: **57 → 18 KB**, 11972 → 3300 parameters, 3.2× fewer MACs.
* **Weak spot:** accuracy collapses under sensor noise (see the 5 dB column) — the model was trained on clean lab data. Noise-augmented training is the obvious next step.
* Latency on the host CPU is ~0.05 ms (full) vs ~0.05 ms (pruned): for a model this small the interpreter's overhead dominates, so the compute saving only matters on slower cores. Not a Raspberry Pi measurement.

### Explainability: which explanations are trustworthy?
| Method | Cost | Deletion AUC (lower = more faithful) | vs random removal (0.694) |
|---|---:|---:|---|
| Gradient saliency | 20 ms | 0.746 | ✘ not better than random |
| SmoothGrad | 30 ms | 0.747 | ✘ not better than random |
| Integrated Gradients | 34 ms | 0.663 | ✔ beats random |
| Occlusion | 28 ms | 0.600 | ✔ beats random |

Plain gradient saliency and SmoothGrad look plausible but are **no better than deleting random parts of the signal**; Integrated Gradients and
occlusion are faithful. Occlusion is nearly the deletion metric itself, so its win is partly by construction.
Integrated-Gradients importance also tracks local impulse amplitude (Spearman ρ ≈ 0.88 for inner-race faults), i.e. the model looks at the bearing impacts.

### Differential privacy (client-level, δ = 1e-5)
| Clients | σ | ε (RDP) | Accuracy |
|---:|---:|---:|---:|
| 3 | clip only | ∞ | 85.0% |
| 3 | 0.1 | 419 | 62.6% |
| 3 | 0.3 | 71 | 30.4% |
| 12 | clip only | ∞ | 29.0% |
| 12 | 0.1 | 419 | 45.4% |
| 12 | 0.3 | 71 | 70.9% |

**Caveat:** the bundled DP sweep is short (6 rounds, 1 seed) and noisy — some points are non-monotonic (e.g. the 12-client clip-only run is unstable). Use it for the *shape* of the trade-off; rerun `python federated/compare_dp.py --seeds 0 1 2 --clients 3 12 24 --rounds 15` for a proper study.

With ~12 000 trainable weights, DP noise only becomes tolerable with many clients; with 3 clients there is no useful operating point.
This is a property of DP-FedAvg at small scale, reported rather than hidden.

---

## Project structure
```
common.py                  shared constants (window length, classes, paths, noise helper)
data/prepare_dataset.py    leak-free temporal split of real CWRU data (3 non-IID machines)
federated/
  fl_core.py               FedAvg + DP-FedAvg engine, baselines (centralized, local-only)
  simulate_federated.py    official training run -> saved_models/global_model.keras
  dp_utils.py              clipping, Gaussian noise, RDP privacy accountant
  compare_dp.py            DP study over #clients x noise
model/
  architecture.py          named-layer 1D-CNN (11 972 params), MAC / RAM bookkeeping
  compress_model.py        fp32 + INT8 TFLite     prune_model.py   structured pruning + federated fine-tune
  explainability.py        4 XAI methods + faithfulness (deletion) test
benchmarking/benchmark.py  size, params, MACs, RAM, latency, accuracy, noise robustness
experiments/               multi-seed studies: fl_comparison, compression_study, generalization
edge_node/                 engine.py (TFLite) + edge_app.py (scenarios, commands, store-and-forward ...)
cloud_aggregator/          Flask website: dashboard_app.py, templates/, static/ (Chart.js bundled, works offline)
mqtt/  docker-compose.yml  broker config + deployment topology
tests/test_smoke.py        sanity tests          docs/PROPOSAL.md   submission document
```

## Honest limitations
* **No real edge hardware was used.** Latency is from a host CPU; peak RAM and MCU timings are analytic estimates.
* **One lab, one load, three defect sizes.** Accuracy on an unseen defect size is far lower (see above). Real machines add speed/load variation and noise.
* **Not noise-robust** (trained on clean data). **FedAvg trails pooled training** (client drift).
* The sensor is simulated by replaying held-out test windows; the MQTT broker is plaintext and unauthenticated (production: TLS, ACLs, signed model updates, secure aggregation).

