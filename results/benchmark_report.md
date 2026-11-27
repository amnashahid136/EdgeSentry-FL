# EdgeSentry Benchmark Report (single run, official model)

| Model | Size KB | Params | MACs | Peak RAM KB | Latency ms (mean / p95) | Accuracy | Macro-F1 | Acc @5 dB noise |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Keras FP32 (cloud baseline) | 97.2 | 11972 | 977K | 48.0 | 9.798 / 10.510 | 0.850 | 0.870 | 0.381 |
| TFLite FP32 | 56.7 | 11972 | 977K | 48.0 | 0.046 / 0.068 | 0.850 | 0.870 | 0.381 |
| TFLite INT8 (quantized) | 27.8 | 11972 | 977K | 12.0 | 0.049 / 0.074 | 0.845 | 0.865 | 0.383 |
| Pruned Keras FP32 | 63.4 | 3300 | 308K | 24.0 | 9.631 / 10.993 | 0.854 | 0.872 | 0.581 |
| Pruned + INT8 (both) | 16.6 | 3300 | 308K | 6.0 | 0.046 / 0.073 | 0.852 | 0.870 | 0.588 |

Size 97.2 KB -> 16.6 KB (83% smaller); accuracy 0.850 -> 0.852.

Single-run numbers; see results/exp_compression.json for the 5-seed mean +/- std. Latency was measured on the host CPU (1 thread), not on Raspberry Pi hardware.
