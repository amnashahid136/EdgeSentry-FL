# EdgeSentry Structured Pruning Report

- Prune ratio: 50% fewer channels/units per layer
- Channels kept: {'conv1': 8, 'conv2': 16, 'conv3': 32, 'dense1': 16} (base: {'conv1': 16, 'conv2': 32, 'conv3': 64, 'dense1': 32})
- Parameters: 11972 -> 3300 (72.4% fewer)

| Stage | Accuracy |
|---|---:|
| Source model (unpruned) | 0.8497 |
| Pruned, before fine-tuning | 0.4432 |
| Pruned, after 4 federated fine-tune rounds | 0.8536 |

Fine-tuning is federated (no data pooling). For a *fair* comparison the unpruned baseline also receives the same extra federated rounds in experiments/compression_study.py.
