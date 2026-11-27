# EdgeSentry DP-FedAvg: privacy / utility trade-off

Rounds = 6, delta = 1e-05, RDP accountant (client-level). sigma = 0 means clipping only, no noise (epsilon = infinity).

| Clients | sigma | clip C | epsilon | accuracy (mean +/- std) |
|---:|---:|---:|---:|---:|
| 3 | 0.0 | 1.68 | inf | 0.850 +/- 0.000 |
| 3 | 0.1 | 1.68 | 418.5 | 0.626 +/- 0.000 |
| 3 | 0.3 | 1.68 | 71.1 | 0.304 +/- 0.000 |
| 12 | 0.0 | 1.22 | inf | 0.290 +/- 0.000 |
| 12 | 0.1 | 1.22 | 418.5 | 0.454 +/- 0.000 |
| 12 | 0.3 | 1.22 | 71.1 | 0.709 +/- 0.000 |

More clients -> the same epsilon costs far less accuracy, because the noise on the averaged update shrinks as 1/K. With only 3 clients there is no useful privacy/utility operating point; this is a property of DP-FedAvg, not a bug.
