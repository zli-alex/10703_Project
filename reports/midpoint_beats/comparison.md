# Controlled comparison

Completed 1,600 training, 252 validation, and 120 final test continuations.

| Policy | BAS | Cov. 100 | P 100 | R 100 | F1 100 | Bin hit | Events/s | Speed | RMS jerk | Stationary | Diversity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen | 0.199 | 0.332 | 0.329 | 0.332 | 0.328 | 0.206 | 1.685 | 0.674 | 364.324 | 0.001 | 1.714 |
| bin RL, seed mean | 0.212 | 0.366 | 0.361 | 0.366 | 0.361 | 0.228 | 1.689 | 0.690 | 370.102 | 0.000 | 1.544 |
| bas RL, seed mean | 0.203 | 0.347 | 0.345 | 0.347 | 0.344 | 0.222 | 1.698 | 0.671 | 361.816 | 0.001 | 1.607 |
| bin-42 | 0.207 | 0.358 | 0.356 | 0.358 | 0.355 | 0.217 | 1.671 | 0.702 | 374.355 | 0.000 | 1.449 |
| bin-43 | 0.216 | 0.374 | 0.365 | 0.374 | 0.367 | 0.240 | 1.706 | 0.677 | 365.850 | 0.000 | 1.639 |
| bas-42 | 0.193 | 0.333 | 0.333 | 0.333 | 0.331 | 0.213 | 1.683 | 0.674 | 363.609 | 0.001 | 1.569 |
| bas-43 | 0.213 | 0.362 | 0.357 | 0.362 | 0.357 | 0.232 | 1.713 | 0.668 | 360.022 | 0.000 | 1.645 |

Diversity compares stochastic draws within the same song and start. Song bootstrap intervals in comparison.json are conditional on each seed.

## Paired primary outcomes

- BAS seed mean vs baseline, bas +0.0033 (song interval -0.0258 to +0.0325)
- BAS seed mean vs baseline, f1_100ms +0.0155 (song interval -0.0257 to +0.0567)
- BAS vs bin, seed mean, bas -0.0087 (song interval -0.0207 to -0.0009)
- BAS vs bin, seed mean, f1_100ms -0.0173 (song interval -0.0419 to +0.0018)
- bin-42 vs baseline, bas +0.0074 (song interval -0.0158 to +0.0305)
- bin-42 vs baseline, f1_100ms +0.0268 (song interval +0.0007 to +0.0491)
- bin-43 vs baseline, bas +0.0167 (song interval -0.0027 to +0.0362)
- bin-43 vs baseline, f1_100ms +0.0388 (song interval +0.0157 to +0.0661)
- bas-42 vs baseline, bas -0.0067 (song interval -0.0336 to +0.0202)
- bas-42 vs baseline, f1_100ms +0.0029 (song interval -0.0277 to +0.0334)
- bas-43 vs baseline, bas +0.0133 (song interval -0.0180 to +0.0447)
- bas-43 vs baseline, f1_100ms +0.0281 (song interval -0.0238 to +0.0799)
- BAS vs bin, seed 42, bas -0.0140 (song interval -0.0253 to -0.0066)
- BAS vs bin, seed 42, f1_100ms -0.0239 (song interval -0.0405 to -0.0092)
- BAS vs bin, seed 43, bas -0.0034 (song interval -0.0162 to +0.0085)
- BAS vs bin, seed 43, f1_100ms -0.0107 (song interval -0.0408 to +0.0193)
