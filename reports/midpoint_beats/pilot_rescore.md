# Cached pilot rescoring

Reproduced 24 of 24 cached dances. Missing: 0. New generations: 0.

Rows weight music IDs equally after averaging starts. Historical diversity compares the two starts within a song. Legacy BAS is shown only as a raw-coordinate diagnostic. The position panels exclude the first eight conditioning frames.

## legacy_full

| Policy | BAS | Cov. 100 | P 100 | R 100 | F1 100 | Bin hit | Events/s | Speed | RMS jerk | Stationary | Diversity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adaptation / supervised | 0.219 | 0.351 | 0.354 | 0.351 | 0.348 | 0.255 | 1.673 | 0.722 | 392.560 | 0.000 | 0.874 |
| adaptation / epoch5 | 0.241 | 0.386 | 0.401 | 0.386 | 0.387 | 0.242 | 1.623 | 0.709 | 388.802 | 0.000 | 1.172 |
| held_out_from_rl / supervised | 0.234 | 0.385 | 0.335 | 0.385 | 0.354 | 0.251 | 1.867 | 0.648 | 351.477 | 0.035 | 2.065 |
| held_out_from_rl / epoch5 | 0.235 | 0.400 | 0.333 | 0.400 | 0.361 | 0.252 | 1.942 | 0.712 | 369.077 | 0.000 | 2.611 |

## position_full

| Policy | BAS | Cov. 100 | P 100 | R 100 | F1 100 | Bin hit | Events/s | Speed | RMS jerk | Stationary | Diversity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adaptation / supervised | 0.198 | 0.345 | 0.381 | 0.345 | 0.356 | 0.219 | 1.562 | 0.722 | 392.254 | 0.000 | 0.888 |
| adaptation / epoch5 | 0.209 | 0.339 | 0.362 | 0.339 | 0.344 | 0.190 | 1.522 | 0.709 | 388.481 | 0.000 | 1.168 |
| held_out_from_rl / supervised | 0.218 | 0.374 | 0.333 | 0.374 | 0.346 | 0.244 | 1.772 | 0.648 | 351.139 | 0.035 | 2.065 |
| held_out_from_rl / epoch5 | 0.225 | 0.399 | 0.335 | 0.399 | 0.362 | 0.233 | 1.914 | 0.713 | 368.869 | 0.000 | 2.613 |

## position_20s

| Policy | BAS | Cov. 100 | P 100 | R 100 | F1 100 | Bin hit | Events/s | Speed | RMS jerk | Stationary | Diversity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adaptation / supervised | 0.192 | 0.346 | 0.391 | 0.346 | 0.363 | 0.194 | 1.625 | 0.732 | 391.303 | 0.000 | 1.725 |
| adaptation / epoch5 | 0.190 | 0.337 | 0.382 | 0.337 | 0.352 | 0.173 | 1.550 | 0.727 | 385.232 | 0.000 | 1.645 |
| held_out_from_rl / supervised | 0.261 | 0.453 | 0.377 | 0.453 | 0.404 | 0.287 | 2.058 | 0.738 | 359.850 | 0.001 | 2.347 |
| held_out_from_rl / epoch5 | 0.239 | 0.420 | 0.348 | 0.420 | 0.373 | 0.245 | 2.067 | 0.751 | 364.810 | 0.000 | 2.672 |

