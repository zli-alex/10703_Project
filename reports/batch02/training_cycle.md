# Bailando++ training-cycle benchmark

November 16, 2026, New York time. Run `cycle-20261117T043240470937Z` completed on a
Tesla T4 with Python 3.8 and PyTorch `1.6.0+cu101`. The benchmark started from the
released supervised policy, the position decoder, and a seed-42 fresh critic. It
used the full native feature sequences for `mHO5` and `mJB5`, the two development
songs. [Modal run](https://modal.com/apps/zhengfel/main/ap-4LsjHFVFGucV1p8frYJmI6)

The earlier [isolated update run](../../runs/batch01/updateM4-20261106T224545Z/update_results.json)
matched independent loss calculations and verified the intended head updates with
fresh Adam state. This benchmark used one model and one Adam optimizer for the
entire epoch. The released rollout path zeroed the pose root before encoding and
selected codes greedily. The training dataset supplied 403 windows (211 and 192
respectively). Batch size 44 was the largest that yielded at least ten minibatches
on these inputs. The final minibatch contained seven windows. The benchmark used
zero DataLoader subprocesses and retained the released replacement sampler.

| Observation | Result |
|---|---|
| Minibatches 0–4 | Actor updates |
| Minibatches 5–9 | Critic updates |
| First actor minibatch | Actor changed with 100 nonzero gradient tensors. Critic had no gradient tensors and stayed unchanged. |
| First critic minibatch | Critic changed with 52 nonzero gradient tensors. Actor also changed with 100 present, all-zero gradient tensors. |
| Frozen components | Music encoder, GPT base, and decoder state fingerprints matched before and after the epoch. |
| Checkpoint | Restored model, Adam, scheduler, RNG, and progress and compared them with the saved epoch boundary. A second epoch was outside this run. |

At the first critic step, legacy `optimizer.zero_grad()` cleared the actor's
existing gradients to zeros. Adam still processed those tensors, so its stored
actor momentum could move actor weights during the critic step. This explains why
the isolated critic step, which began with fresh Adam state, showed no actor
movement. [Measured results](benchmark_results.json)

| Measured phase | Time |
|---|---:|
| Greedy rollout of both songs | 7.90 s |
| Decode and reward scoring over ten minibatches | 2.81 s |
| Forward, backward, and optimizer steps | 13.58 s |
| Training loop wall time including instrumentation | 19.69 s |
| Checkpoint save / reload and comparison | 3.13 s / 3.51 s |
| GPU worker, including setup and verification | 61.23 s |

Peak allocated GPU memory was 10.06 GiB. The checkpoint is 1,057,384,893 bytes and
is stored on Modal volume `bailandopp-batch01` at
`runs/cycle-20261117T043240470937Z/benchmark_checkpoint.pt`. Modal itemized
**$0.01248** for this app ($0.00983 T4, $0.00158 CPU, $0.00107 memory). Storage is
separate. At the retrieved rates, the resource total is $0.7486 per worker-hour.

Scaling the measured worker time by 1.5 for a third comparable-length adaptation
song at batch size 44 gives about **$0.02 per epoch in worker resources**, or
**$0.10 for five epochs**, before storage, startup differences, and matched
evaluation. This is a budgeting estimate. The third adaptation song, three
RL-held-out songs, and a representative batch-size check for that split are still
pending. Both teammates' walkthroughs and human audiovisual review also remain
open.

To reproduce using Conda environment `10703` and the staged Modal volume:

```sh
python scripts/check_modal_account.py
modal run -m infra.modal_benchmark
```
