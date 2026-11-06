# Using Modal for this project

The smoke test already ran in workspace `zhengfel`. A teammate can reuse that
workspace, volume, and image. Cloning the Git repository alone is not enough:
the Bailando checkout, checkpoints, audio, and run arrays are intentionally
not committed.

## One-time setup

1. Install the local orchestration environment from [environment.yml](../environment.yml).
   The environment name is `10703`. Python 3.11 is required. Do not use the
   system Python.

   ```sh
   conda env create -f environment.yml
   conda activate 10703
   ```

2. Check out the upstream source at the pinned commit:

   ```sh
   git clone https://github.com/lisiyao21/Bailando.git third_party/Bailando
   git -C third_party/Bailando checkout cc90b98bff81c9709570db413c9610c2562e27ca
   ```

3. Authenticate to the shared workspace. From the project directory:

   ```sh
   modal token new
   ```

   The browser login must end on workspace `zhengfel`. A personal workspace
   cannot see this project's volume. Modal also requires a payment method on
   that workspace before it will start a T4, even when credits later cover the
   charge.

4. Confirm the account gate:

   ```sh
   python scripts/check_modal_account.py
   ```

   It prints whether a launch is allowed. It does not print the token.
   Credit expiration is not available from Modal's billing API. The gate allows
   a run while this month's metered cost is under $4, inside the $5 setup
   allowance. The first-batch ceiling remains $10 including CPU, RAM, and image
   builds.

## What is already remote

Volume `bailandopp-batch01` contains the supervised policy, the position
decoder, the mHO5 motion record, its 60 Hz features, and `mHO5.wav`. The
completed smoke test is `smokeB-20261106T204747Z`. Its clip is
[reports/batch01/smoke_mHO5_skeleton.mp4](../reports/batch01/smoke_mHO5_skeleton.mp4).
The large arrays are under `runs/batch01/`, which Git ignores; the same files
are on the volume at `runs/smokeB-20261106T204747Z/gpu/`.

Initialization I for later comparisons is upper code 436 and lower code 289.
Do not regenerate that clip for the music experiment. Reuse it.

## Run the existing smoke test

```sh
python scripts/launch_batch01_smoke.py
```

This checks the local crop, checks the account gate, then runs one CPU load
and one T4 generation. The function uses a 15-minute timeout, no retries, and
one T4. It tries Python 3.8 with PyTorch 1.6.0+cu101 first. The image also
contains Python 3.11 with PyTorch 2.4.1+cu118 as a fallback. The November 6
run succeeded on the legacy interpreter, so the fallback was not used.

The terminal should not ask for input. If an image build stops on a timezone
question, stop it. The image definition already sets UTC and a noninteractive
package install.

## Cost recorded on November 6

| Item | Amount |
|---|---|
| Workspace metered cost | $0.03039 |
| Billed cost | $0.00 |
| Credits applied | $0.03000 |
| Successful run's T4 / CPU / memory | $0.00492 / $0.00231 / $0.00153 |

Rates read from Modal that day: T4 $0.590 per hour, CPU $0.0473 per core-hour,
memory $0.008 per GiB-hour. A new smoke run is a few cents after the image
exists. Do not start a second GPU size if a run fails.

## Do not commit

Tokens live in the Modal profile (`~/.modal.toml`), not in the repository.
Weights, audio, the upstream tree, and `runs/` stay out of Git.
