# Checkpoint availability resolved

Inspected November 6, 2026 against pinned upstream revision
`cc90b98bff81c9709570db413c9610c2562e27ca`. All work used local CPU/network;
no Modal job or paid GPU resource was used.

**The official Bailando++ archive contains the supervised checkpoint required
by the released RL configuration. It and the required position VQ-VAE have been
downloaded, integrity checked, and strictly loaded on CPU.** A CUDA forward
pass and generated clip have not been run.

## Asset readiness

| Asset | Archive member | Size on disk | Findings | Status |
|---|---|---:|---|---|
| Supervised ++ policy | `cc_gpt_music_trans/ckpt/epoch_400.pt` | 455,528,241 bytes | `CrossCondGPT2MW`, 269 state entries, music encoder/base/head present, no critic state | Downloaded; CRC, SHA256, storage lengths and metadata checked |
| Position VQ-VAE | `experiments/sep_vqvae_root/ckpt/epoch_500.pt` | 262,495,255 bytes | `SepVQVAER`, 228 state entries, two 512 × 512 codebooks | Downloaded; CRC, SHA256, storage lengths and metadata checked |
| ++ rotation decoder | `sep_vqvae_root_mix_w_vel_acc/ckpt/epoch_500.pt` | 370,916,527 bytes listed | Present in ++ archive | Available; deferred |
| Released ++ RL policy | `actor_critic/ckpt/epoch_10.pt` | 540,627,091 bytes listed | Present in ++ archive | Optional reference; deferred |
| Processed data | `data.zip` | 4,103,564,785 bytes compressed | 952 training records, 40 full test records, 60 external 7.5 Hz feature files | Two full test records selectively downloaded and validated, with matching WAVs |

The two downloaded checkpoints occupy 718,023,496 bytes together. Selective
byte-range extraction transferred approximately 667 MB for those files, plus
roughly 2 MB for archive indexes; no complete multi-GB archive was stored. The
processed data would expand to approximately 9.87 GB if extracted in full, so
we retrieved only two selected records initially.

## Important identity finding: filename and epoch disagree

The policy filename is `epoch_400.pt`, exactly the filename referenced by the
released ++ RL configuration. Its internal `epoch` is **440**, and its saved
evaluation setting also points to epoch 440. Preserve both values and refer to
the file by its hash; do not report it as a verified epoch-400 training snapshot.

Evidence for supervised-policy identity:

- Stored policy class: `CrossCondGPT2MW`, rather than `CrossCondGPT2MWAC`.
- State groups: `music_trans` (56 entries), `gpt_base` (107), `gpt_head` (106).
- No `critic_net` parameters are present. RL initialization must explicitly
  initialize and report missing critic parameters; other missing actor keys
  require investigation rather than blanket acceptance of `strict=False`.
- Music input projection shape: 768 × 55. Upper/lower output projections:
  512 × 768. These agree with the configured feature count and vocabulary.
- Shared policy configuration values match the released RL configuration except
  for the expected class-name change. Shared position-decoder configuration
  values match without differences.

This is strong evidence of the intended released supervised baseline; it does
not independently establish the complete training history or provenance of every
training example.

## Integrity and inspection method

| Checkpoint | SHA256 |
|---|---|
| Supervised ++ policy | `ed84ccd47d861c9b8a8a61a60a67aa3b8cbb59c928487daec1fdb65190b02da2` |
| Position decoder | `35670f42a3b3092438f73f0af3ace7b52e318a8b5c00b2b05c92078176b21716` |

ZIP extraction verified each member's CRC. SHA256 values identify our downloaded
copies; no independently published author checksum was available for comparison.
The metadata inspector does not instantiate the model. A later CPU load with
PyTorch 2.15 accepted every policy and decoder tensor under `strict=True`.
That load is recorded in the smoke-test report and is not a GPU inference.

Machine-readable records:

- [Baseline manifest](../../configs/baseline_assets.json)
- [Downloaded asset hashes](downloaded_assets.json)
- [Supervised checkpoint metadata](supervised_checkpoint_metadata.json)
- [Position decoder metadata](position_decoder_metadata.json)
- [++ archive index](bailandopp_index.json), [original archive index](bailando_original_index.json), [data archive index](preprocessed_data_index.json)

## Data prerequisite and next work

The [AIST terms](https://aistdancedb.ongaaccel.jp/terms_of_use/) ask users to
submit the [application form](https://forms.gle/9nVAxPFUhXNPrKQ5A) before using
the dataset. The user confirmed submission on November 6, 2026. The published
terms and [download instructions](https://aistdancedb.ongaaccel.jp/getting_the_database/)
do not specify waiting for an approval email. We therefore treat submission as
satisfying this prerequisite unless the user's submission confirmation says
otherwise. No form was submitted on the user's behalf. Two motion/audio pairs
have now been downloaded and validated.

Development tracks are `mHO5` and `mJB5`, with the corresponding
single-dancer records identified in the manifest. Lengths, array values, and
audio/feature clock consistency pass; perceptual synchronization remains pending.
Original WAV links are documented
on the [official musical-pieces page](https://aistdancedb.ongaaccel.jp/all_links/all_links_musical_pieces/).

The archive does not contain a separate `aistpp_music_feat_60fps` directory.
The upstream full-record preprocessing uses 30,720 Hz audio and a default 512-sample
feature hop. The downloaded full-record JSONs contain native 60 Hz music arrays;
their durations match the WAVs within one frame. These arrays are exported without
resampling; the separate 7.5 Hz features are unused.

See [data validation](data_validation.md) for exact dimensions, hashes, revised
scoring windows, and the timeline plot. Remote-runtime preparation and actual
model loading are the next unresolved gates.
