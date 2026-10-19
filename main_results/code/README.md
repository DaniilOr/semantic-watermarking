# wmworld — prefix-conditioned world-model watermarking (Qwen3.5-0.8B test specification v1.1)

Experiment package testing whether a generator can be steered, by candidate selection against a whole-text
likelihood-contrast score under a secret-prefix-conditioned "world", so that a detector recovers a key-specific
signal from the finished text and key alone. All model-backed roles use one checkpoint:
`Qwen/Qwen3.5-0.8B` @ `2fc06364715b967f1860aea9cf38778875588b17`. A reproducible negative result is a valid outcome.

## Environment (locked)

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda create -y -n wmworld python=3.11
conda activate wmworld
pip install -r requirements.lock            # exact versions used (torch 2.9.0+cu128, transformers 5.17.0, ...)
python -c "import nltk; nltk.download('punkt_tab')"   # only for the IFEval evaluator
export HF_HUB_OFFLINE=1                      # after the one-time downloads below
```

One-time pinned downloads (model snapshot at the sha above into the HF cache; datasets into `data/raw/`;
IFEval evaluator sparse-checked-out at commit `e6890f8…` into `third_party/google_research_ifeval`):
see `scripts/download_pins.sh`.

Runtime facts that matter: the checkpoint is multimodal but is loaded into the text-only `Qwen3_5ForCausalLM`
(vision and MTP weights ignored; zero missing keys is asserted). Weights run in **float32** with the reference
PyTorch Gated-DeltaNet path and TF32 disabled: measured bf16 batched-vs-single score discrepancies reached
0.037 nats/token, which the spec's floor rule (100 × discrepancy) would turn into "no usable scale"; fp32 agrees
to ~2e-6 at ≈1.5× scoring cost and no sampling cost (3.7 GB peak on an A100-40GB). The chat template is rendered
with `enable_thinking=False`; its fixed `<think>\n\n</think>\n\n` block is context, never scored. Generation stops
on both `<|im_end|>` (248046) and `<|endoftext|>` (248044).

## Commands (each processes one stage; all resumable; outputs under `runs/`)

```bash
P="python -m wmworld.cli"
$P prepare-data --config configs/pilot.yaml                       # fixed filters, dedup, grouped partitions -> data/prepared
$P make-keys    --config configs/pilot.yaml --set development     # 20 keys (4 per family), rarity-scored -> assets/keys_development.json
$P selftest     --config configs/smoke.yaml                       # model-backed checks (GPU or CPU); writes runs/selftest.json
$P fit-scorer   --config configs/smoke.yaml --partition smoke --toy   # toy sigma table for smoke -> assets/scales_smoke.json
$P generate     --config configs/smoke.yaml --stage smoke --dry-run
$P generate     --config configs/smoke.yaml --stage smoke
$P evaluate     --config configs/smoke.yaml --run runs/smoke      # scores under all keys, toy calibration, tables/plots/report/samples
$P fit-scorer   --config configs/pilot.yaml --partition scale_fit # frozen sigma table -> assets/scales.json (+ diagnostics)
$P generate     --config configs/pilot.yaml --stage dev --dry-run # budget manifest (required before large stages)
$P generate     --config configs/pilot.yaml --stage dev --shard 0 --num-shards 3   # one GPU per shard
$P evaluate     --config configs/pilot.yaml --run runs/dev
$P generate     --config configs/pilot.yaml --stage ablation --run-name ablation_N16 --override '{"candidates": 16}'
$P freeze       --config configs/pilot.yaml --run runs/dev --output runs/frozen_protocol.json \
                --select '[{"arm": "llm_global", "overrides": {}, "families": ["common_single", "rare_single"]}]' --rationale '...'
$P make-keys    --protocol runs/frozen_protocol.json --set confirmation
$P generate     --protocol runs/frozen_protocol.json --stage null_calibration --dry-run   # then without --dry-run (shardable)
$P calibrate    --protocol runs/frozen_protocol.json               # frozen thresholds -> assets/calibration.json
$P generate     --protocol runs/frozen_protocol.json --stage null_test
$P generate     --protocol runs/frozen_protocol.json --stage positive_test
$P topic-null   --protocol runs/frozen_protocol.json
$P attack       --protocol runs/frozen_protocol.json --run-name positive_test
$P evaluate     --protocol runs/frozen_protocol.json --run runs/positive_test   # also runs/null_test, runs/topic_null, runs/attacks_positive_test
$P ifeval       --protocol runs/frozen_protocol.json               # generation; then --evaluate-only for the pinned evaluator
$P report       --protocol runs/frozen_protocol.json               # runs/report/{report.md,results.json,figures}
$P detect --text-file sample.txt --key-id confirmation_003 --calibration assets/calibration.json
```

`freeze` never selects automatically: it requires `--select` and a written rationale. Development stages use
`--config`; confirmation stages use `--protocol` (the frozen manifest, content-hashed).

### Slurm (cluster: `cscc-gpu-p`, QoS `cscc-gpu-qos`, 4 jobs per user; CPU work on `cscc-cpu-p`)

Nothing runs on the login node. CPU-only steps (tests, data preparation, key building, analysis scripts,
`evaluate --no-score`) go through the CPU partition:

```bash
sbatch --job-name=wm_tests --export=ALL,CMD="python -m pytest -q tests" slurm/cpu.sbatch
sbatch --job-name=wm_prep  --export=ALL,CMD="python -m wmworld.cli prepare-data --config configs/pilot.yaml" slurm/cpu.sbatch
sbatch --job-name=wm_keys  --export=ALL,CMD="python -m wmworld.cli make-keys --config configs/pilot.yaml --set development --device cpu" slurm/cpu.sbatch
```

```bash
sbatch slurm/smoke.sbatch                                                     # selftest + toy fit + smoke + evaluate
sbatch slurm/fit_scorer.sbatch
sbatch --job-name=wm_dev0 --export=ALL,STAGE=dev,SHARD=0,NUM_SHARDS=3 slurm/stage.sbatch   # repeat for shards 1,2
sbatch --job-name=wm_eval_dev --export=ALL,RUN=runs/dev slurm/evaluate.sbatch
```

Scripts must not call `nvidia-smi`/`scontrol`/`squeue` (the submit filter rejects them) and write logs to `slurm_logs/`.

### Extensions beyond the specification (labelled amendments)

Development showed the spec's candidate-selection steering to be real but weak (dev: paired shift +0.05 score
units, ~1.3 null SD). The following extensions are implemented on top of it and reported under their own labels;
none of them changes the spec's five core arms.

Generation arms / overrides (`--arms`, `--override '{...}'`):

- `poe_steer` — token-level product-of-experts steering: at every generated token the generator's next-token
  distribution is multiplied by `exp(beta_t * clip(log p_key(t) - log p_neutral(t), ±poe_clip))`, where the
  key/neutral log-probabilities come from the two scoring contexts (same weights, separate caches). `poe_gate`
  = `entropy` (beta ramps from 0 at `poe_entropy_min` to `poe_beta` at `poe_entropy_ref` nats; confident tokens are
  not touched), `lp` (grid search on an expected base log-likelihood loss), or `none`. `poe_steer_eos: false`
  pins p'(EOS) to the generator's p(EOS) exactly (steered otherwise, the key context, which never sees the task,
  favours stopping early; merely zeroing its contrast made outputs longer through renormalisation).
  Guards and control: `poe_block_key_tokens` (the key's own words are never pushed), `poe_block_category` (nor any
  catalog word of the key's categories), `poe_block_repeat_n` (nor a token completing an n-gram already in the text),
  `poe_adapt: tstat` with `poe_t_target`/`poe_beta_floor` (beta scaled by how far the running within-text t-statistic
  is below the target: each text is pushed only as hard as its detection needs). Per-token entropy, beta, contrast,
  raw contrast, running t, generator log-prob and KL are stored in the private trajectory.
- `objective: specific` — decoy-relative gain `F_k − mean_d F_d` over five fixed same-family decoys
  (`assets/steering_decoys.json`; scales from `fit-scorer --objective specific`).
- `selection: greedy | rank | softmax | uniform`, `center_gains` — candidate-selection variants.
- `activation_steer`, `steer_select` — residual-stream steering vectors (`steer_alpha`, `steer_layers`); tested
  and rejected (destructive before detectable).
- `base` never receives overrides, so it stays the unwatermarked control in every ablation run.

Key sets: `make-keys --set decoy` builds a 100-key decoy bank (never targets; feeds the rank detector) and
`--set randomwords` builds seed-phrase keys (family `random_words`: 4 or 10 unrelated common words drawn by PRNG,
e.g. `harbor velvet compass ember`), the user-preferred alternative to persona-like statements. Stages
`ablation` (development keys) and `ablation_rw` (seed-phrase keys) use the frozen 60-prompt dev subset.

Secondary detectors (evaluated in `evaluate`, exposed by `detect --mode rank|both`):

- decoy-rank: the correct key's z-scored score ranked against the decoy bank on the same text (`rank_frac`,
  `z_diff`); "beats all decoys" has a discrete floor of p ≈ 1/(n_decoys+1).
- within-text t-statistic: `t = mean(d)/(sd(d)/sqrt(n))` over per-token contrasts `d_t = log p_key − log p_neutral`.

```bash
sbatch slurm/abl_poe2.sbatch     # 512-token PoE ablations on the 60-prompt subset (base + poe_steer, then evaluate)
sbatch slurm/abl_poe3.sbatch     # EOS-excluded steering; seed-phrase keys (make-keys --set randomwords + ablation_rw)
sbatch slurm/abl_rw.sbatch       # seed-phrase keys: reference settings + second clean draw (quality null) + pooled-null t detector
sbatch slurm/abl_rw3.sbatch      # adaptive controller variants;  abl_rw4.sbatch: repetition guard
python scripts/poe_compare.py runs/rw_poe5c05_g runs/rw_poe5c05_g_ad     # cross-run table (CPU queue)
```

### Standalone detector

`detect` needs only the calibration manifest (which embeds the key specs and scorer configuration), the model files,
and the text. Copy `assets/calibration.json` and the text into a clean directory and run the command above; the
output JSON has the raw score, component ranks, combined rank, decision (`detected` / `not_detected` /
`abstain_short` / `abstain_overflow` / `error`), eligibility status and scorer/calibration ids.

## Layout

- `wmworld/runtime.py` shared Qwen3.5 runtime (one weight load; per-role contexts; fp32 chunked head; right-padded teacher forcing; cache fork gated by the self-test)
- `wmworld/scoring.py` `F_k(y) = (ell_k − ell_k0)/n_y`, abstentions, neutral-sum cache
- `wmworld/generation.py` candidate-selection loop, units, arms (`base`, `budget_control`, `passive_prefix`, `llm_global`, `llm_local`; extensions `poe_steer`, `activation_steer`, `steer_select`), private trajectories
- `wmworld/keys.py`, `key_catalog.py` template catalog, rarity, families, PRNG digits, balanced assignment
- `wmworld/data/` pinned loaders, fixed filters, MinHash dedup, grouped disjoint partitions, manifest
- `wmworld/scales.py` frozen σ per family/unit/objective/length bin with floors and fallbacks
- `wmworld/calibration.py` rank p-values, combined cohorts, known-key `detect`, Bonferroni multi-key helper
- `wmworld/stages.py`, `cli.py` enumeration, dry-run budgets, sharded resumable generation, fit-scorer, scoring
- `wmworld/evaluation.py`, `plots.py` join layer, cluster bootstrap, AUROC, confusion, length views, leakage, tables/figures, `samples.md`
- `wmworld/attacks.py`, `topic_null.py`, `quality.py`, `ifeval.py`, `protocol.py`, `report.py`
- `wmworld/backend.py` backend protocol + deterministic fake LM used by `tests/` (no model download)
- `tests/` focused tests for spec Section 14 (run `pytest -q tests`); model-backed checks live in `wmworld selftest`

## Records

`runs/<stage>/generations*.jsonl` (visible text, ids, arm, key, seed, stop reason, cost), `trajectories*.jsonl`
(private per-decision candidates/scores/gains/probabilities), `references*.jsonl`, `detections_raw.jsonl`
(text hash, key, score, eligibility — no prompt, label or arm), `eval/` (results.json, CSV tables, figures, report.md,
samples.md/csv side-by-side outputs per prompt), `budget.json`, `manifest.json`, `summary*.json`.

## Status and caveats

See `runs/report/report.md` (built by `report`) for what has actually been run. Model-assisted paraphrases and
judgments reuse the same 0.8B checkpoint and are exploratory; human review is marked pending. Development p-values
use toy calibration and are not false-positive estimates.
