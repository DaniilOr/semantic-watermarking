# Prefix-conditioned world-model watermarking with Qwen3.5-0.8B: approach and results

Development-scale summary, 14 September 2026. Everything below was measured on the same 60 development prompts
(20 each from WritingPrompts, Dolly and C4 realnewslike), one generation seed, 512-token outputs unless stated,
with `Qwen/Qwen3.5-0.8B` (sha `2fc0636…`) as generator, world model, scorer and paraphraser. Numbers are
development estimates: about 16 configurations were compared on the same prompts and keys, calibrations are
"toy" (built from the run's own clean outputs), and no human read the texts. The spec's confirmation stage
(fresh keys, 2,100 clean calibration texts, 600 fresh prompts) is implemented but was not run.

## 1. The idea

A generator answers an ordinary task. A **key** is a secret text prefix. The same checkpoint, conditioned on a fixed
scoring context that contains the key (`Background: <key>`) or a neutral line, defines two "worlds". The
watermark makes the generated text more likely under the keyed world than under the neutral one, and the detector
recovers that from the finished text and the key alone, without the prompt, the trajectory or any classifier.

Per-token signal: `d_t = log p_key(t | text_<t) − log p_neutral(t | text_<t)`, computed by teacher-forcing the text
under the two scoring contexts. The spec's whole-text score is `F_k(y) = mean_t d_t`.

## 2. What was built, in order

1. **Spec method (candidate selection).** Every 16 tokens, sample N=8 candidate continuations from the task-conditioned
   generator and pick one by softmax over the standardized gain in `F_k`. Result on the 150-prompt dev set: real but
   weak. Paired shift +0.049 score units (1.3 null SD), AUROC 0.89 against clean text, roughly 5–15% of texts
   detectable at 1% false positives, zero leakage, no quality cost. Selection is weak because unsteered candidates
   barely differ in `d_t`.
2. **Token-level product-of-experts steering (`poe_steer`).** At every token, multiply the generator's distribution by
   `exp(β_t · clip(d_t, ±c))`. Three contexts (generator, keyed world, neutral world) advance in lockstep with their own
   caches. An entropy gate scales β_t down on confident tokens. This is the mechanism all later results use.
3. **Guards.** Leak guard: the key's own words (all surface variants) get zero push. Repetition guard: a token that
   would complete an n-gram already in the text gets zero push (n=4). EOS pin: the stop probability is kept exactly at
   the generator's value (merely zeroing its contrast made outputs 8% longer through renormalisation).
4. **Adaptive strength.** β is scaled by the running within-text t-statistic of the accepted tokens: a linear ramp
   toward a target t (`tstat`) or full strength until the target then off (`bang`). Each text is pushed only as hard
   as its own detection needs.
5. **Seed-phrase keys.** Random unrelated common words (`kelp dune mesa oyster`, 4 or 10 words) instead of persona
   statements. They attribute better and remove topic drift by construction. A low-incidence vocabulary (words
   occurring in ≤1 per 1,000 reference texts) makes incidental key-word matches negligible.
6. **Decoy-relative push.** Replace `log p_neutral` in the push by the mean over three fixed same-family decoy phrases.
   This removes the component shared by "any background phrase" and makes the push key-specific.

## 3. Detection statistics

| statistic | answers | how |
|---|---|---|
| whole-text score `F_k` | spec's primary; presence | mean of `d_t`; rank p-value against clean texts |
| **t-statistic** | presence ("watermarked at all?") | `mean(d_t) / (sd(d_t)/√n)`; threshold from a pooled clean null (60 clean texts × 12 keys = 720 values, correlated) |
| **same-family decoy rank** | attribution ("which key?") | the key's `F_k` ranked against 100 decoy phrases of the same construction; "beats all" has a floor of 1/101; a wrong key is exchangeable with a decoy, so its false attribution is at chance by construction |
| decoy-standardized z | attribution, continuous | `(F_key − mean_d F_d) / sd_d F_d`, threshold from clean texts |
| decoy-relative t | attribution, token level | t of `log p_key − mean of 8 decoys`; weaker than the rank test |

Two findings shaped the detector choice:

- The t-statistic carries a **shared component**: strong neutral-push steering makes a text look "watermarked by some
  phrase" under any key. Under a wrong key, the t-detector flags 19–69% of watermarked texts at the clean 1%
  threshold. It answers presence, not attribution.
- The original 100-decoy bank (persona sentences and digit strings) is **family-mismatched** for random-word keys and
  inflated the rank test (a wrong random-word key beat all persona decoys 20–38% of the time). With a same-family
  bank of 100 random-word phrases, wrong keys sit at 1–2.5% everywhere (chance 1%).

## 4. Results, seed-phrase keys, 512 tokens, 60 prompts

`detect` = t-detector true-positive rate at 1% clean FPR; `FA` = t-detector flagging under a wrong key; `attr` =
correct key beats all 100 same-family decoys; `attr_z` = continuous attribution at 1%; `cost` = paired change in
task-conditioned mean log-likelihood per token (95% CI about ±0.1, so differences below ~0.13 are noise); `leak` =
outputs mentioning a key word (clean: 2–3%); `loops` = outputs with an 8-word sequence repeated (clean draws: 3–12%).
All rows use the leak guard, EOS pin and, where marked, the repetition guard (r4).

| configuration | detect | FA | attr | attr_z | cost | leak | loops |
|---|---|---|---|---|---|---|---|
| β=2, clip 1 | 22% | – | – | – | −0.10 | 3% | – |
| β=5 fixed | 62% | 19% | 48% | 48% | −0.16 | 7% | 12% |
| β=5, r4 | 55% | 24% | 43% | 33% | −0.02 | 5% | 12% |
| β=5, bang-bang t4, r4 | 78% | 42% | 33% | 30% | −0.19 | 3% | 13% |
| β=5, clip 1.0, r4 | 82% | 52% | 65% | 60% | −0.15 | 17% | 13% |
| β=5, **decoy push**, bang t4, r4 | 38% | 4% | 47% | 40% | −0.06 | 3% | 13% |
| β=10 fixed | 93% | 69% | 78% | 70% | −0.35 | 20% | 13% |
| β=10, ramp t4 | 95% | 26% | 60% | 53% | −0.21 | 12% | 12% |
| β=10, ramp t6 | 93% | 34% | 70% | 60% | −0.17 | 10% | 12% |
| β=10, ramp t4, r4 | 95% | 43% | – | – | −0.23 | 7% | 10% |
| β=10, ramp t4, r4, maintenance push 0.3 | 92% | 52% | 58% | 50% | −0.20 | 8% | 8% |
| β=10, ramp t4, r4, uncertain positions only | 53% | 18% | 27% | 23% | −0.14 | 3% | 10% |
| β=10, bang t4, r4 | 92% | 55% | 62% | 58% | −0.30 | 3% | 15% |
| β=10, bang t3, r4 | 93% | 49% | 65% | 55% | −0.26 | 3% | 12% |
| **β=10, decoy push, bang t4, r4** | 75% | 9% | 73% | 68% | −0.20 | 2% | 10% |
| β=10, decoy push, clip 1.0, bang t4, r4 | 78% | 11% | 82% | 78% | −0.34 | 8% | 7% |
| **β=15, decoy push, bang t4, r4** | 77% | 14% | 80% | 82% | −0.25 | 5% | 7% |

Reading:

- Strength sets presence detection: every β=10 neutral push lands at 92–95%, every β=5 at 55–82%.
- Controller shape (ramp, bang-bang, target, maintenance floor) moves cost and attribution by a few points within the
  noise. It is not the lever.
- The decoy-relative push trades presence detection (75% at β=10) for key-specific attribution (73%, false attribution
  in single digits), the lowest leakage and loop rates, neutral length, and the lowest cost at that strength. It is the
  most balanced configuration measured.
- The repetition guard keeps key words at the clean rate; its apparent cost saving at β=5 did not replicate.
- More attribution can be bought two ways: a wider clip (1.0) or more strength (β=15). Both reach about 80%
  attribution; β=15 at clip 0.5 does it at lower cost (−0.25 vs −0.34) and less leakage (5% vs 8%). With the neutral
  push, clip 1.0 leaks 17%.

Persona-style keys (the spec's five families) reached similar presence detection at the same β but drifted toward the
key's topic (category vocabulary in 50% of outputs vs 37% clean) and attributed worse; they were dropped after the
seed-phrase runs.

### Length (β=10 decoy push, both arms capped)

| tokens | detect (t) | detect (decoy-relative t) | attr | attr_z | cost |
|---|---|---|---|---|---|
| 128 | 43% | 63% | 65% | 52% | −0.22 |
| 256 | 62% | 67% | 80% | 73% | −0.22 |
| 512 | 75% | 67% | 73% | 68% | −0.20 |

### Robustness (AUROC of the t-detector against attacked clean texts; same-model paraphraser)

| attack | β=5 ramp, neutral push (unattacked 0.96) | β=10 decoy push (unattacked 0.93) |
|---|---|---|
| light paraphrase | 0.92 | 0.92 |
| strong paraphrase | 0.86 | 0.88 |
| sentence resegmentation | 0.94 | 0.91 |
| paragraph reordering | 0.84 | 0.86 |
| dilution, append 100% clean text | 0.95 | 0.92 |
| dilution, interleave 100% | 0.90 | 0.87 |
| crop, keep first 50% | 0.94 | 0.91 |
| crop, keep last 50% | 0.76 | 0.82 |
| crop, keep middle 50% | 0.78 | 0.82 |

The mark is front-loaded under adaptive control (the controller pushes hard early and eases off), which is why
tail crops hurt most. A maintenance push after the target did not fix it (0.76 → 0.75).

At the **unchanged clean-text threshold** (spec Section 10: the detector and its 1% threshold are frozen before any
attack; `ctrl FPR` = attacked clean texts above that threshold), β=10 decoy push, 60 texts per cell
(`scripts/attack_proof.py`):

| attack | TPR at frozen threshold | ctrl FPR | AUROC [95% CI] | words changed |
|---|---|---|---|---|
| none | 75% | 0% | 0.93 | – |
| light paraphrase | 72% | 0% | 0.92 [0.88, 0.96] | 7% |
| strong paraphrase | 50% | 0% | 0.88 [0.82, 0.93] | 39% |
| resegmentation | 72% | 2% | 0.91 [0.86, 0.95] | 2% |
| reordering (42 applicable) | 57% | 0% | 0.86 [0.78, 0.93] | 36% |
| dilution, append 25 / 50 / 100% | 72 / 67 / 70% | 0–2% | 0.94 / 0.92 / 0.92 | 13 / 22 / 36% |
| dilution, interleave 25 / 50 / 100% | 63 / 62 / 57% | 0–2% | 0.91 / 0.89 / 0.87 | 13 / 22 / 36% |
| crop, keep first 50 / 75% | 63 / 70% | 2% | 0.91 / 0.92 | – |
| crop, keep last 50 / 75% | 33 / 33% | 2–3% | 0.82 / 0.82 | – |
| crop, keep middle 50 / 75% | 28 / 40% | 3% | 0.82 / 0.86 | – |

"Words changed" is the word-level edit fraction between parent and attacked text: the light paraphrase is a light
edit (7%), so its result says little; the strong paraphrase rewrites 39% of words and halves the detection rate while
keeping it 50× above the false-positive rate. Paraphrases are same-model and unreviewed; a subset must be checked by
a human for meaning preservation before any of this is quoted as robustness.

### Quality

- Likelihood cost is real (a second clean draw of the same prompts differs from the first by 0.004 nats/token) but
  modest: 0.05–0.2 for β=5-class settings, 0.2–0.35 for β=10-class.
- Word-level loops stay within the range spanned by clean draws; the earlier "repetition rises" reading came from a
  4-gram metric inflated by markdown bullets.
- Length is neutral with the EOS pin; topic drift is absent with random-word keys; key-word mentions are at the clean
  rate with the leak guard at clip 0.5.
- Not measured: human preference, IFEval, an independent paraphraser.

## 5. Things that did not work

Static activation steering (destructive before detectable); decoy-relative candidate selection (no gain, 2.7× cost);
KL- and likelihood-budget gates (weak signal); "many small pushes" (β=10, clip 0.2: no cheaper at full length);
uncertainty-only steering; the maintenance push; the low-incidence vocabulary (clean leakage measurement, no
detection gain); the token-level decoy-relative detector (weaker than the rank test).

## 6. Caveats

- 60 prompts, 12 keys, one seed, toy calibrations, many configurations compared on the same data: the best rows are
  optimistically biased. The confirmation pipeline exists to fix this and has not been run.
- Presence is semi-public with the neutral push: anyone with the model and any background phrase can tell a text was
  steered (that is the shared component). The key protects attribution, not existence. The decoy push reduces this.
- Keys are low-entropy phrases, not secrets in a cryptographic sense.
- Detection needs the generating checkpoint (two or more forward passes over the text).
- The paraphraser shares weights with the generator, so paraphrase numbers are the optimistic end.

## 7. Recommended settings

- "Is this text watermarked with key K?": β=10, clip 0.5, adaptive ramp, leak guard, EOS pin. 93–95% at 1% clean
  FPR on 512-token texts; do not use it to decide between keys.
- "Which key, if any?": decoy-relative push, clip 0.5, bang-bang t=4, both guards, at β=10 (75% presence, 73%
  attribution, false attribution 3–9%, cost −0.20, key words at the clean rate) or β=15 (77% presence, 80–82%
  attribution, false attribution 3–14%, cost −0.25, 5% key words). β=10 is the cleaner text, β=15 the stronger
  attribution; the difference in cost is within the ±0.1 noise.

## 8. Where things are

- Code: `~/wmworld` (git). Extensions are documented in the README under "Extensions beyond the specification".
- Runs: `runs/rw_*` (seed-phrase keys, 512 tokens), `runs/rw128_*`, `runs/rw256_*` (length sweep),
  `runs/attacks_*` (attack suites), `runs/abl_*` (persona keys), `runs/dev` (spec method, 150 prompts).
- Tables: `scripts/summary_table.py --bank decoy_rw runs/...` (consolidated), `scripts/poe_compare.py`,
  `scripts/attack_table.py`, `scripts/rank_false_attribution.py --bank decoy_rw`, `scripts/t_null_pool.py` (GPU).
- Keys: `assets/keys_randomwords.json` (targets), `assets/keys_decoy_rw.json` (same-family decoys),
  `assets/steering_decoys.json` (push decoys), `assets/keys_confirmation.json` (16 fresh keys, unused).
- Frozen draft protocol for the confirmation stage: `runs/frozen_protocol_draft.json` (`slurm/freeze_draft.sh`).
