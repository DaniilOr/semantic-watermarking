# A likelihood-steering text watermark: specification of the method as built and tested

Version of 17 September 2026. This document describes, from first principles, how the watermark is embedded, how it is
detected, how keys are made, how the system was calibrated and evaluated, and what the measurements show. It uses no
internal code names; every quantity is defined where it first appears. Numbers marked *held-out* come from the
large-sample run (600 prompts and 16 keys never used during development); numbers marked *development* come from the
60-prompt tuning set.

---

## 1. What the system does

A language model answers ordinary requests (write a story, answer a question, continue an article, write a function).
While it writes, the sampling procedure is tilted so that the finished text is slightly more probable under a
*secret reference context* than under a neutral one. A detector that has only the finished text and the secret can
measure that tilt and decide, at a chosen false-positive rate, whether the text was produced with that secret. Nothing
is inserted into the text; the visible words are the model's own choices, nudged.

One model checkpoint plays every role: it generates, it defines the two reference contexts, it scores texts at
detection time, and it rewrites texts in the robustness tests. The checkpoint is a 0.8-billion-parameter instruction
model, run in full 32-bit precision so that scores are numerically repeatable to about 10⁻⁶ per token.

---

## 2. Definitions

**Secret (key).** A short phrase of unrelated common English words, for example `kelp dune mesa oyster`, drawn at random
from a fixed vocabulary of 120 words (4 words for a short key, 10 for a long one). Nothing about the key is meaningful;
its only job is to define a reference context that other phrases do not define.

**Reference contexts.** Two fixed chat conversations that never mention the user's actual task:

- *keyed context*: a system message `You are a helpful assistant.` followed by the line `Background: <key>`, then a
  user message `Write a coherent response.`, then the assistant turn in which the text to be scored is placed;
- *neutral context*: identical, but the background line reads `No additional background is specified.`

**Per-token contrast.** For a text `y = (y₁ … yₙ)` and position `t`, let `p_key(yₜ)` be the probability the model
assigns to token `yₜ` given the keyed context and the preceding tokens `y₁ … yₜ₋₁`, and `p_neutral(yₜ)` the same under
the neutral context. The contrast is

    d_t = log p_key(y_t) − log p_neutral(y_t)

It is positive when the keyed context finds that token more plausible than the neutral context does. On a text
written without the key, `d_t` is noise around a slightly negative mean; the watermark makes it positive on average.

**Detection statistic.** The mean contrast divided by its standard error within the text:

    T(y) = mean(d_t) / ( sd(d_t) / √n )

`T` grows with the square root of the text length for a fixed per-token effect, which is why long texts detect more
reliably than short ones. On clean 512-token texts `T` averages about −1.7 with a standard deviation of about 1.2.

---

## 3. Embedding: how the text is generated

Generation is ordinary autoregressive sampling with one extra factor at each step.

### 3.1 Three contexts advance together

Three independent model states are kept and extended by one token per step:

1. the *generator*: the real conversation (system prompt + the user's task + the assistant's reply so far);
2. the keyed context with the reply so far appended;
3. the neutral context with the reply so far appended.

At every step the generator gives a next-token distribution `p_G(v)` over the vocabulary, and contexts 2 and 3 give
`p_key(v)` and `p_neutral(v)`. Each step therefore costs three forward passes (five with the decoy-relative variant of
§3.5). Because the reference contexts contain only a fixed wrapper plus the reply, exactly the same quantities can be
recomputed later by anyone who has the text and the key.

### 3.2 The push

Standard sampling first applies the usual temperature (0.8) and nucleus filter (top-p 0.95) to `p_G`, leaving a set of
candidate tokens with probabilities `p_G(v)`. The watermark reweights them:

    p'(v) ∝ p_G(v) · exp( β_t · clip( log p_key(v) − log p_neutral(v), −c, +c ) )

and samples the next token from `p'`. `c` is a clip on the contrast (0.5 nats in all final settings); it exists
because a few tokens, notably the key's own words, have contrasts of several nats, and without the clip the sampler
would simply insert them. `β_t` is the strength at step `t` (§3.3). With `β_t = 0` the procedure is ordinary sampling.

Three tokens are handled specially before the reweighting (§3.4): the stop token, the key's own words, and tokens that
would repeat an n-gram already in the text.

### 3.3 Strength control

`β_t` is the product of a nominal strength `β` (10 or 15 in the final settings) and two multipliers:

- **Uncertainty gate.** Let `H_t` be the entropy of the filtered generator distribution at step `t`. The multiplier is 0
  when `H_t ≤ H_min`, rises linearly to 1 at `H_t = H_ref`, and stays 1 above. Confident positions (function words,
  obvious continuations) are left alone; uncertain positions carry the mark. Prose settings: `H_min = 0.2`,
  `H_ref = 2.0` nats. Code settings: `H_min = 0.05`, `H_ref = 1.0`, because code is confident almost everywhere.
- **Adaptive controller.** The sampler tracks the running detection statistic of the tokens it has already emitted,
  `T_t = mean(d₁…dₜ) / (sd / √t)` (using the true, unclipped contrasts of the chosen tokens). Two schedules were used:
  a *linear ramp*, multiplier `clip((T* − T_t)/T*, 0, 1)`, which reduces the push as the text approaches a target
  `T* = 4`; and an *on/off* schedule, full strength while `T_t < T*` and zero afterwards. Each text is pushed only as
  hard as its own detection requires. In practice the ramp settles below its target (the mean final `T_t` was about
  2.4 for `T* = 4`), and the two schedules give results within noise of each other.

### 3.4 Guards

- **Stop token.** The probability of ending the reply is pinned to the generator's own value: after reweighting, the
  stop token's probability is reset to `p_G(stop)` and the remaining mass is renormalised. Merely zeroing its contrast
  is not enough, because the boosted tokens would take probability mass away from stopping and outputs get longer.
  With the pin, output length is unchanged.
- **Key words.** The first token of every surface form of each key word (lower case, capitalised, plural, with and
  without a leading space) gets zero contrast. The key can steer *toward its world* but can never write itself down.
- **Repetition.** If the last three emitted tokens have occurred earlier in the text followed by token `v`, then `v`
  gets zero contrast: the mark may not be bought by looping. (Repetition in the outputs stayed within the range that
  clean generation itself shows, with or without this guard; it mainly keeps key-word mentions at the clean rate.)

### 3.5 Decoy-relative push (attribution setting)

The neutral context is a single sentence, so `log p_key − log p_neutral` contains a component shared by *every*
background phrase (the model behaves differently as soon as any background line is present). A text pushed with that
component looks "watermarked by something" under other keys too. The attribution setting replaces the push contrast by

    log p_key(v) − mean over 3 decoy phrases of log p_decoy(v)

where the decoy phrases are fixed random-word phrases of the same construction as keys, chosen from a frozen pool of
12 so that none shares a word with the key. Two extra contexts are advanced per step (five forward passes). The
detection statistic still uses the neutral context; the adaptive controller tracks that same neutral-based statistic.

### 3.6 Cross-model steering (the world may be a different model)

Nothing in the construction requires the keyed and neutral distributions to come from the generating model. A second,
independent checkpoint can play the world, which changes what the secret is: not a phrase alone, but the pair
(phrase, reference model). Detection then needs that reference model and not the generator.

*Detection needs no vocabulary bridge.* The detector only teacher-forces the finished text under the reference
model's two contexts, in the reference model's own tokenization. It is exact, and the same code runs unchanged.

*Generation needs a bridge, and may be approximate.* Three problems and their solutions:

1. **Different token inventories.** A table computed once per model pair maps each generator token to the *first*
   reference token of its surface string; the contrast for a generator candidate is read at its mapped entry.
   Unmappable tokens (specials, odd fragments) simply get no push.
2. **Tokenization drift.** Byte-pair tokenization is not incremental: appending one generator token can change the
   last reference token. After each committed token the response is re-tokenized, the reference cache is cropped to
   the longest common reference-token prefix, and only the tail is recomputed — one or two tokens per step.
3. **Approximation is safe.** A wrong mapping wastes push; it cannot corrupt detection, because the detector
   re-derives everything from the finished text.

**The reference model must respond to the background line.** With a 410M *base* model as the world, the contrast on
clean text is dominated by a generic "there is some background line" term: its clean distribution sits at +0.57 with a
1% tail at 4.02, above the watermarked mean of 2.19, so detection at a strict threshold is 2% despite an AUROC of
0.86. A 360M *instruction-tuned* model from another family removes that offset (clean mean −0.49, 1% tail 2.31) and
detection rises to 80–90% (§8.7). A base model is trained to continue text, not to condition on an instruction, so
the key barely moves it.

Contexts use the reference model's own chat template when it has one, and plain text (`{system}\n\n{user}\n\n`)
otherwise. Cost is three forward passes per token, two of them on the smaller reference model, so a cross-model run
is *cheaper* than the same-model one.

### 3.7 The two final settings

| | presence setting | attribution setting |
|---|---|---|
| push | key vs neutral | key vs 3 decoys |
| nominal strength β | 10 | 15 |
| clip c | 0.5 | 0.5 |
| gate H_min → H_ref | 0.2 → 2.0 nats | 0.2 → 2.0 nats |
| controller | linear ramp to T* = 4 | on/off at T* = 4 |
| guards | stop pin, key words | stop pin, key words, repetition |
| cost per generated token | 3 forward passes | 5 forward passes |

(The cross-model variant of §3.6 can be combined with either setting.)

---

## 4. Detection

Input: the text and one candidate key. Not available and not needed: the task prompt, the generation seed, which
positions were pushed, or any classifier.

1. Normalise the text (line endings to `\n`, otherwise verbatim) and tokenise it with the model's tokenizer.
2. Teacher-force it under the keyed context and under the neutral context; take per-token log-probabilities of the
   visible tokens only (wrapper tokens and the final stop token are excluded).
3. Compute the contrasts `d_t` and the statistic `T(y)`. Texts shorter than 32 tokens are refused ("abstain") rather
   than scored.
4. **Presence decision.** Compare `T(y)` with a threshold chosen on clean texts (§5). Report the rank of `T(y)` among
   the clean scores as the p-value; flag when it is ≤ 1%.
5. **Attribution decision (which key).** Compute the whole-text mean contrast `F(y) = mean(d_t)` for the candidate key
   and for 100 fixed *decoy* keys of the same construction (random-word phrases drawn from the same vocabulary and
   lengths, never used as real keys). The candidate is attributed when its `F` exceeds all 100 decoys. Because a wrong
   key is statistically exchangeable with a decoy, this test attributes a wrong key about 1 time in 101 by
   construction, and the held-out data confirm 1–2.5%.

Both decisions are reproducible from a copied text file with a standalone command that reads only the key manifest
and the calibration file.

---

## 5. Calibration

The presence threshold must come from texts that were never watermarked and never used for tuning.

- **Clean cohort.** 1,200 replies generated with no key (400 per text domain), each scored under all 16 held-out keys:
  9,616 clean values of `T`. The 1% threshold is the value exceeded by the top 1% of them (a conservative rank rule:
  flag when fewer than 1% + one of the clean values are at least as large).
- **Human cohort.** The 1,200 human or web texts that the same prompts came with, scored the same way, as a second
  null for the spec's combined detector (a text must be unusual with respect to both cohorts).
- The threshold is fixed before any watermarked text is scored and is never changed for attacks.

---

## 6. Keys and decoys

- **Vocabulary.** 120 ordinary, unrelated English nouns (harbor, violet, ladder, meadow, …). A subset of 61
  "low-incidence" words, each occurring in at most 1 of 1,000 reference texts, exists for experiments where chance
  mentions must be negligible; detection was equivalent with either vocabulary.
- **Keys.** Drawn by a seeded random generator, 4 or 10 words, no repeated phrase. Development used 12 keys;
  the held-out run used 16 fresh keys drawn with a different seed. Keys are assigned to prompts by a fixed balanced
  schedule so that every compared arm sees the same prompt–key pairs.
- **Detection decoys.** 100 phrases drawn from the same vocabulary and lengths, excluding the exact key phrases.
- **Push decoys.** 12 phrases from the same vocabulary; per key, the first 3 that share no word with it.

These keys are experimental conditions, not cryptographic secrets: with 120 words the space of 4-word keys is about
2×10⁸.

---

## 7. Evaluation protocol

**Data.** Three prose domains, equally weighted: creative writing prompts, instruction-following questions, and news
article continuations; plus a code domain of 400 programming tasks that come with hidden test assertions. Prompts are
grouped by source, deduplicated (near-duplicate documents share a group), and partitioned so that tuning, calibration
and held-out prompts never share a group.

**Arms.** For each prompt: a clean reply, a reply with the presence setting, a reply with the attribution setting, all
with the same sampling seed stream. 512-token budget for prose (minimum 64), no minimum for code.

**Detection metrics.** Fraction of watermarked texts above the fixed 1% clean threshold ("detected at 1% FPR"), the
same at 5%, area under the ROC curve against clean texts, and *false attribution*: the fraction of watermarked texts
that the presence test flags when scored under a wrong key.

**Quality metrics.** (a) Change in the model's own log-probability of the reply given the real task, paired per prompt
against the clean reply (a proxy for "would the unwatermarked model have written this"; a second clean draw differs
from the first by 0.004 per token, so differences of 0.1 are real, differences below that are noise on 60 prompts).
(b) Fraction of replies containing any 8-word sequence twice (loops). (c) Fraction of replies mentioning a key word.
(d) Reply length. (e) For code: the fraction of generated programs that pass the task's test assertions when executed
in a sandbox, paired against the clean program for the same task.

**Robustness.** Registered transformations applied to both watermarked and clean replies, then scored with the
unchanged detector and threshold: light and strong paraphrase by the same model, sentence resegmentation, paragraph
reordering, keeping the first/middle/last 50% or 75% of the text, and dilution with 25–100% unrelated clean text.

---

## 8. Results

### 8.1 Held-out prose (600 prompts, 16 fresh keys, 512-token budget)

| | clean control | presence setting | attribution setting |
|---|---|---|---|
| detected at 1% false positives | 2% | **91%** | **77%** |
| detected at 5% false positives | 5% | 97% | 85% |
| AUROC against clean text | 0.50 | 0.995 | 0.964 |
| flagged under a wrong key at the 1% threshold | 1% | 38% | 13% |

The development estimates on 60 prompts and 12 keys were 95% and 77%; they held on data never seen during tuning.

**Attribution (which key), same held-out texts, 100 same-construction decoy phrases:**

| | clean control | presence setting | attribution setting |
|---|---|---|---|
| correct key beats all 100 decoys | 1.0% | 55% | **82%** |
| a wrong key beats all 100 decoys (chance 0.99%) | 0.94% | 0.96% | 0.98% |
| continuous version, detected at 1% | 1% | 60% | **85%** |
| correct key ranked first among the 16 keys | – | 75% | 89% |

Wrong keys sit exactly at chance over 9,000 text–key pairs, so an attribution is wrong about 1 time in 100 regardless
of how strongly the text was pushed.

**The originally specified detector** (whole-text mean contrast, rank against *both* the clean-generator cohort and the
human-text cohort, frozen from 1,200 + 1,200 calibration texts) flags 0.8% of clean texts, 54% with the presence
setting and 64% with the attribution setting. It is weaker than the within-text statistic `T` because it does not
normalise each text's own noise. By domain it flags 62% / 92% of story texts, 70% / 70% of news continuations and
30% / 31% of instruction answers; clean false positives are 0% for stories and answers and 2.5% for news continuations.

**By domain, within-text statistic `T` at the pooled 1% threshold and the 100-decoy attribution test** (200 prompts
each; presence setting / attribution setting):

| domain | mean reply length (tokens) | clean flagged | detected | correct key beats all decoys |
|---|---|---|---|---|
| stories | 480–500 | 1.5% | 99.5% / 95% | 75% / 94% |
| instruction answers | 430–455 | 1.0% | 86% / 58% | 49% / 78% |
| news continuations | 180–270 | 3.5% | 88% / 78% | 41% / 75% |

Open-ended stories carry the mark best; instruction answers are as long as stories but more constrained (facts,
lists), so there is less freedom to write into; news continuations are the shortest replies.

**Quality on the held-out prose:** log-probability cost 0.18 (presence) and 0.27 (attribution) nats per token; loops
11% and 8.5% against 8.3% clean; replies mentioning a key word 9.5% and 8.7% against 2.0% clean (these keys use the
ordinary vocabulary, so part of this is chance); length +11% for the presence setting, unchanged for the attribution
setting.

### 8.2 Held-out code (400 tasks with executable tests)

| | clean | presence setting | attribution setting | presence setting, opened gate (100 tasks) |
|---|---|---|---|---|
| detected at 1% false positives | 1% | 19% | 12% | **39%** |
| detected at 5% | 5% | 40% | 27% | 66% |
| AUROC | 0.51 | 0.85 | 0.74 | 0.94 |
| test pass rate | 24.5% | 21.5% | 23.0% | 22% (clean 24%) |

Code is confident almost everywhere: with the prose gate, only about a quarter of code tokens received any push and
the effective strength was near 1 instead of 10, so the mark was barely applied and the pass rate did not move.
Opening the gate (0.05 → 1.0 nats) doubled detection at no measurable pass-rate cost. Opening it fully (0 → 0.3 nats)
reached 45% at 1% but cut the pass rate from 23% to 15% and raised wrong-key flags to 34%, so the first opened setting
is the usable one. Attribution on code is near zero at every setting (4–6% beat all decoys). Reference solutions pass
their own tests 99.8% of the time, which validates the harness.

### 8.3 Quality (development scale unless noted)

- Log-probability cost: about 0.2 nats per token for β=10 settings, 0.25 for the attribution setting; 0.05–0.15 for
  β=5 settings.
- Loops: within the 3–12% range that clean draws themselves span.
- Key-word mentions: 2–5% with the guards (clean 2–3%); 12–20% without them.
- Length: unchanged with the stop-token pin.
- Code: pass rate within noise of clean at every setting tested.

### 8.4 Robustness

*Held-out* (102 prompts of the 600, strong same-model paraphrase rewriting 41% of the words, unchanged 1% threshold):
presence setting 91% → 47% still detected; attribution setting 77% → 54%; paraphrased clean texts flagged 4%. The
decoy-relative push loses less under rewriting than the neutral push.

*Development scale* (attribution setting at β=10):

At the unchanged clean-text threshold, with attacked clean texts flagged 0–3%: light paraphrase (7% of words changed)
72% still detected; strong paraphrase (39% of words changed) 50%; resegmentation 72%; reordering 57%; dilution with
100% appended clean text 70%; keeping the first half 63%; keeping the last half 33%. The mark is front-loaded because
the controller pushes hardest early; a maintenance push after the target did not fix that.

### 8.5 What was tried and dropped

Selecting among several candidate continuations by their whole-text score (the original proposal: real but weak,
5–15% detection); steering the model's internal activations (destructive before detectable); budgeting the push by KL
or expected likelihood loss (weak signal); very small clips with high strength (no cheaper); pushing only at highly
uncertain positions (weak); the maintenance push; persona-sentence keys (topic drift, worse attribution than random
words).

---

## 8.6 Conclusion

Yes, it works on prose. Text generated with the tilt reads like the model's ordinary output, and a detector holding
only the text and the key recognises 77–91% of 512-token texts while flagging 1–2% of clean ones, on prompts and keys
that played no part in tuning. With the decoy-relative push, 82% of texts are also tied to the correct key, and a
wrong key is accepted about once in a hundred. About half of the detections survive a strong rewrite of the text. The
price is a small, measurable loss of likelihood and some leakage of key words. It does not yet work well on short or
highly constrained text: code is detected 19–39% of the time and cannot be attributed. The world may also be a
*different, smaller* model (§8.5): detection then requires that model as well as the phrase, and the generator's own
model can no longer see the mark.

---

## 8.7 Cross-model steering (§3.6): results

60 prompts, seed-phrase keys, 512-token budget, clip 0.5, leak + repetition guards, stop-token pin, adaptive ramp.
"Detected" is against each detector's own pooled clean null (60 clean texts x 12 keys).

| reference model | nominal β | effective β | clean null (mean, sd → 1% threshold) | statistic on watermarked text | detected at 1% | cost (Δ log-lik/token) | loops (clean) | seen by the generator (AUROC) |
|---|---|---|---|---|---|---|---|---|
| base LM, 410M (`pythia-410m`) | 10 | 1.6 | +0.57, 1.28 → 4.02 | 2.19 | **2%** | −0.02 | 12% (8%) | 0.55 |
| instruction-tuned, 360M (`SmolLM2-360M-Instruct`) | 30 | 3.2 | −0.49, 1.23 → 2.31 | 2.88 | **80%** | −0.19 | 17% (5%) | 0.58 |
| instruction-tuned, 360M | 60 | 3.4 | −0.49, 1.23 → 2.31 | 3.12 | **90%** | −0.24 | 20% (5%) | 0.60 |
| instruction-tuned, 1B (`Llama-3.2-1B-Instruct`) | 30 | 3.4 | −0.18, 1.21 → 2.60 | 3.21 | **87%** | −0.25 | 10% (7%) | 0.67 |
| instruction-tuned, 1B | 60 | 3.6 | −0.18, 1.21 → 2.60 | 3.58 | **97%** | −0.28 | 15% (7%) | 0.67 |
| (same model as generator, for comparison) | 10 | 2.4 | −1.56, 1.15 → 0.97 | 2.31 | 95% | −0.23 | 10% (8%) | – |

Three findings:

1. **The reference model must respond to the background line.** A base language model was never trained to treat
   `Background: <phrase>` as an instruction: its keyed-minus-neutral contrast on clean text is a large, per-text
   variable offset (clean null centred at +0.57 with a 1% tail at 4.02, above the watermarked mean), so the
   key-specific part is buried. An instruction-tuned reference of *smaller* size moves the clean null to −0.49 with a
   1% tail at 2.31 and detects 80–90%.
2. **Cross-model matches or beats same-model.** A 1B instruction-tuned reference reaches 97% at a cost of −0.28
   (same-model: 95% at −0.23), and 87% at −0.25; a 360M one reaches 80–90% more cheaply. Reference quality, not
   reference size, is the first-order variable, but within instruction-tuned models the larger one is better on every
   axis except key-word leakage (6.7% at β=30 and 11.7% at β=60, against 0–3.3% for the 360M model).
3. **The mark is nearly invisible to the generator's own model**: scored with the generating checkpoint, cross-steered
   texts give AUROC 0.58–0.67 against clean text (the stronger the reference, the slightly more visible). The secret is the pair (phrase, reference model), which is a
   stronger secrecy property than the same-model setting, where anyone holding the generator and any background
   phrase can tell a text was steered.

Costs unchanged from §3.6: three forward passes per token, two of them on the smaller reference model. The vocabulary
bridge attenuates the nominal strength (effective β about a tenth of nominal), so cross-model settings need a
proportionally larger β. Open: wrong-key flags are 26–29% (the same non-specific component the decoy-relative push
fixes in the same-model case, implemented for cross-model but not yet measured), and loops rise from 5% to 17–20%,
the one quality measure clearly worse than the same-model setting.

---

## 9. Limitations, plainly

- The detector needs the generating model. Detection costs two forward passes over the text (plus 100 more for
  attribution).
- Presence is semi-public with the presence setting: anyone with the model and *any* background phrase can tell that
  the text was steered by something (that is the 38% wrong-key figure). The key protects attribution, not existence.
  The attribution setting reduces this to 13%.
- Short texts detect less: about 45–65% at 128 tokens; code 12–39%.
- Paraphrase tests use the same model as the generator, which is the optimistic case; no human has judged meaning
  preservation or output quality.
- Everything was measured on one 0.8-billion-parameter model and English text.

---

## 10. All fixed parameters

| parameter | value |
|---|---|
| model | Qwen3.5-0.8B, text-only path, thinking disabled, float32, TF32 off |
| sampling | temperature 0.8, top-p 0.95, top-k off |
| output budget | 512 new tokens; minimum 64 for prose, 0 for code |
| keyed context | system `You are a helpful assistant.\nBackground: <key>`; user `Write a coherent response.` |
| neutral context | same with `Background: No additional background is specified.` |
| clip c | 0.5 nats |
| strength β | 10 (presence), 15 (attribution) |
| gate | H_min 0.2, H_ref 2.0 nats (prose); 0.05 / 1.0 (code, opened) |
| controller target T* | 4 |
| repetition guard | 4-grams |
| key vocabulary | 120 words; keys of 4 or 10 words |
| decoys | 100 for detection, 12 (3 per key) for the push |
| detector minimum length | 32 tokens; maximum 4,096 tokens |
| clean calibration | 1,200 generated + 1,200 human texts × 16 keys; 1% conservative rank threshold |
| held-out set | 600 prompts (200 per domain), 16 keys, seed 0; code 400 tasks |

## 11. Reproduction

```bash
python -m wmworld.cli prepare-data --config configs/pilot.yaml                     # data, dedup, partitions
python -m wmworld.cli make-keys --config configs/pilot.yaml --set confirmation      # 16 fresh keys
python -m wmworld.cli make-keys --config configs/pilot.yaml --set decoy_rw          # 100 detection decoys
python -m wmworld.cli generate --config configs/pilot.yaml --stage conf_null --run-name null_calibration
python -m wmworld.cli calibrate --config configs/pilot.yaml                         # frozen thresholds
python -m wmworld.cli generate --config configs/pilot.yaml --stage conf_pos         # 600 prompts x 3 arms
python -m wmworld.cli evaluate --config configs/pilot.yaml --run runs/conf_pos      # tables, figures, side-by-sides
python -m wmworld.cli generate --config configs/pilot.yaml --stage code             # 400 code tasks x 3 arms
python -m wmworld.cli code-eval --config configs/pilot.yaml --run runs/code         # execute against the tests
python -m wmworld.cli detect --text-file sample.txt --key-id confirmation_003 --calibration assets/calibration.json
```

Every run directory keeps the generated texts, the private per-token trajectory (entropy, strength, contrast at each
step), the detection records (text hash, key, score; no prompt or label), and the evaluation tables.
