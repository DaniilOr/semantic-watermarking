# Watermarking LLM text with a secret phrase: approach, results so far, open problems

*Living document. Last updated 20 September 2026, 19:00. The held-out round, the key-length experiment and the
controls of the pairwise measure are complete.*

---

## 0. Summary

We hide a watermark in generated text by nudging the generator toward words that a small **reference language
model** finds slightly more likely after it has been "told" a **secret phrase** (four random words such as
`pebble umber echo meadow`). Anyone who knows the phrase can replay the finished text through the reference model
and test whether the text leans toward the phrase more than ordinary text does. The detector needs **only the text
and the secret phrase**: no prompt, no access to the generator, no logs from generation.

Where it stands (held-out numbers: 150 unseen prompts, 16 fresh keys, thresholds frozen beforehand):

* **Detectable, confirmed on unseen data.** The selected gentle setting (constant push of strength 4) is detected in
  **85%** of texts (95% interval 78–90%) with 0.25% of clean text–key pairs flagged; at an actual 1% false-positive rate
  detection is 89%. The detector is a different, smaller model than the generator and needs only the text and the phrase.
* **No cost on verifiable instruction following** (IFEval, official checker: 54.0% vs 53.5% clean), and the mark is
  still detected in 72% of those short answers.
* **Gentler, not harmless.** A blind judge prefers the clean answer about 2 : 1 on prose (preference 0.31, where an
  independent clean run scores 0.47; 0.44 on instruction answers). In a separate run that lists flaws per text, major flaws rise from 79% to 89% and unrelated
  insertions from 7% to 22% — real (4–14× the clean-to-clean noise) but moderate: about half the damage of our earlier
  settings. Reading random pairs side by side, the difference is not obvious.
* **Erased by a strong paraphraser.** After a rewrite by a 31B model, detection falls from 85% to 12% (light) and 4%
  (strong). This is the main open weakness.
* **What made a gentle push detectable was the detector**: capping how much any single token can count raises
  detection of the very same texts from 53% to 85%.
* **Key length:** 8–10 words is the sweet spot for presence detection (72% with 4 words → 85–90%, saturating at 16),
  but longer keys are much less specific (flagged under a wrong key 19% → 64%). Halving the strength (β = 2) removes
  the measurable quality cost but leaves only 20–42% detection.
* The development sweep (60 prompts, best of 14 settings) had made the selected setting look indistinguishable from
  clean text; that did not replicate. The held-out numbers are the ones to quote.

One line: *a gentle push is reliably detectable by a second small model and barely visible to a reader, but it is
measurably not free, and a capable paraphraser removes it.*

---

## 1. The idea in plain words

A language model writes one token at a time by sampling from a probability distribution over the next token. At
many positions several continuations are about equally good.

We take a second, small language model (the **reference model**) and let it read the text-so-far **twice**:

1. once after a system line containing the secret phrase (`Background: pebble umber echo meadow`), and
2. once after a neutral line (`No additional background is specified.`).

For every candidate next token the two readings give two probabilities. Their log-ratio

```
d(token) = log p_reference(token | secret phrase, text so far) − log p_reference(token | neutral line, text so far)
```

says how much the secret phrase makes that token more (or less) likely. For most tokens `d` is about zero.

**Embedding.** We tilt the generator's distribution slightly toward tokens with positive `d`:

```
p'(token) ∝ p_generator(token) · exp( β · clip(d(token), ±c) )
```

**Detection.** Given a finished text and a candidate secret phrase, run the reference model over the text twice
(teacher forcing), collect `d_t` for every token actually present, and ask whether these values are, on average,
more positive than in ordinary text:

```
T = mean(d_t) / ( sd(d_t) / √n )
```

`T` is compared with a threshold taken from clean (unwatermarked) texts scored under the same phrase. A text is
flagged if fewer than 1% of clean texts score that high.

```
 EMBEDDING
   task prompt ─► GENERATOR ──────────────► p_generator(next token) ─┐
                                                                     ├─► p' ∝ p_gen · exp(β·clip(d)) ─► sample
   text so far ─► REFERENCE MODEL, read twice:                       │
                    with the secret phrase  → log p_key     ┐        │
                    with a neutral line     → log p_neutral ┴─ d ────┘

 DETECTION   (no prompt, no generator)
   finished text + candidate phrase ─► REFERENCE MODEL twice ─► d_t per token ─► T ─► above clean-text threshold?
```

### Why this design is interesting

* **Text + key only.** The detector never sees the prompt or the generator. This is what a third-party verifier
  would actually have.
* **The key is a phrase.** The key space is as large as language; keys are human-readable; different keys mark
  different customers/sessions, and we can ask *which* key marked a text (attribution), not only *whether*.
* **The mark is defined by a model, not a hash of previous tokens.** Classic "green-list" watermarks derive a
  pseudo-random token list from the preceding tokens, so rewording destroys them token by token. Here the
  preference comes from a language model's reading of the whole context, which gives some resilience to rewording
  (measured below) — though less than we want.
* **Any generator can be marked.** The reference model is small and fixed; only a vocabulary bridge is needed
  during generation (below). Detection needs no bridge.

---

## 2. Details that matter

**Guards during embedding.**

| guard | what it does | why |
|---|---|---|
| entropy gate | no push where the generator is confident (entropy < 0.2 nats), full push above a reference entropy | confident tokens are usually the only correct ones |
| clip `±c` (0.5) | caps how much any one token's contrast can count in the push | limits per-token distortion |
| leak guard | zero push on the secret phrase's own words (all surface forms) | the text must not recite the key |
| repetition guard | zero push on a token that would complete an n-gram already in the text (n=4) | pushes otherwise feed loops |
| stop-token pin | the probability of ending the text is kept exactly at the generator's value | otherwise texts get longer |

**Strength control (until today).** A controller watched the running value of `T` on the tokens already written and
scaled β down as `T` approached a target, so that "each text is pushed only as hard as its detection needs".
Section 4.6–4.7 explain why it was dropped in favour of a constant push.

**Cross-model bridge.** Generator and reference model have different vocabularies. For each generator token we
look up the first reference-model token of its surface string (a frozen table; 248,068 of 248,077 generator tokens
map) and read `d` there. The mapping is approximate, which can only *weaken* the push; it cannot make detection
wrong, because detection recomputes everything from the text in the reference model's own tokenisation. We verified
that the generator-side running estimate of `T` tracks the detector's value within 7–9% across strengths and lengths.

**Keys.** Four (or ten) random, unrelated, common words. Earlier "persona" keys (`You eat chicken every day.`)
caused obvious topic drift and attributed worse. 12 development keys, 16 held-out confirmation keys.

**Attribution** ("which key?") uses 100 decoy phrases of the same construction: the claimed key must beat all of
them. A wrong key is exchangeable with a decoy, so false attribution sits at chance by construction — and it does.

---

## 3. Experimental setup

| | |
|---|---|
| generator | `Qwen/Qwen3.5-0.8B`, temperature 0.8, top-p 0.95, up to 512 new tokens (1,024 in long-text runs) |
| reference model | same checkpoint (first phase); `meta-llama/Llama-3.2-1B-Instruct` (cross-model phase) |
| tasks | stories (WritingPrompts), instruction answers (Dolly), news continuation (C4 realnewslike); equal shares |
| development set | 60 prompts, 12 keys |
| held-out sets | 600 unseen prompts, 16 fresh keys; 1,200 clean texts for calibration, separate clean texts for false positives |
| quality judge | blind pairwise comparison by `google/gemma-4-31b-it` through an API, cross-checked by `qwen/qwen3.8-27b` |

The generator is deliberately tiny (0.8B): experiments are cheap, but its clean texts are themselves often flawed,
so **all quality numbers are paired comparisons against the clean answer to the same prompt**.

---

## 4. Results in the order they were obtained

### 4.1 The original proposal: pick among candidate continuations — weak

Every 16 tokens, sample 8 candidate continuations and prefer the one that raises the whole-text score. Real but
weak: AUROC 0.89, only 5–15% of texts detectable at 1% false positives. Unsteered candidates barely differ in `d`.

### 4.2 Token-level push, reference = the generator itself — detection works (held-out)

600 unseen prompts, 16 fresh keys, thresholds frozen beforehand from 1,200 clean texts:

| | clean | "presence" setting | "attribution" setting |
|---|---|---|---|
| detected at 1% false positives | 2% | **91%** | **77%** |
| AUROC vs clean | 0.50 | 0.995 | 0.964 |
| correct key beats all 100 decoys | 1.0% | 55% | **82%** |
| a *wrong* key beats all 100 decoys (chance 0.99%) | 0.94% | 0.96% | 0.98% |

By domain (presence / attribution setting): stories 99.5% / 95% detected, instruction answers 86% / 58%, news
continuations 88% / 78%. Open-ended text carries the mark best.

Robustness (strong paraphrase rewriting ~41% of the words, unchanged threshold): 91% → 47% and 77% → 54% still
detected; paraphrased clean texts flagged 4%. The paraphraser here was the same 0.8B model, i.e. a weak attacker.

> **Caveat added on 19 September.** Quality in this phase was measured only with proxies computed by the same small
> model (log-probability cost ≈ 0.2 nats/token, loop rate, key-word mentions). Section 4.5 shows such proxies miss
> semantic damage. These configurations have **not** yet been re-examined by the independent judge.

### 4.3 A different model as the reference — works, if it is instruction-tuned

Generator stays Qwen; the mark is defined by another model.

| reference model | AUROC vs clean | comment |
|---|---|---|
| Pythia-410M (base model) | 0.86 | a base model barely reacts to a `Background:` line; the key-specific signal drowns in a generic offset |
| SmolLM2-360M-Instruct | 0.99 | works |
| **Llama-3.2-1B-Instruct** | **0.994–0.996** | best; used from here on |

If someone scores the Llama-marked texts with the *generator's own* model instead of the right reference model,
separation drops to AUROC 0.58–0.67: the secret is effectively "phrase + which reference model".

### 4.4 An audit of our own numbers (19 September)

* **Detection rates in the cross-model phase were unreliable.** The threshold had been estimated from each run's own
  60 clean texts (× 12 keys = 720 *values*, but only 60 *independent texts*; the 1% tail rested on ~6 observations).
  Two runs of the same configuration gave thresholds of 2.60 and 3.85 — which turned the same signal (`T` ≈ 3.2) into
  "87% detected" and "5% detected". AUROC (0.98–0.997) and mean `T` were stable throughout.
  **Fix:** one frozen null of 1,200 clean texts, per-key thresholds, Wilson intervals (results in Section 4.8a).
* **The controller is a faithful instrument**: its internal estimate of `T` is 0.91–0.93 of the detector's value in
  every arm we checked.
* **Length is a confound.** Watermarked texts are systematically longer than clean ones (medians 839–958 vs 650
  tokens in the long-text run) and `T` grows like √n, so honest comparisons must be length-matched.
* **Long texts exposed a controller flaw.** `T` plateaued at ≈ 3.2 regardless of length: a running mean over the whole
  text cannot be "un-marked", so once the opening is marked the controller stops pushing for good. Two repairs were
  built (reset the statistic every N tokens; or measure it over a sliding window). Both became moot after 4.6.

### 4.5 An independent judge changes the picture (19 September)

Blind pairwise comparison, both presentation orders, judge never sees key or arm. For every response the judge also
reports whether it addresses the task, which flaws it has, and whether both responses answer the same question.
60 prompts per row; win rate counts ties as half; 0.50 would mean "cannot tell which is better".

| configuration (internal label) | win rate vs clean | addresses the task (marked / clean) | major flaws (marked / clean) |
|---|---|---|---|
| controller **off**, full strength — known to be damaged (`wm_cross30f`) | **0.11** | 0.39 / 0.79 | 100% / 51% |
| controller with a raised target (`wm_cross30s`) | 0.21 | 0.57 / 0.81 | 92% / 58% |
| controller, doubled strength ceiling (`wm_cross60`) | 0.23 | 0.57 / 0.75 | 93% / 57% |
| **controller, standard — our "best" until today** (`wm_cross30`) | **0.26** (CI 0.18–0.35) | 0.60 / 0.74 | 88% / 62% |
| same, long texts where the controller stalls, i.e. least push (`wm_cross30`) | 0.37 | 0.70 / 0.77 | 82% / 71% |

* The judge passes its built-in validity check: the arm we *knew* was broken loses hardest, and the ranking follows
  push strength exactly.
* **Two unrelated judges agree.** On the identical 120 pairs: win rates 0.263 vs 0.275 and 0.229 vs 0.204; when both
  take a side they pick the same side 94% of the time; per-pair correlation 0.78.
* "Both responses answer the same question": 98–100%. The watermark corrupts answers; it does not replace them.
* What goes wrong, per 100 texts (standard controller, marked vs clean): **unrelated words or insertions 35 vs 9**,
  **topic drift 13 vs 2.5**, incoherence 73 vs 50, ignored instructions 15 vs 10; repetition is *not* worse (10 vs 15).
* Our earlier proxies (−0.25 nats/token, loops 10% vs 7%) could not see any of this.

**An example the judge flagged.** Task: *"What's the preferred mode of transportation in Vietnam?"*

> **Clean:** "In Vietnam, the **most preferred mode of transportation is the motorcycle** …"
>
> **Watermarked** (secret phrase `amber copper delta atlas`): "While there is no single 'preferred' mode … the
> **Hybrid of Road and Sea** … ### 1. The River (The Gold Standard) … the Mekong **Delta** …"

The key's *meaning* leaked into the *content* of the answer. The leak guard blocks only the literal key words.

### 4.6 Diagnosis

**The evidence was narrow, and made of the wrong tokens.** In texts marked with the old settings:

* **the top 5% of tokens carry 91% of a text's evidence**; 1.65% of tokens (those with `d` > 1.5) carry half of it;
* those tokens are (a) response openers ("The", "While", "Following") and (b) words from the key's semantic
  neighbourhood — *valley, haven, wash, air* for `pebble umber echo meadow`; *Sea, city* for
  `amber copper delta atlas`; *maze, dense* for `kelp dune mesa oyster`.

A statistic dominated by a handful of large values is noisy, and it rewards exactly the words that do not belong in
the answer. This explains the judge's "unrelated insertions" and the fragility under paraphrase (reword a dozen
tokens and the evidence is gone). **This diagnosis held up**, and it led to the capped-evidence detector.

**A second diagnosis did not hold up.** The per-token traces showed that the adaptive controller delivers the push in
bursts (top 10% of tokens = 74–84% of all distortion; the first 16 tokens of every answer get ≈ 3.5× the average push;
doubling the strength ceiling only sharpened the bursts). Since distortion grows with β² and signal at most linearly,
we expected a thin, even push to hurt less at equal signal. **The sweep says otherwise:** a constant push that carries
the same signal as the controller is judged no better (win rate 0.25 vs 0.24). Damage follows *total push strength*,
not burstiness.

### 4.7 What each fix did

| fix | side | outcome in the sweep (60 prompts per setting) |
|---|---|---|
| capped-evidence statistic | detector | **Decisive.** +6 points on texts from the old settings (91% → 97%); on the gentle setting 50% → 80%; no clean text flagged in the sweep. |
| frozen null of 1,200 clean texts, per-key thresholds | detector | Stable thresholds; false positives measured on separate clean texts. |
| constant push, no controller | generator | Not better at equal signal — but simpler, no blind start, and nothing that can stall on long texts. |
| 48-token warm-up | generator | No measurable gain. |
| semantic guard (no push on tokens with `d` > 1.0) | generator | Helps at equal strength (insertions 43% → 31% at strength 6; win rate 0.12 → 0.20 at 12) but does not move the limit. |

**Selection rules were committed before these results existed** (`docs/preregistration.md`, with three dated
amendments: clipped detector candidates and guard arms; pooled false-positive eligibility; four frontier arms after
an early look). The rule: most detectable setting among those with judge win rate ≥ 0.40 and ≤ 10 points more major
flaws than clean. Outcome, applied mechanically by `scripts/select_config.py`: **constant push 4**, the only one of 14
settings that passes.

### 4.8 Results of the current round

**(a) Detection with a stable threshold, and which text-only detector is best — done.** One frozen null (1,200 clean
texts, per-key 1% thresholds); clean false positives measured on each run's own clean arm, which shares nothing with
the null. Texts are the finished development runs, i.e. generated under the old bursty regime.

| detector | standard controller, 512 tok | doubled ceiling | standard, long texts | controller off | raised target | **mean** | clean flagged (pooled, 2,160 pairs) |
|---|---|---|---|---|---|---|---|
| plain `T` (used so far) | 0.85 [0.74–0.92] | 0.97 | 0.88 | 0.97 | 0.88 | **0.91** | 1.4% |
| evidence clipped at ±1 (`T_clip1`) | 0.95 | 0.98 | 0.93 | 0.98 | 0.98 | **0.97** | 1.6% |
| **clipped ±0.5, entropy-weighted (`T_clip05_gate`)** | 0.97 [0.89–0.99] | 0.95 | 0.97 | 1.00 | 0.98 | **0.97** | 1.5% |
| entropy-weighted matched filter (`matched_gate_hi`) | 0.87 | 0.93 | 0.88 | 1.00 | 0.97 | 0.93 | 1.0% |
| direction only (`T_sign`) | 0.63 | 0.43 | 0.33 | 0.83 | 0.58 | 0.56 | 0.7% |

* **Detection was never the problem.** With a stable threshold the standard controller is detected 85–88% of the time and
  AUROC is 0.99+; the earlier "87% vs 5%" was threshold noise.
* **Capping each token's evidence makes the detector better, not worse** (+6 points on average), even on texts whose
  evidence is concentrated in a few tokens: those few huge values mostly add variance. This is the detector the
  "broad and shallow" redesign needs.
* Clean false positives are **clustered**: in the long-text run the 22 flagged text–key pairs come from 4 of 60 clean
  texts (a text that looks "marked" does so under several keys at once). Per-run rates therefore swing between 0.1% and
  3%; pooled they sit at 1.0–1.6% for every candidate. The pre-registered eligibility test was amended accordingly —
  openly, after this table and before any sweep text existed (`docs/preregistration.md`, amendment 2). Under the original
  wording plain `T` would have stayed; both statistics are reported everywhere from now on.
* A marked text is also flagged under a *wrong* key 21–57% of the time by these presence statistics: they answer
  "marked by some phrase?", not "by which phrase?". Attribution uses the decoy-rank test (Section 4.2), where wrong
  keys sit at chance.
* Tokenisation note: the reference model needs only ~0.8 tokens per generator token on this prose, so "512 generator
  tokens" is ~410 reference tokens; length-matched views use reference-token prefixes of 128 and 256.

**(b) Rotating keys: each span's key can be recovered — done (old controller).** 60 texts of up to 1,024 tokens, a new
key every 128 tokens (A, B, C, D, A …). For each generated span we ask which of the four keys scores highest:

| statistic | spans whose own key scores highest (chance 25%) |
|---|---|
| plain `T` | **79%** (260 / 329) |
| entropy-weighted | 82% (264 / 324) |
| entropy-weighted, high gate | **84%** (267 / 318) |

Whole-text detection does not suffer: 98% of rotating-key texts and 98% of single-key texts are detected at the frozen
1% threshold (clipped statistic; 93% / 92% with plain `T`; clean texts flagged 0.1–0.3%), and the first 128 reference
tokens alone already give 98–100%. Resetting the controller at each span also removed the long-text plateau (mean
`T` 4.9 instead of 3.2). **But** this run still used the old bursty controller, and the judge is as unhappy as before:
win rate against clean 0.25 (one key) and 0.30 (rotating keys; intervals overlap). So rotation is *recoverable*; whether
it is *acceptable* waits for the rerun with the gentle delivery.

**(c) The sweep: three separate measures — done.** 60 development prompts × 14 settings, 512-token budget.

| setting | detected at 1% FPR (clipped detector) | same texts, old detector | pairwise win rate vs clean | unrelated insertions or drift (clean 15%, band 7–25%) |
|---|---|---|---|---|
| **constant 4 (selected)** | **80%** [68–88] | 50% | **0.49** [0.40–0.59] | **13% — inside** |
| constant 4 + guard | 73% | 40% | 0.36 | 13% — inside |
| constant 6 / 6 + guard | 93% / 95% | 70% / 52% | 0.27 / 0.29 | 43% / 31% — outside |
| constant 8 / 8 + guard / 8 + warm-up | 95% / 95% / 92% | 80% / 73% / 70% | 0.25 / 0.28 / 0.24 | 33% / 38% / 38% — outside |
| old adaptive controller | 97% | 90% | 0.24 | 32% — outside |
| constant 12 / 16 / 20 | 98% / 93% / 100% | 90% / 90% / 98% | 0.12 / 0.17 / 0.13 | 52% / 62% / 70% — outside |

Clean texts flagged in this run: 0 of 720 text–key pairs. Constant 4, length-matched: 80% on the first 128 reference
tokens, 83% on 256, 88% on 384. Between strengths 4 and 6 the damage appears abruptly, because the push multiplies a
token's probability by up to e^(β/2): 7× at 4, 20× at 6, 55× at 8.

**(d) Absolute quality, kept separate from the win rate — done.** The win rate is a *relative* preference between two
answers; a forced comparison amplifies small differences and says nothing about acceptability — and the 0.8B generator's
clean texts are often flawed themselves. So the clean generator was estimated **separately**: a pointwise judge rates
one anonymous text at a time; five independent clean draws of the same 60 prompts give the clean rate *and its
draw-to-draw variation*; each setting is tested against the 95% band of a single clean draw and by its paired
difference to the per-prompt clean expectation.

| metric | clean mean (5 draws) | 95% band of one clean draw | constant 4 | constant 6 | old controller | constant 20 |
|---|---|---|---|---|---|---|
| texts with a major flaw | 83% (77–88) | 73–92% | 85% inside | 95% outside | 85% inside | 100% outside |
| flaws per text | 1.67 | 1.48–1.87 | 1.65 inside | 2.13 outside | 1.98 outside | 2.65 outside |
| not fully on task | 44% (33–50) | 32–57% | 50% inside | 62% outside | 48% inside | 83% outside |
| unrelated insertions or drift | 15% (10–22) | 7–25% | **13% inside** | 43% outside | 32% outside | 70% outside |
| mean 1–5 score | 1.50 | 1.28–1.73 | 1.57 inside | 1.17 outside | 1.40 inside | 1.02 outside |

For constant 4 every paired difference to clean has an interval spanning zero (e.g. insertions −2 points [−11, +8]).
Note how much clean text varies by itself ("not fully on task": 33% to 50% between two clean draws of the same
prompts), and that the coarse metrics sit near their floor (clean score 1.5 of 5) — the informative metric is the
watermark-specific one.

**(e) Held-out round — done.** Selected configuration: constant push 4; detector: clipped, entropy-weighted. Run once.

*Quality is measured in two separate judge runs.* **Which is better:** two anonymous responses to the same prompt, both
orders, one question, ties allowed (win rate counts a tie as half). **Flaws:** one anonymous text per call — addresses
the task? which flaws? severity? 1–5 score — compared with the clean generator's own rates from independent clean draws
of the same prompts. (An earlier mixed prompt that listed flaws and picked a winner in one call gave the same win rates
within 0.05; all "which is better" numbers below are from the preference-only run.)

| check | constant 4 (selected) | old adaptive controller | clean |
|---|---|---|---|
| detected, frozen thresholds (clipped detector) | **85%** [78–90], AUROC 0.984 | 91% [86–95] | 0.25% of 2,400 pairs flagged |
| detected at an *actual* 1% false-positive rate | 89% | 97% | — |
| same texts, old plain detector | 53% | 89% | 0.50% |
| first 128 / 256 / 384 reference tokens | 76% / 86% / 88% | — | — |
| which is better, prose (win rate vs clean) | **0.31** [0.26–0.37], ties 28% | 0.26 [0.21–0.31] | 0.50 by definition |
| flaws: texts with a major flaw | 89% — outside the clean band | 90% | 79% (band 73–86%) |
| flaws per text | 1.79 — outside | 2.03 | 1.48 (1.36–1.61) |
| unrelated insertions or drift | **22%** — outside | 31% | 7% (3–12%) |
| mean 1–5 score | 1.35 — outside | 1.33 | 1.74 (1.56–1.91) |
| IFEval, all constraints satisfied (strict / loose) | **54.0% / 56.0%** | — | 53.5% / 56.5% |
| IFEval answers: detected / which is better | 72% [65–78] / 0.44 [0.38–0.49] | — | 0.88% flagged |
| after a **light** paraphrase by a 31B model | **12%** [8–18] | 19% | 1.1% flagged |
| after a **strong** paraphrase | **4%** [2–8] | 7% | 2.3% flagged |

* **The flaw test is calibrated.** Fed a clean draw as if it were watermarked, it flags 0 of 10 comparisons on the
  held-out prompts (5 of 25, two-sided, on the smaller development set); two clean draws differ by at most 1–4 points.
  The differences above are 4–14 times that noise: real, but moderate — about 15 more texts in a hundred with an
  unrelated insertion, roughly half of what the old controller does.
* **The damage is general, not a few bad keys:** it appears in all three text types (news 24% vs 12% clean, instruction
  answers 18% vs 10%, stories 24% vs 6%) and across most of the 16 keys.
* **A blind read of nine random pairs** (order shuffled, key hidden): 8 of 9 identified, but 5 at low confidence — both
  responses read like the same small model; one "confident" call was right for the wrong reason. The visible cost
  concentrates in news continuation, where the clean model stops early and the marked one keeps inventing specifics.
* **Why the sweep looked better:** the selected setting scored 0.48 there — the best of 14 settings on 60 prompts —
  and regressed to 0.31 on 150 unseen prompts under the same judge and prompt. Selection on a small set, not the judge.
* **Paraphrase:** the external model's "light" rewrite already replaces 57% of the content words. Earlier figures
  (about half of detections surviving) were against the 0.8B model paraphrasing itself.

**(f) Long texts with the gentle push — done.** Up to 1,024 tokens, 60 prompts: one key throughout is detected in 85%
[74–92] of texts (old detector 52%); with four keys rotating every 128 tokens the first key is still detected in 65% of
whole texts, and each span's own key is the best-scoring of the four in **53–57%** of spans (chance 25%; 79–84% with the
old, much stronger push). Rotation is no worse for quality than a single key (win rate 0.33 vs 0.30, mixed prompt).

**(g) Key length — done.** Six keys per length with no word shared between keys of a set, the same 60 development
prompts, a frozen 1,200-text null per key set, both judge runs.

| key length | β = 4: detected | flagged under a wrong key | preference | insertions / drift | β = 2: detected | preference | insertions / drift |
|---|---|---|---|---|---|---|---|
| 4 words | 72% [59–81] | 19% | 0.35 | 25% | 20% | 0.38 | 8% |
| 8 words | 85% [74–92] | 37% | 0.29 | 32% | 28% | 0.37 | 13% |
| 10 words | **90%** [80–95] | 48% | 0.35 | 27% | 30% | 0.38 | 12% |
| 16 words | 88% [78–94] | 64% | 0.30 | 32% | 42% | 0.40 | 17% |

Reference levels on these prompts: clean-vs-clean preference 0.42–0.45, clean insertions/drift 13–17%; clean texts
flagged ≤ 0.56% for every key set.

* **Longer keys mark more strongly at the same β**, saturating at 8–10 words, at no clear extra quality cost.
* **Specificity falls steadily, even with disjoint keys:** a longer phrase adds more generic "written under some
  background phrase" signal, which the presence statistic picks up under any key of that length. Long keys are for
  presence; attribution needs short keys or the decoy-rank test.
* **β = 2 has no measurable quality cost** (preference 0.37–0.40 against a clean level of 0.42–0.45; insertions at or
  below the clean rate) **but is detected only 20–42% of the time.** A longer key does not rescue a much gentler push:
  halving β roughly halves the signal at every length.

**(h) Controls of the pairwise measure — done.**

| comparison against the reference clean run | preference |
|---|---|
| held-out prompts (150): an independent clean run | **0.47** [0.41–0.53] — calibrated |
| development prompts (60): two further clean runs | 0.45 [0.37–0.55], 0.42 [0.33–0.51] |
| development prompts: β = 0 through the watermarking sampler | 0.43 [0.35–0.52]; flaws and length equal to clean |

* The watermarking code path adds nothing by itself: with zero push it is indistinguishable from the clean arm.
* On the development prompts the reference clean run is a slightly good draw, and the measure has intervals of about
  ±0.09: read development preferences against ~0.44, not 0.50, and ignore gaps below ~0.1. Against that level β = 2
  costs nothing measurable and β = 4 about 0.12.
* On the held-out prompts the measure is calibrated (0.47), so the held-out gap of β = 4 (0.31, about 0.16 below the
  clean level) is real.

---

## 5. Long texts and rotating keys

*Motivation:* a long story cannot be treated as one block; it is generated in spans.

* **What was built.** Generation is one continuous pass; the reference model's state is never restarted. A span
  boundary can (a) reset the controller's statistic so that every span is pushed again, and/or (b) switch to the next
  key of a rotation (span 0 → key A, 1 → B, 2 → C, 3 → D, 4 → A …). The detector can scan windows and, for rotation,
  ask for each span which of the rotation's keys scores highest.
* **What rotation buys:** a mark that is local (any surviving excerpt carries a complete mark) and a payload (the
  *order* of keys is information).
* **What it costs:** each key's whole-text evidence is divided by the number of keys, so whole-text detection of a
  single key gets weaker by design; detection has to work per span.
* **Status.** With the old, strong push, 79–84% of spans are attributed to their own key (chance 25%) without hurting
  whole-text detection — but that text is visibly damaged. With the selected gentle push the figure is 53–57%: a
  128-token span pushed this gently carries little evidence, so rotation needs longer spans or a stronger mark
  (Section 4.8f).

---

## 6. Things that did not work

Candidate selection by whole-text score (weak); steering internal activations (destructive before detectable);
budgeting the push by KL or expected likelihood loss (weak); pushing only at very uncertain positions (weak);
persona-sentence keys (topic drift, worse attribution); a base (non-instruction-tuned) reference model (no usable
signal); a √n-scheduled controller target (measured: no effect — the problem was the estimator's memory, not the
setpoint); a higher strength ceiling (same average push, sharper bursts, worse quality); and — the big one — a
high-gain adaptive controller as the delivery mechanism.

---

## 7. Limitations

* The generator is 0.8B. The pointwise judge scores its clean texts 1.5 out of 5 and finds a major flaw in 83% of them,
  so coarse quality metrics sit near their floor and cannot show much damage; the watermark-specific metric
  (insertions and drift) is the informative one. A stronger generator has more near-equivalent choices to hide a mark
  in — and more quality to lose. How the trade-off looks there is open.
* The sweep is 60 development prompts and 12 keys; the interval on the headline detection rate is 68–88%.
* The presence statistic also flags a marked text under a *wrong* key about a quarter of the time; "which key" needs the
  decoy-rank test.
* The pairwise measure is noisy on 60 prompts (±0.09) and depends on which clean run is the comparator; the held-out
  comparison (150 prompts, clean-vs-clean 0.47) is the reliable one.
* No human ratings yet; quality rests on two LLM judges that agree with each other.
* Seed-phrase keys are research keys, not cryptography: whoever holds the phrase and the reference model can detect,
  and could also try to remove or forge the mark.
* **Not robust to paraphrase by a capable model** (85% → 12% → 4%): whoever runs the text through a strong rewriter
  removes the mark.
* Quality on the development sweep was flattered by selecting the best of 14 settings on 60 prompts; quote the held-out
  numbers.
* False-positive rates below roughly 0.5% cannot be resolved with 1,200 clean texts.
* English only; three prose domains (plus IFEval). Code generation was looked at early and is out
  of scope here.

---

## 8. Explaining it in two minutes

> "Imagine a small language model that has been told a secret — four random words. Because of the secret, it finds
> some next words slightly more likely than it otherwise would. While our generator writes, wherever it is undecided
> anyway, we nudge it toward those words. The text reads normally, but over a few hundred words the nudges add up.
> Whoever knows the secret replays the text through the small model twice, with and without the secret, and sees
> that the word choices fit the secret far too consistently to be chance. Without the secret there is nothing to see.
>
> Detection needs only the text and the secret, works with a different model than the one that wrote the text, and
> tells different secrets apart. On unseen prompts we detect about nine texts in ten at one false alarm in a hundred.
>
> The question was how hard one can nudge. With an independent judge we measured two things in separate runs: which of
> two anonymous answers is better, and which flaws a single text has — compared with what the clean model produces
> anyway. A gentle nudge is detected in about 85% of texts on unseen prompts at a quarter of a percent of false alarms,
> by a different model than the one that wrote the text, and costs nothing on verifiable instruction following. It is
> not free: the judge prefers the clean answer about two to one, and words related to the secret show up in about 15
> more texts per hundred — though reading pairs side by side, the difference is hard to see. Nudge harder and detection
> goes to 95–100% with clearly visible damage.
>
> Two lessons. What made the gentle nudge detectable was the detector: capping how much any single word can count, so
> the evidence comes from many small preferences instead of a few loud ones. And the open weakness is paraphrase: a
> strong rewriting model erases the mark."

**Questions colleagues are likely to ask**

* *How is this different from green-list watermarks?* The preference list is not a hash of previous tokens but a
  language model's reaction to a secret phrase; detection is a likelihood contrast, not a token count; the key is a
  phrase; the marking model can differ from the generator.
* *Can I detect without the generator?* Yes. Only the reference model (1B parameters) and the phrase are needed.
* *What is the false-positive rate?* Thresholds are set empirically at 1% on 1,200 clean texts and checked on separate
  clean texts: 0.25% on the held-out prompts, 0.9% on IFEval answers, 1–2% on paraphrased clean text. The `T` formula is a score, not a p-value; the p-value comes
  from the empirical clean distribution.
* *Does it survive paraphrasing?* Not against a capable rewriter: after a rewrite by a 31B model detection drops from
  85% to 12% (light) and 4% (strong). Against the 0.8B model paraphrasing itself about half had survived.
* *Does the text get worse?* Moderately. On unseen prompts the clean answer is preferred about 2 : 1 (win rate 0.31 on
  prose, 0.44 on instruction answers); major flaws rise from 79% to 89% of texts and unrelated insertions from 7% to
  22% — about half the damage of the earlier settings. In a blind read of nine pairs the difference was not obvious,
  and the official instruction-following checker shows no loss.
* *How is quality measured?* In two separate judge runs: "which is better" (two anonymous responses, both orders, one
  question, ties allowed) and "flaws" (one anonymous text per call, compared with the clean generator's own rates from
  independent clean draws of the same prompts).
* *Why such a small generator?* Cost. It makes hundreds of controlled comparisons affordable; it also makes absolute
  quality numbers meaningless, hence paired comparisons only.

---

## 9. Where things are

| what | where |
|---|---|
| precise algorithm description (no product names) | `docs/algorithm_spec.md` |
| rules fixed before the current round | `docs/preregistration.md` |
| related work | `docs/related_work.md` |
| generation (push, guards, controller, spans, rotation, warm-up, semantic guard) | `wmworld/runtime.py`, `wmworld/generation.py` |
| reference model, vocabulary bridge, per-position statistics | `wmworld/reference.py` |
| detector variants, frozen-null evaluation | `wmworld/posstats.py`, `scripts/posstats_dump.py`, `scripts/detector_variants.py` |
| blind judge and paraphrase attack through an API | `scripts/api_judge.py`, `scripts/api_paraphrase.py` |
| mechanical selection | `scripts/select_config.py` → `runs/selection.json` |
| raw results | `runs/<run>/eval/*.json`, `runs/detector_variants_*.json` |
