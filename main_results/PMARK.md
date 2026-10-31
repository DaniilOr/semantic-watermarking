# PMark baseline (online), run under our protocol

PMark: *Towards Robust and Distortion-free Semantic-level Watermarking with Channel Constraints* (Huo et al., ICLR 2026).
Reference code: github.com/Z1zs/PMark, commit `75140fe`.

## Overview: SemStamp vs Push + Margin vs PMark

The model is Llama-3.2-3B-Instruct unless a row says otherwise. "Mean" is an unweighted mean over datasets.

- **Continuation** = RealNews and BookSum (SemStamp's) plus ArXiv, BookSum and Wikipedia (ours).
- **Attacked** = Pegasus, Pegasus-bigram, Parrot and Parrot-bigram on RealNews, ArXiv, BookSum (ours) and Wikipedia (16 cells; SemStamp and push were not attacked on SemStamp's BookSum).
- **Instruct thresholds** are 1% on the plain model's own answers, because PMark's detector flags plain answers at its threshold on human text.

| | SemStamp | Push + Margin | PMark |
|---|---|---|---|
| **Detection, clean continuation**, % at 1% / 0.1% false positives (mean of 5) | 54.0 / 25.5 | **79.4 / 47.9** | 49.1 / 15.4 |
| **Detection, clean instruct** (WritingPrompts, Dolly, GSM8K, IFEval), % at 1% (mean of 4) | 44.6 | **64.0** | 27.2 |
| **Detection after paraphrase**, % at 1% / 0.1% (mean of 16) | 38.0 / 15.0 | **53.2 / 25.0** | 20.6 / 5.4 |
| **OPT-1.3B, SemStamp's setting**: clean, % at 1% / 0.1% | 97.7 / 93.6 | **99.1 / 98.5** | 98.7 / 95.6 |
| **OPT-1.3B**: after paraphrase, % at 1% / 0.1% (mean of 4) | 92.0 / 83.7 | **93.5 / 86.8** | 77.9 / 59.0 |
| **False alarms on the plain model's own text**, % flagged at 1% on human text | **0.8–1.6** (9 datasets; same detector as push) | **0.8–1.6** | 0.0–1.6 on continuation; **45.3** Dolly, **72.6** GSM8K, **15.5** OPT |
| **Perplexity, OPT-2.7B**, median / mean (plain: 8.77 / 9.45; mean of 5 continuation datasets) | 8.84 / 9.76 | 9.24 / 11.34 | **8.71 / 9.35** |
| **Perplexity, OPT-1.3B setting**, median / mean (plain: 8.55 / 8.84) | **8.46 / 8.34** | 9.90 / 10.06 | 8.94 / 9.31 |
| **GSM8K accuracy**, % (plain 78.5) | **77.6** [75.2–79.7] | 74.5 | 75.1 |
| **IFEval strict prompt-level**, % (plain 61.9) | **62.8** [58.7–66.8] | 59.7 | 59.9 |
| **Judge win rate vs plain**, continuation (SemStamp's / ours) | 0.482 / 0.478 | 0.476 / 0.482 | not judged yet (API credits) |
| **Candidates drawn per kept sentence**, continuation | 44.8–49.0 | **10.8–11.0** | 64.2 |
| **Generation time**, median s per 200-token text (3 generators per A100-40GB) | 105–108 | 41–44 | **30–32** |
| **What detection needs** | sentence encoder only | sentence encoder only | the generating LM, with 64 re-samples per sentence (32–35 s per text on a GPU) |

On GSM8K, push and PMark are both below SemStamp's 95% interval. On IFEval, both are inside it.

### Per task: detection and robustness

All rows are Llama-3.2-3B-Instruct except the OPT-1.3B row. Each cell is SemStamp / Push + Margin / PMark.

- **Clean** is % detected at 1% / 0.1% false positives on human text. The instruct rows use 1% on the plain model's answers.
- **After paraphrase** is % at 1%, as a mean over Pegasus, Pegasus-bigram, Parrot and Parrot-bigram.
- **Plain flagged** is % of the plain model's own texts flagged at 1% on human text. The first number is the detector SemStamp and push share; the second is PMark's.

| Task | Clean | After paraphrase | Plain flagged |
|---|---|---|---|
| RealNews | 59.6/32.8 · 79.8/63.7 · **90.6/67.8** | 46.6 · **58.4** · 54.7 | 1.6 · 0.7 |
| BookSum (SemStamp's) | 67.6/33.0 · **90.8/63.5** · 51.2/2.8 | not run · not run · 6.9 | 1.0 · 0.0 |
| ArXiv | 44.3/7.8 · **77.5/19.1** · 2.2/0.0 | 34.3 · **52.7** · 0.1 | 1.6 · 0.0 |
| BookSum (ours) | 66.5/46.9 · **91.6/77.4** · 48.8/5.6 | 45.5 · **62.9** · 5.7 | 1.0 · 0.0 |
| Wikipedia | 31.9/7.1 · **57.4/16.0** · 52.8/0.8 | 25.6 · **38.9** · 22.1 | 1.4 · 1.6 |
| WritingPrompts | 94.5 · **99.6** · 97.9 | not run · not run · 58.8 ³ | 1.3 · 0.7 |
| Dolly | 29.8 · **61.7** · 3.2 | not run · not run · 11.0 ³ | 0.8 · **45.3** |
| GSM8K | 6.7 · **24.0** · 5.4 | not run · not run · 13.1 ³ | 1.1 · **72.6** |
| IFEval | 47.2 · **70.7** · 2.2 | not run · not run · 0.0 ³ | no human answers |
| OPT-1.3B, RealNews | 97.7/93.6 · **99.1/98.5** · 98.7/95.6 | 92.0 · **93.5** · 77.9 | 1.0 · **15.5** |

³ PMark under paraphrase on the instruct tasks, at its 1% threshold on human text (IFEval: on plain answers). SemStamp and push were not attacked on these tasks. On Dolly and GSM8K this threshold also flags 45% / 73% of plain answers (last column).

### Per task: quality and cost

Quality is perplexity (OPT-2.7B, median, lower is better) or the task metric (higher is better). Each cell is Plain / SemStamp / Push + Margin / PMark.

- **Judge** is the win rate vs plain, SemStamp / push. PMark has not been judged yet.
- **Candidates** are drawn per kept sentence, SemStamp / push / PMark.
- **Seconds per text** is the mean generation time as run. Generators were 3 per A100-40GB, except the PMark instruct refill, which ran 2 per GPU.

| Task | Quality metric | Plain / SemStamp / Push / PMark | Judge (SemStamp / push) | Candidates | Seconds per text |
|---|---|---|---|---|---|
| RealNews | PPL median | 7.6 / 7.7 / 7.9 / 7.7 | 0.477 / 0.462 | 42.1 / **10.6** / 64.1 | 99 / 41 / **28** |
| BookSum (SemStamp's) | PPL median | 9.3 / 9.4 / 9.8 / **9.2** | 0.474 / 0.478 | 47.9 / **11.1** / 64.0 | 123 / 45 / **34** |
| ArXiv | PPL median | 8.7 / 8.7 / 9.2 / 8.7 | 0.501 / 0.497 | 49.4 / **10.7** / 64.0 | 127 / 44 / **29** |
| BookSum (ours) | PPL median | 9.3 / 9.3 / 9.9 / 9.4 | 0.480 / 0.484 | 47.3 / **11.0** / 64.1 | 127 / 46 / **36** |
| Wikipedia | PPL median | 9.1 / 9.1 / 9.4 / **8.6** | 0.452 / 0.465 | 51.4 / **10.8** / 64.5 | 87 / 31 / **22** |
| WritingPrompts | judge only | – | 0.493 / 0.486 | 57.3 / **12.0** / 64.0 | 225 / 149 / **90** |
| Dolly | judge only | – | 0.451 / 0.409 | 78.1 / **14.1** / 64.0 | 350 / 92 / **61** |
| GSM8K | accuracy, % | 78.5 / **77.6** / 74.5 / 75.1 | 0.482 / 0.450 | 86.5 / **14.9** / 64.0 | 312 / 68 / **49** |
| IFEval | strict prompt-level, % | 61.9 / **62.8** / 59.7 / 59.9 | 0.459 / 0.463 | 61.9 / **12.6** / 64.0 | 253 / 79 / **53** |
| OPT-1.3B, RealNews | PPL median | 8.6 / **8.5** / 9.9 / 8.9 | – | – | – |

## What was run

- **Method.** This is online PMark with the released settings:
  - N = 64 candidate next sentences per sentence.
  - b = 4 orthonormal pivots, from the QR decomposition of a seed-42 Gaussian matrix.
  - all-mpnet-base-v2 embeddings.
  - Channel-by-channel halving at the Harrell–Davis median.
  - The released key file (`secret_mbit.json`), which sets one group of bits per sentence position.
  - Detection re-samples 64 candidates per sentence with the generating LM. It scores with the released soft count (K = 250, δ = 0.001) and a z-test.
- **Code.**
  - The reference needs vLLM, so we re-implemented it inside our generator (`code/scripts/pmark_arm.py`).
  - `code/scripts/pmark_check.py` runs the reference's own functions (with vLLM stubbed out) on random inputs. Pivots, median, key bits, the candidate filter (300/300 trials) and the detector's counts and z-score (200/200 trials) all match.
- **Protocol: the same as for SemStamp and Push + Margin.**
  - Same prompts, chat template and sampling.
  - Texts are capped at 200 tokens (512 for instruct tasks).
  - Thresholds are set at 1% and 0.1% false positives on the same human texts (2,000 per dataset).
  - The same sentence-level paraphrase attacks (Pegasus, Parrot; top beam and bigram variants).
- **One change to sentence splitting.** A leading line break joins the next sentence, because PMark's key depends on sentence position and generator and detector must split text the same way.
  - Generator and detector split identically in 96–98% of PMark texts.
  - In the remaining 2–4%, the key bits shift after the first mismatch.
- **Models.**
  - Llama-3.2-3B-Instruct on SemStamp's datasets (RealNews, BookSum), our benchmark (ArXiv, BookSum, Wikipedia) and the instruct tasks (WritingPrompts, Dolly, IFEval, GSM8K).
  - OPT-1.3B in SemStamp's original setting (RealNews).

## Detection on clean texts

% of watermarked texts detected at 1% / 0.1% false positives on human text. For IFEval there are no human answers, so the threshold is 1% on the plain model's answers.

| Model | Dataset | SemStamp | Push + Margin | PMark |
|---|---|---|---|---|
| Llama-3B chat | RealNews | 59.6 / 32.8 | 79.8 / 63.7 | **90.6 / 67.8** |
| | BookSum (SemStamp's) | 67.6 / 33.0 | **90.8 / 63.5** | 51.2 / 2.8 |
| | ArXiv | 44.3 / 7.8 | **77.5 / 19.1** | 2.2 / 0.0 |
| | BookSum (ours) | 66.5 / 46.9 | **91.6 / 77.4** | 48.8 / 5.6 |
| | Wikipedia | 31.9 / 7.1 | **57.4 / 16.0** | 52.8 / 0.8 |
| | WritingPrompts | 95.6 / 89.0 | **99.6 / 99.3** | 97.8 / 84.9 |
| | Dolly | 27.2 / 7.0 | 60.1 / 27.6 | 75.5 / 52.8 ¹ |
| | GSM8K | 7.4 / 2.2 | 25.8 / 10.1 | 98.6 / 86.3 ¹ |
| | IFEval (plain-model threshold, 1%) | 47.2 | **70.7** | 2.2 |
| OPT-1.3B | RealNews (SemStamp's setting) | 97.7 / 93.6 | **99.1 / 98.5** | 98.7 / 95.6 |

¹ PMark's detector also flags most of the plain model's own unwatermarked answers on these tasks; see the next section.

## Detection after SemStamp's sentence paraphrase attacks

% detected at 1% false positives on human text. Each cell is SemStamp / Push + Margin / PMark.

| Model | Dataset | Pegasus | Pegasus-bigram | Parrot | Parrot-bigram |
|---|---|---|---|---|---|
| Llama-3B chat | RealNews | 44.0 / 53.2 / 46.1 | 48.1 / 57.4 / 55.4 | 50.4 / 65.5 / 63.8 | 43.7 / 57.4 / 53.5 |
| | BookSum (SemStamp's) ² | – / – / 1.9 | – / – / 5.4 | – / – / 10.9 | – / – / 9.5 |
| | ArXiv | 32.7 / 44.8 / 0.0 | 37.7 / 54.1 / 0.0 | 34.2 / 60.2 / 0.2 | 32.4 / 51.7 / 0.2 |
| | BookSum (ours) | 36.0 / 52.9 / 0.8 | 43.9 / 62.8 / 5.0 | 53.1 / 70.9 / 8.4 | 49.1 / 64.8 / 8.4 |
| | Wikipedia | 22.5 / 29.4 / 9.4 | 24.9 / 35.8 / 18.2 | 28.2 / 49.4 / 34.2 | 26.8 / 41.0 / 26.6 |
| OPT-1.3B | RealNews (SemStamp's setting) | 86.1 / 87.1 / 69.1 | 91.8 / 92.7 / 72.2 | 96.0 / 97.2 / 87.7 | 94.1 / 97.0 / 82.7 |

² SemStamp and push were not attacked on this dataset.

- **At 0.1% false positives, push detects more than PMark in every attacked row above.** All values, including the 0.1% ones, are in `results/pmark_tables.txt`.
- **Instruct tasks:** PMark's attacked results are in `results/pmark_tables.txt` (mean at 1%: WritingPrompts 58.8, Dolly 11.0, GSM8K 13.1, IFEval 0.0). SemStamp and push have no instruct-task attack runs.

## False positives on the plain model's own text

% of the plain model's own unwatermarked texts that PMark's detector flags at its 1% threshold on human text:

| Model | Dataset | Flagged |
|---|---|---|
| Llama-3B chat | RealNews / BookSum (SemStamp's) | 0.7 / 0.0 |
| | ArXiv / BookSum (ours) / Wikipedia | 0.0 / 0.0 / 1.6 |
| | WritingPrompts | 0.7 |
| | Dolly | **45.3** |
| | GSM8K | **72.6** |
| OPT-1.3B | RealNews (SemStamp's setting) | **15.5** |

**Why.** The released detector counts a sentence as a hit for either key bit when its proxy is within δ = 0.001 of the re-sampled median. In near-deterministic text (math steps, list headers), all 64 re-samples are often the same sentence, so every channel scores a free hit.

| Plain Llama-3B text | Channel decisions on the key's side (0.5 expected without a watermark) |
|---|---|
| GSM8K | 0.78 |
| Dolly | 0.73 |
| Human answers | 0.50–0.51 |

The PMark paper calibrates only against human text, where this does not show up.

## Why PMark fails on BookSum and ArXiv

The watermark itself is as strong on these datasets as on RealNews. Ignoring ties, 88% of channel decisions in PMark's texts land on the key's side on every dataset. What fails is the threshold:

1. **Without a watermark, the soft count drifts upward.** Ties count as hits and misses still earn partial credit, exp(−K·abs(F − median)). The drift grows with the number of sentences in a text.
2. **The human texts have more sentences than PMark's 200-token texts:**

   | Dataset | Human-text sentences | PMark-text sentences | Human-text 1% / 0.1% threshold (z) |
   |---|---|---|---|
   | BookSum (SemStamp's) | 19.7 | 8.5 | 4.75 / 7.35 |
   | ArXiv | 7.8 | 5.7 | 5.29 / 8.93 |
   | RealNews | 8.8 | 6.2 | 2.97 / 3.92 |

   On ArXiv, 4.2% of human sentences tie exactly with the median.
3. **A text with T sentences can reach at most z = √(4T).** That is 5.8 for 8.5 sentences and 4.8 for 5.7, below the BookSum 0.1% and ArXiv 1% thresholds.

The PMark code avoids this situation by construction. Its generator keeps going until a text has at least 12 sentences, and its detector cuts human texts to their first 12 sentences.

**Diagnostic (ours, not PMark's rule).** We re-scored the same cached data with a neutral rule: a tie counts 0.5 and a miss counts 0.

| Dataset | % detected at 1% / 0.1%: released rule | Neutral rule | Human-text z at the 99.9th percentile: released → neutral |
|---|---|---|---|
| BookSum (SemStamp's) | 51.2 / 2.8 | 95.8 / 79.6 | 7.04 → 3.21 |
| RealNews | 90.6 / 67.8 | 93.3 / 79.1 | 3.89 → 2.83 |

The plain model's text stays at 0.0–0.6% flagged under both rules.

## Cost

- **Generation:** PMark draws 64 candidates for every sentence. For comparison, SemStamp averages 31–51 candidates per kept sentence and push 7–11 in our runs.
- **Detection:** the detector needs the generating model and draws another 64 per sentence. With 2 detection lanes per A100-40GB, our runs took:
  - 32–35 s per 200-token text (Llama-3B)
  - 52–58 s per instruct answer
  - 17 s per OPT text

## Coverage

- **Out-of-memory skips.** A few very long contexts would not fit in GPU memory even after splitting the batch, so those texts are left unscored:
  - SemStamp's datasets, our benchmark and OPT: 1 attacked text per setting.
  - Instruct tasks: 1 human Dolly text.

  Per-group counts are in each report's `coverage` field.

## Files

| Path | Contents |
|---|---|
| `results/pmark_detect_{sp,ob,opt,it}.json` | thresholds, detection rates with Wilson intervals, AUROC, coverage, sentence alignment |
| `results/pmark_tables.txt` | the comparison tables, plain-model false positives, the neutral-tie diagnostic |
| `code/scripts/pmark_arm.py` | generation (wraps our generator), online scoring, report |
| `code/scripts/pmark_check.py` | equivalence checks against the reference functions |
| `code/scripts/pmark_compare.py`, `pmark_domain_probe.py`, `pmark_neutral_ties.py`, `pmark_null_probe.py`, `pmark_len_null.py` | comparison and diagnostics |
| `code/slurm/conf/pmark_*.sbatch` | the jobs (generation, detection with attacks, instruct refill, smoke test) |
