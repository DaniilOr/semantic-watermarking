# Likelihood-ratio detector on the large-scale runs

The detector is the likelihood-ratio (LR) detector from "Push+Margin with a likelihood-ratio detector: results to date" (3 November 2026), section 1. Here it is applied to the large-scale runs behind that report's tables 7–11, next to the count detector they used.

## Method

- **Score.** For sentences i = 2..n, dᵢ is sentence i's signed depth inside the valid set given by the key and sentence i−1's region (`semwm.signed_distance`; d > 0 inside):

  S = (1/√(n−1)) · Σᵢ log( p_wm(dᵢ) / p_null(dᵢ) )

- **Densities.** p_wm and p_null are 52-bin histograms of d, fitted on one half of the prompts:
  - p_wm on clean Push + Margin texts, reused for every arm (as in the report).
  - p_null on human texts and the plain model's texts.
- **Held-out evaluation.** Thresholds and rates come only from the other half.
- **Threshold ("dual null").** The stricter of the 1% thresholds on human texts and on the plain model's texts.
- **Comparison.** The count detector (`z_chain`, SemStamp's) is re-scored on the same evaluation half with the same threshold rule, so both detectors see identical texts.
- **Choices the report does not specify:**
  - Bins: 50 equal-width bins between the 0.5% and 99.5% quantiles of the pooled fitting depths, plus one tail bin on each side.
  - Pseudo-count: 0.5 per bin.
  - Halves: assigned by a keyed hash of the prompt id.
  - Texts with fewer than 2 sentences get no score, as with the count.
- **Scope:** 17 settings.
  - 6 generators on SemStamp's datasets.
  - 6 generators on our benchmark.
  - 3 chat models on the instruct tasks.
  - OPT-1.3B and Llama-3B base in SemStamp's setting.
  - Every attack that was run on these: Gemma paraphrase light and strong, Pegasus, Parrot, SemStamp's own Parrot pipeline, the bigram variants, reorder and resegment.
- **Size:** 682,441 distinct texts.

## Clean texts

% detected at the dual-null 1% threshold, as an unweighted mean over each benchmark's datasets.

| Setting | SemStamp, count | SemStamp, LR | Push, count | Push, LR |
|---|---|---|---|---|
| SemStamp's datasets, Llama-3B chat | 61.0 | 78.6 | 83.2 | **94.8** |
| SemStamp's datasets, Llama-1B chat | 61.5 | 78.2 | 83.2 | **94.1** |
| SemStamp's datasets, Qwen-0.8B chat | 58.4 | 83.6 | 67.9 | **92.5** |
| SemStamp's datasets, 3 base models | 78.2–93.0 | 95.6–98.5 | 85.2–98.2 | **98.7–99.7** |
| Our benchmark, Llama-3B chat | 35.1 | 67.5 | 62.7 | **87.7** |
| Our benchmark, Llama-1B chat | 43.7 | 71.0 | 72.1 | **86.2** |
| Our benchmark, Qwen-0.8B chat | 43.0 | 77.2 | 54.3 | **87.7** |
| Our benchmark, 3 base models | 69.2–78.4 | 88.9–92.2 | 79.7–91.2 | **95.6–98.1** |
| Instruct tasks, Llama-3B / 1B / Qwen | 35.1 / 39.0 / 42.6 | 42.6 / 52.7 / 59.5 | 50.8 / 58.5 / 57.2 | **50.8 / 61.8 / 69.4** |
| OPT-1.3B, SemStamp's setting | 97.6 | 98.9 | 99.6 | **99.6** |

Across the 17 settings, LR detects more than the count:

| | Settings where LR detects more | Mean gain | Range |
|---|---|---|---|
| SemStamp | all 17 | +16.5 points | +1.3 to +34.2 |
| Push + Margin | 15, plus 2 ties (Llama-3B instruct, OPT) | +11.0 points | 0 to +33.4 |

The plain model's second sample is flagged at 0.4–2.0% by LR and 0.3–1.3% by the count.

## Under attack

% detected at the dual-null 1% threshold, as a mean over the settings where each attack was run.

| Attack | Settings | SemStamp, count | SemStamp, LR | Push, count | Push, LR | Settings where LR is higher (SemStamp / push) |
|---|---|---|---|---|---|---|
| Gemma paraphrase, light | 12 | 27.7 | 39.2 | 29.3 | **41.1** | 10 / 10 |
| Gemma paraphrase, strong | 12 | 8.9 | 20.2 | 9.5 | **21.2** | 11 / 11 |
| Pegasus | 14 | 41.8 | 59.0 | 48.6 | **66.5** | 12 / 12 |
| Pegasus-bigram | 14 | 47.7 | 64.6 | 54.8 | **72.4** | 12 / 13 |
| Parrot | 14 | 51.4 | 74.1 | 62.4 | **82.5** | 14 / 14 |
| Parrot-bigram | 14 | 48.5 | 69.1 | 56.9 | **76.6** | 14 / 12 |
| Parrot, SemStamp's pipeline | 5 | 50.5 | 58.3 | 57.5 | **64.4** | 3 / 3 |
| Parrot-bigram, SemStamp's pipeline | 5 | 51.0 | 58.7 | 57.8 | **64.8** | 3 / 3 |
| Sentence reorder | 12 | 29.3 | 46.4 | 29.1 | **46.6** | 12 / 12 |
| Resegment | 12 | 22.1 | 34.0 | 25.3 | **36.1** | 11 / 10 |

Of the 114 attacked cells per method:

| | LR detects more | LR detects less | Within 0.5 points |
|---|---|---|---|
| SemStamp | 102 | 9 | 3 |
| Push + Margin | 100 | 8 | 6 |

All 8 push cells where LR is lower are Llama-1B chat on SemStamp's datasets. The largest drop is under light Gemma paraphrase, 24.0 → 17.1. This is not explained yet.

## Caution: thresholds on human text alone

The LR histograms partly learn the difference between model-written and human text. At a threshold set on human text alone, LR flags 24% of the plain Llama-3B BookSum texts (3.6–4.1% on RealNews and Reddit-TIFU), against 0.6–1.4% at the dual-null threshold. So LR rates should only be reported at the dual-null threshold.

A 0.1% rate is not reported, because about 500 plain texts per evaluation half cannot support it.

## Files

| Path | Contents |
|---|---|
| `results/lr/lr_detect_<setting>.json` | per setting: fitted histogram (edges and log-ratio per bin), thresholds per dataset and detector, and rates with Wilson intervals and AUROC per run × arm × dataset |
| `results/lr/lr_summary.json` | the means behind the tables above |
| `code/scripts/lr_detect.py` | depth computation (GPU) and evaluation (CPU) |
| `code/scripts/lr_summary.py` | the tables |
| `code/slurm/conf/lr_*.sbatch` | the jobs; they read the `/nfs-stor` copy of the runs |
