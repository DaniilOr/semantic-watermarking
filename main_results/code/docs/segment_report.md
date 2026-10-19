# Segment-level SemStamp: a bounded selection chain with a semantic push

*Report, 25 September 2026. Every number is measured and comes from the files listed in Section 10. Intervals are
95% Wilson intervals unless stated otherwise. The OpenStamp-style benchmark (ArXiv, BookSum, Wikipedia) is still
running; Section 9 gives its setup only, without results.*

---

## 1. Summary

SemStamp (Hou et al., NAACL 2024) is a semantic watermark. The generator rejection-samples every sentence until the
sentence's embedding falls into a secret region of embedding space. We built a variant that differs in three ways:

1. **Units are fixed 16-word segments, not sentences.**
2. **There is no open-ended rejection loop.** The generator draws at most 16 candidates per segment and keeps the best.
3. **A light semantic push (optional).** While a candidate is being sampled, tokens that move the segment deeper into
   the secret region get a small logit bonus. The push is limited to the model's own plausible tokens.

All numbers below are for Llama-3.2-3B-Instruct on the same 150 held-out test prompts: 50 news-article continuations
from C4 RealNewsLike, 50 instructions from Databricks Dolly-15k, and 50 story prompts from WritingPrompts. Section 3.1
describes the datasets. Detection uses a threshold set at 1% false-positive rate on unwatermarked text.

| | Shipped SemStamp | Segment chain | Segment + push | Segment + push + margin | Sentence version (push + margin) |
|---|---|---|---|---|---|
| TPR @ 1% FPR, clean text | 68.0% [60–75] | 83.3% [77–88] | 93.3% [88–96] | **94.0% [89–97]** | 91.3% [86–95] |
| TPR @ 1% FPR, light paraphrase | **27.3%** | 2.0% | 2.0% | 10.7% | 26.0% |
| TPR @ 1% FPR, strong paraphrase | **7.3%** | 1.3% | 1.3% | 3.3% | **7.3%** |
| Candidates per unit | 60.2 | 8.06 | **6.90** | 12.04 | 12.59 |
| Seconds per text | 319.5 | **42.6** | 58.1 | 141.2 | 111.6 |
| Judge win rate vs clean (no-effect level 0.425) | 0.485 | 0.458 | 0.400 | 0.402 | 0.433 |
| Flaw measures outside the clean band (of 5) | 0 | 0 | 0 | 0 | 0 |

Plain generation without a watermark takes 14.1 s per text under the same conditions.

**Findings**

- The segment arms give the highest clean detection we measured: 93–94% with the push, against 68% for shipped
  SemStamp. They use 5–9× fewer candidates and 2.3–5.5× less time.
- On quality, none of the arms has a flaw measure outside the clean band. The judge's win rate drops from 0.458 to
  0.400 when the push is added. The no-effect level is 0.425, and the intervals overlap.
- The segment arms lose most of the watermark under paraphrase: 2–11% survives light paraphrase, against 26–27% for the
  sentence version and for SemStamp. Paraphrase robustness is why the sentence version, not the segment version, is
  our current reference configuration.

---

## 2. Background: how SemStamp works

- **Embedding and bins.** An encoder maps each sentence to a unit vector. The key fixes 3 random hyperplanes through
  the origin. The signs of the vector's 3 projections give a 3-bit number: its **bin**, one of 8.
- **Mask.** Each sentence has a **mask**: 2 of the 8 bins (λ = 0.25), chosen pseudo-randomly from the **previous
  sentence's bin** and the key. Consecutive sentences therefore form a chain.
- **Generation (rejection sampling).** The model samples a sentence. The sentence is accepted only if its bin is in the
  mask *and* it lies at least 0.02 (in cosine) from every hyperplane. That distance is the **margin**, which makes the
  bin survive small edits. Otherwise the model samples again. After a cap (100 candidates in our implementation), an
  unfiltered candidate is kept.
- **Detection.** The detector needs only the text and the key. It splits the text into sentences, embeds them, counts
  how many sentences (from the second on) land in the mask seeded by their predecessor, and computes a z-score against
  the 25% expected by chance.
- **Appeal.** Every accepted sentence is an unmodified sample from the model, so text quality stays close to
  unwatermarked text. And because the signal lives in meaning rather than in word choice, part of it survives
  paraphrase.
- **Cost on our stack.** Shipped SemStamp needed **60.2 candidate sentences per accepted sentence** (188,072 candidates
  for 3,126 sentences). **46.5%** of accepted sentences hit the 100-candidate cap and were forced through without
  meeting the rule. It took **319.5 s per text**, against 14.1 s for plain generation.

---

## 3. Experimental setup (identical for every arm)

### 3.1 Data

Three text domains, chosen to cover news, instruction answering and creative writing.

| Domain | Source | Split used | Filters | Prompt given to the model |
|---|---|---|---|---|
| **C4 RealNewsLike** (news) | `allenai/c4`, `realnewslike` subset, validation shard `c4-validation.00000-of-00001` (revision 1588ec45) | validation | English; document at least 96 + 64 tokens long | "Continue the following news article. Do not repeat the given text; write what comes next." followed by the document's first 96 tokens (Qwen3.5 tokenizer) |
| **Dolly** (instructions) | `databricks/databricks-dolly-15k` (revision bdd27f4d) | the single released file | Categories `brainstorming`, `creative_writing` and `general_qa` only; instruction at least 5 words; reference answer at least 40 words; context at most 400 words; English | The instruction, plus "Context: …" when the item has one |
| **WritingPrompts** (stories) | Fan et al. 2018 (ACL), fairseq release `writingPrompts.tar.gz` | test split for test prompts, validation split for null prompts | Prompt 5–120 words; story 60–6,000 words; no deleted or removed posts; English; no URL in the prompt | "Write a story based on the following writing prompt." followed by "Prompt: …" |

How the partitions were built:
- The same pipeline (`wmworld/data/prepare.py`) filtered every source and grouped near-duplicate prompts and
  documents.
- It then split each domain into partitions that share no duplicate group. Two of them are used here:
  - **`positive_test`**, 200 prompts per domain. The test set is its first 50 per domain in a fixed hash order: 150
    prompts, none of which was used during development.
  - **`null_calibration`**, a separate partition. It supplies the prompts for the unwatermarked texts that set the
    detection thresholds: 400 per domain answered by Qwen3.5-0.8B (1,200 texts), and 200 per domain answered by
    Llama-3.2-3B (600 texts).
- WritingPrompts follows its official splits: test prompts come from its test split and null prompts from its
  validation split. Dolly and C4 have no suitable official splits, so they are split deterministically by hash.

The watermarked texts, the two extra clean draws and the attacked copies all use the 150 test prompts. The null texts
never share a prompt, or a near-duplicate of one, with the test set.

### 3.2 Models, key, thresholds and evaluation

| Item | Value |
|---|---|
| Generator | Llama-3.2-3B-Instruct, bf16 (revision 0cb88a4f) |
| Sampling | temperature 0.8, top-p 0.95, no top-k; 64–512 new tokens |
| Encoder | `AbeHou/SemStamp-c4-sbert`, the SemStamp authors' encoder (revision a77dce85) |
| Key | SemStamp's shipped constants: hyperplanes from `RandomState(1234)`, mask seed 15485863 × previous bin, 3 planes, 8 bins, λ = 0.25, margin 0.02 |
| Test prompts | 150 held-out prompts, 50 per domain (Section 3.1). Mean output 306–312 words for every arm |
| Thresholds | 1% upper tail of the 1,800 unwatermarked null texts of Section 3.1, one threshold per statistic, computed from these null texts only. Segment chain: z > 3.118. Sentence chain: z > 2.837 |
| False-positive checks | Two more clean Llama-3B draws on the test prompts (150 texts each); 6 wrong keys |
| Attacks | Listed in Section 6.2 |
| Quality judge | `google/gemma-4-31b-it` via OpenRouter; details in Section 6.4 |
| Hardware | NVIDIA A100-SXM4-40GB. Every arm was generated as 3 processes sharing one GPU. "Seconds per text" is the mean wall-clock time per text of one process, so it includes that sharing |

---

## 4. The method

### 4.1 Units: 16-word segments

The text is cut every 16 whitespace-separated words. The detector cuts the same way, and drops a final piece shorter
than 8 words. The generator controls exactly where every unit ends: it stops each candidate at the 16th word.
Sentence boundaries, by contrast, are decided by the model's punctuation. For scale, sentences average about 14.8
words in these texts, so the two unit types are about the same length.

**Why segments, and why 16 words.** The segment unit comes from the first design (round 1 in Section 5), which had two
goals:
- **Steering without rejection.** A token-level push can change where the model puts punctuation, and so where a
  sentence ends. A word count gives each unit a fixed end that the push cannot move.
- **Surviving reordering.** Each text was aimed at one fixed bin, and each segment was scored on its own, so the score
  does not depend on segment order. A chain detector cannot have that property. As measured later, reordering sentences
  still re-cuts the 16-word segments.

The length was a default, 16, and was not fitted. A study run before any generation (`runs/embedder_study.json`)
compared sentences with 8-, 16- and 32-word segments on existing texts. It measured the share of units whose bin is
unchanged after each attack:

| Unit | Units per text | Light paraphrase | Strong paraphrase | Sentence reorder | Resegment |
|---|---|---|---|---|---|
| Sentence | 21.3 | **0.642** | **0.567** | **0.991** | **0.777** |
| 8 words | 35.8 | 0.479 | 0.447 | 0.583 | 0.657 |
| 16 words | 17.8 | 0.494 | 0.454 | 0.548 | 0.670 |
| 32 words | 8.9 | 0.514 | 0.466 | 0.536 | 0.737 |

- The three segment lengths differ little from each other.
- Sentences keep their bins better under every attack.
- For the fixed-bin score, all unit types were close: the across-text correlation of the mean projection, original
  against attacked text, was 0.85–0.90 under paraphrase and 0.95–0.99 under reordering and resegmentation.

When the method moved to SemStamp's chain (round 3), the fixed-bin argument no longer applied: the chain score depends
on order for both unit types. Segments were kept as the primary unit because, in a 30-prompt pilot replay, the chain
detected 88% of segment texts against 80% of sentence texts. On the 150 test prompts:
- **without the push:** sentences won, 90.0% against 83.3%
- **with the push:** segments won, 93.3% against 89.9%
- **under paraphrase:** segments lost, as the study above had already indicated

### 4.2 The bounded chain (selection only)

For every segment after the first:

```
mask   = the 2 bins chosen by (key, bin of the previous segment)
drawn  = []
repeat until 16 candidates have been drawn:
    sample 4 candidate continuations of 16 words each (one batch, exact samples from the model)
    embed them and compute their bins
    stop if any candidate is in the mask
         (margin variant: in the mask AND at least 0.02 from every hyperplane)
keep the candidate that lies deepest inside the mask; if none is inside, keep the one closest to it
```

- The first segment is a plain sample. It only seeds the chain.
- **Depth** is how far inside the mask a unit lies: the smallest signed projection over the three hyperplanes of its
  nearest mask bin. It is positive exactly when the unit is inside.
- With the push switched off, every kept segment is an exact, unmodified sample from the model. The only change is
  which sample is kept.
- There is no retry beyond 16. A segment that misses stays in the text and counts against detection.

### 4.3 The semantic push (optional)

The push acts on every token while a candidate is being sampled:

1. **Restrict to the model's own candidates.** Take the model's top-p set (temperature 0.8, top-p 0.95). Tokens
   outside it keep probability zero, so the push cannot introduce a token the model would not have sampled.
2. **Score the options exactly.** Take the 20 most likely word tokens in that set. For each one, append it to the
   segment written so far, embed the resulting text, and measure its depth in the nearest mask bin, capped at 0.05.
3. **Turn the scores into a small bias.** Standardise the 20 scores, using the model's probabilities as weights, and
   clip them to ±1 standard deviation. Multiply by β = 1 and by an entropy gate. The gate is 0 when the model is
   already confident (entropy ≤ 0.2 nats) and reaches full strength at 2.0 nats. The largest possible change is ±1
   logit.
4. **Guards:**
   - Punctuation, whitespace and end-of-text tokens are never biased. The word tokens are shifted as a group so that
     the probability of ending or punctuating stays exactly the model's own. Measured over the push arm: 0.030119
     before and 0.030119 after.
   - A token that would repeat a 4-gram already in the text gets no positive bias.
   - Once all 20 options reach depth 0.05, their scores are equal and nothing is pushed.

Measured for the push + margin arm:
- mean |bias| on pushed steps: 0.437 logits (maximum 0.98)
- per text: 10,976 embedded strings and 33 repetition-guard interventions

The embedding calls are where the push's extra time goes.

### 4.4 Detection

- **Primary statistic:** SemStamp's chain z-score computed over 16-word segments,
  z = (in-mask count − 0.25·n) / √(n · 0.25 · 0.75), where n = number of segments − 1.
- **Also recorded:** a continuous variant that sums each segment's clipped depth (±0.05) instead of counting hits.
  Its results are close to the z-score's; for example, 94.0% for push + margin.
- The detector needs only the text and the key. It does not need the prompt, the model or any record of generation.

### 4.5 Worked example

The example is a C4 news continuation from the push + margin arm. It was chosen as a typical text, not the best one:
at most 30% of its segments are in bin 4 (see Section 6.6), and at least one segment misses. The prompt ends with
"…It can work with dozens of data types -- not only Excel, Access and CSV files, but".

| # | Segment (16 words) | Previous bin | Mask | Bin | In mask | Depth | Candidates drawn |
|---|---|---|---|---|---|---|---|
| 0 | …also JSON, XML, and a wide range of cloud-based data sources, including Azure Blob Storage, Azure | – | – | 3 | seed | – | 1 |
| 1 | Data Lake, and SQL Server. Power BI's robust security features allow users to control who has | 3 | {4, 6} | 4 | yes | 0.025 | 12 |
| 2 | access to their reports, ensuring that sensitive data remains protected. One of the key differentiators of | 4 | {4, 1} | 4 | yes | 0.028 | 8 |
| 3 | Power BI is its collaboration capabilities. Users can share reports with others in real-time, track changes, | 4 | {4, 1} | 1 | yes | 0.018 | 16 |
| 4 | and even create separate workspaces for different teams or projects. This enables seamless communication and data-driven | 1 | {3, 7} | 3 | yes | 0.020 | 16 |
| 5 | decision-making across the organization. In addition to its core features, Power BI also integrates with other | 3 | {4, 6} | 1 | **no** | −0.014 | 16 |
| 6 | Microsoft tools, such as Dynamics 365 and SharePoint, making it a comprehensive solution for businesses looking | 1 | {3, 7} | 7 | yes | 0.022 | 8 |
| 7 | to centralize their data analytics and visualization needs. With its user-friendly interface and robust capabilities, Power | 7 | {5, 4} | 4 | yes | 0.023 | 8 |
| 8 | BI is poised to disrupt the data analysis market and become a leading player in the | 4 | {4, 1} | 1 | yes | 0.022 | 16 |
| 9 | space. | 1 | {3, 7} | 3 | yes | 0.019 | 16 |

How to read the table:
- Segments 3 and 4 used all 16 candidates without finding one clear of the 0.02 margin. The deepest in-mask candidate
  was kept.
- Segment 5 found no in-mask candidate in 16 draws. The candidate closest to the mask was kept, and it counts as a
  miss.
- Segment 9 is one word, so the detector drops it.

The detector, recomputed from the text alone, reproduces bins 0–8 exactly. That gives 8 transitions, 7 of them in
the mask: z = (7 − 2) / √1.5 = **4.08**, above the 3.118 threshold, so the text is flagged.

---

## 5. How we got here: every segment-level arm we ran

All rows use the same 150 prompts, key, encoder and thresholds. Earlier rounds aimed each whole text at **one fixed
bin**, with no chain, and scored it with an order-free statistic ("t_seg": the mean clipped depth in that bin).

| Round | Arm | What it does | Clean detection | Judge win | Flaw measures outside band | s/text |
|---|---|---|---|---|---|---|
| 1 | steered_seg β=4 | Token-level bias toward the fixed bin, looked up from a precomputed table of each token's embedding direction (no rejection) | 46.0% | 0.102 | 5/5 | 16.8 |
| 1 | steered_seg β=2 | Same, weaker bias | 10.0% | 0.387 | 0/5 | 14.9 |
| 2 | segA β=2 / β=3 | Table push restricted to the model's top-p set, with the guards of 4.3 | 1.3% / 2.7% | 0.462 / 0.443 | 1/5 each (insertions) | 16.7 / 16.7 |
| 2 | segB β=2 | Exact push (4.3) toward the fixed bin | 10.7% | 0.418 | 0/5 | 18.2 |
| 2 | best-of-4 | 4 exact samples per 16-word chunk, keep the deepest in the fixed bin | 22.0% | 0.443 | 0/5 | 22.7 |
| 2 | best-of-4 + push β=1 | Same, with the exact push | 52.7% | 0.425 | 2/5 | 35.3 |
| 3 | segment chain | SemStamp's chain over segments, bounded selection (4.2) | 83.3% | 0.458 | 0/5 | 42.6 |
| 3 | + margin 0.02 | Stop only on a candidate clear of the margin | 86.7% | 0.485 | 0/5 | 69.5 |
| 3 | + push β=1 | Chain with the exact push (4.3) | 93.3% | 0.400 | 0/5 | 58.1 |
| 3 | + push + margin | Both | 94.0% | 0.402 | 0/5 | 141.2 |

What each round showed:
- **Round 1:** at the strength needed for detection, a token bias alone destroyed the text: the judge scored the β=4 texts 1.82 out of 5
  (clean text: 3.68), and their win rate was 0.102.
- **Round 2:** limiting the bias to the model's own top-p set and adding the guards kept the text clean, but detection
  stayed at 1–11%.
- **Round 3:** switching from one fixed bin to SemStamp's chain, and choosing among exact samples, gave the jump from
  22–53% to 83–94%.

---

## 6. Results (150 held-out prompts)

### 6.1 Detection and false positives

| Arm | Detection at 1% FPR | AUROC vs the 1,800 null texts | Wrong-key flag rate (6 keys) |
|---|---|---|---|
| Shipped SemStamp (sentence chain) | 68.0% [60–75] | 0.946 | 0.7% |
| Segment chain | 83.3% [77–88] | 0.987 | 1.2% |
| Segment + margin | 86.7% [80–91] | 0.982 | 1.3% |
| Segment + push | 93.3% [88–96] | 0.991 | 0.8% |
| Segment + push + margin | 94.0% [89–97] | 0.994 | 0.9% |
| Sentence version (push + margin) | 91.3% [86–95] | 0.986 | 0.6% |

False positives of the segment statistic (z > 3.118) on unwatermarked text:
- the two null sets: 0.92% of 1,200 Qwen texts and 1.0% of 600 Llama-3B texts
- the two clean Llama-3B draws on the test prompts: 3 of 150 (2.0%) and 2 of 150 (1.3%)

### 6.2 Robustness to edits

The same thresholds are applied to edited copies of every text:
- **Sentence reorder.** Shuffle the sentences and join them with single spaces. Texts with fewer than 4 sentences are
  skipped (n = 145–148).
- **Resegment.** Merge adjacent sentence pairs with ", and", and split long sentences at a comma. The words and their
  order are unchanged; only the boundaries move.
- **Light paraphrase.** gemma-4-31b-it, temperature 0.7: "Rewrite the text below with different wording while
  preserving its meaning, facts, names, numbers and approximate length. Keep the same structure and order of ideas."
- **Strong paraphrase.** Same model: "Substantially rewrite the text below: change the sentence structure and the
  vocabulary throughout, merge or split sentences freely, but preserve the same content, facts, names and numbers.
  Aim for 80-120% of the original length."

| Arm | Clean | Reorder | Resegment | Light paraphrase | Strong paraphrase |
|---|---|---|---|---|---|
| Shipped SemStamp | 68.0 | 12.3 | 9.6 | **27.3** | **7.3** |
| Segment chain | 83.3 | 4.1 | 23.8 | 2.0 | 1.3 |
| Segment + margin | 86.7 | 3.4 | 30.1 | 4.0 | 0.0 |
| Segment + push | 93.3 | 4.8 | 27.6 | 2.0 | 1.3 |
| Segment + push + margin | 94.0 | 8.1 | **39.9** | 10.7 | 3.3 |
| Sentence version (push + margin) | 91.3 | **20.4** | 19.0 | 26.0 | **7.3** |

All values are detection rates in %.

Why segments are fragile under paraphrase: the detector re-cuts the text at every 16th word. Any edit that adds or
removes a word shifts every later boundary, so each re-cut segment is a different piece of text with a new embedding.
And because the chain seeds each mask from the previous segment's bin, one changed bin also changes the next mask.
Resegmentation adds one word per merge, and the segment arms keep more under it (24–40%) than the sentence arms
(8–19%).

### 6.3 Cost and mechanics

| | Segment chain | + margin | + push | + push + margin | Sentence version (push + margin) |
|---|---|---|---|---|---|
| Candidates per unit, mean (median) | 8.06 (4) | 13.63 (16) | 6.90 (4) | 12.04 (16) | 12.59 (16) |
| Units that used all 16 candidates | 23.9% | 74.1% | 17.3% | 58.5% | 64.9% |
| Kept unit in its mask | 79.6% | 79.7% | 85.2% | 85.0% | 76.3% |
| Kept unit in its mask and clear of 0.02 | 16.4% | 30.3% | 26.5% | 46.7% | 39.2% |
| Drawn candidates in the mask | 16.5% | 20.9% | 24.0% | 29.0% | 26.9% |
| Drawn candidates in the mask and clear of 0.02 | 2.55% | 2.74% | 5.43% | 5.25% | 4.06% |
| Drawn candidates identical to another in the same unit | 8.7% | 8.2% | 11.4% | 11.9% | 29.4% |
| Seconds per text | 42.6 | 69.5 | 58.1 | 141.2 | 111.6 |

How to read the table:
- **"Drawn candidates" rows are shares of all candidates drawn.** Units that hit early contribute only 4 draws, while
  hard units contribute 16. So these shares are weighted toward hard units and are not the chance that a single draw
  hits. For comparison, 25.6% of transitions in clean text are in the mask.
- **Identical candidates** occur when the model writes exactly the same 16 words twice. A budget of 16 therefore does
  not always mean 16 distinct options.
- **Shipped SemStamp:** 60.2 candidates per accepted sentence, and 46.5% of sentences forced through at the cap.

### 6.4 Text quality

**Preference.** The judge sees the prompt and two answers: the watermarked text and the clean text for the same
prompt, each clipped to 800 words. It says which is better, or tie, and every pair is judged in both orders. Win rate
= P(watermarked better) + 0.5 · P(tie). A second clean draw scored against the first clean draw gets **0.425
[0.36–0.50]**. That is the no-effect level to compare against, not 0.5.

**Flaw measures.** The same judge rates each text on its own at temperature 0: whether it addresses the task, which
flaws it has, how severe they are, and a 1–5 score. For each measure, the **clean band** is the 95% range that a
single clean draw of these prompts produces, estimated from two clean draws.

| Arm | Win rate [95% CI] | Major flaw | Flaws per text | Not fully on task | Insertions or drift | Score 1–5 |
|---|---|---|---|---|---|---|
| Clean band | – | 0.127–0.253 | 0.413–0.587 | 0.047–0.140 | 0.000–0.013 | 3.48–3.87 |
| Shipped SemStamp | 0.485 [0.42–0.55] | 0.200 | 0.540 | 0.100 | 0.007 | 3.63 |
| Segment chain | 0.458 [0.39–0.53] | 0.187 | 0.493 | 0.073 | 0.007 | 3.66 |
| Segment + margin | 0.485 [0.42–0.55] | 0.200 | 0.500 | 0.100 | 0.007 | 3.63 |
| Segment + push | 0.400 [0.33–0.47] | 0.227 | 0.527 | 0.093 | 0.007 | 3.62 |
| Segment + push + margin | 0.402 [0.34–0.47] | 0.193 | 0.520 | 0.060 | 0.000 | 3.69 |
| Sentence version (push + margin) | 0.433 [0.37–0.50] | 0.207 | 0.520 | 0.093 | 0.007 | 3.71 |

Every value in this table is inside the clean band.

**Factual drift probe.** The judge does not check facts, so we added a probe on the 50 Dolly prompts: the share of
numbers and of capitalised name spans in a text that do not appear in the clean answer to the same prompt. Two clean
draws already differ here, so a second clean draw is the reference.

| Arm | Novel names | Novel numbers |
|---|---|---|
| Second clean draw (reference) | 0.475 [0.41–0.53] | 0.160 [0.11–0.22] |
| Segment chain | 0.470 | 0.180 |
| Segment + margin | 0.512 | 0.199 |
| Segment + push | 0.500 | 0.157 |
| Segment + push + margin | 0.496 | 0.197 |
| Shipped SemStamp | 0.497 | 0.183 |
| Sentence version (push + margin) | 0.537 | 0.175 |

### 6.5 What the push buys, in isolation

Compare the chain with and without the push, with no margin in either:
- clean detection: 83.3% → 93.3%
- per-drawn-candidate rate of "in the mask and clear of 0.02": 2.55% → 5.43%
- candidates per unit: 8.06 → 6.90
- time per text: 42.6 s → 58.1 s (the push's embedding calls cost more than the saved candidates)
- judge win rate: 0.458 → 0.400; all five flaw measures stay inside the band

### 6.6 A property of this key: bin 4 is in its own mask

| Previous bin | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| Mask | {2, 4} | {3, 7} | {3, 6} | {4, 6} | **{4, 1}** | {7, 0} | {2, 4} | {5, 4} |

- Bin 4 is in the mask of 5 of the 8 bins, including its own.
- A text that stays in bin 4 is therefore "in the mask" at every step. The first worked example we drew (a Dolly list
  answer) stayed in bin 4 for 10 of its 11 segments.

We measured how much of the detection depends on this, re-embedding every text with the detector's own segmentation:

| Arm | Segments in bin 4 | Transitions 4 → 4 | In-mask rate of all other transitions | Texts with ≥ 80% of segments in bin 4 |
|---|---|---|---|---|
| Clean (2 draws) | 11.9% / 11.2% | 2.3% / 2.1% | 23.8% / 21.4% | 0% / 0% |
| Segment chain | 25.6% | 12.5% | 72.2% | 1.3% |
| Segment + push | 28.1% | 14.8% | 79.3% | 3.3% |
| Segment + push + margin | 28.3% | 14.6% | 81.0% | 3.3% |

What this shows:
- Selection shifts segments toward bin 4, from about 12% to 28%.
- Detection does not rest on it: with every 4 → 4 transition removed, 81% of the remaining transitions are in the
  mask, against 21–24% for clean text.
- This mask table belongs to this key only, and every number in this report was measured with this one key. The
  wrong-key flag rates in 6.1 are 0.6–1.3%.

---

## 7. Where the segment version stands

- **What it does better than shipped SemStamp** on these prompts:
  - higher clean detection: 94.0% against 68.0%
  - 5× fewer candidates, and 2.3× less time, with push + margin
  - with the push alone: 8.7× fewer candidates and 5.5× less time, at 93.3% detection
  - no flaw measure outside the clean band
- **What it does worse:**
  - paraphrase: 10.7% against 27.3% light, 3.3% against 7.3% strong
  - sentence reordering: 8.1% against 12.3%
  - a lower judge win rate with the push: 0.40 against 0.485, where the no-effect level is 0.425 and the intervals
    overlap
- **Against our sentence version**, which uses the same chain, push and margin, the segment version wins only on
  resegmentation (39.9% against 19.0%) and on clean detection (94.0% against 91.3%, within the intervals). It loses
  on reordering and on both paraphrase levels. The sentence version is therefore our reference configuration.

**What these numbers do not cover**

- **Sample.** 150 prompts in 3 domains, one generator (Llama-3.2-3B), one key, texts up to 512 tokens.
- **Judges.** One LLM judge, which is the same model as the paraphraser. There is no human evaluation.
- **Timing.** Measured with 3 processes sharing an A100-40GB. It is not a single-stream latency.
- **Determinism.** Generation is not bit-identical across GPU nodes.
- **Post-hoc analyses.** The duplicate-candidate and bin-4 analyses (6.3, 6.6) were computed for this report and were
  not specified in advance.

---

## 8. Pre-registration

- Each round was specified in `docs/preregistration.md` before its results were read: amendments 5–10, covering the
  arms, the decision rules and the success criteria. Each outcome was recorded there afterwards.
- Round 3's success criterion was: detection ≥ 63%, at most 1/5 of SemStamp's candidates, and every flaw measure
  inside the clean band. It was met by the plain segment chain (amendment 7). The push arm improved on it under the
  amendment-8 rule, and the push + margin arm was preferred over the margin-only arm under the amendment-10 rule.

---

## 9. In progress: OpenStamp-style benchmark

The benchmark follows the datasets of the OpenStamp paper (arXiv 2608.27899): ArXiv (`ccdv/arxiv-summarization`),
BookSum (`kmfoda/booksum`) and Wikipedia (`wikimedia/wikipedia`, 20231101.en).

- **Prompts.** 500 test prompts per dataset, each a 50-token excerpt with an instruction to continue it. The
  watermarked continuations are 200 tokens.
- **Negatives.** Human-written continuations: 2,000 separate documents per dataset. They set per-dataset thresholds
  at 1% and 0.1% false-positive rate.
- **Arms.**
  - clean generation (two draws)
  - shipped SemStamp
  - the sentence reference
  - both segment arms (with the push, and with the push and margin)
  - RL-tuned variants, not covered here
- **Status.** Generation is running (segment + push: 477 of 1,500 texts at 13:49 on 25 September). No results yet.

---

## 10. Files and reproduction

Code:
- `scripts/steered_semstamp.py`: generation. Segment + push + margin:
  `--arm chain --unit seg --W 16 --block 4 --best-of 16 --steer marginal --beta 1 --stop-margin 0.02`.
  Drop `--steer`/`--beta` for no push; drop `--stop-margin` for no margin.
- `wmworld/semwm.py`: key, bins, masks, detection statistics (`score_text`), and the push (`NucleusSteerProcessor`).
- `scripts/semstamp_detect.py`: detection, thresholds, wrong keys.
- `scripts/reorder_attack.py`, `scripts/api_paraphrase.py`: attacks.
- `scripts/api_judge.py`, `scripts/api_pointwise.py`, `scripts/quality_vs_clean.py`: judge.
- `scripts/fact_probe.py`: factual drift probe.
- `scripts/seg_report_numbers.py`, `scripts/seg_report_bins.py`: the cost, mechanics and bin analyses of this report.
- Job scripts: `slurm/conf/ss_chain_{prod,push,margin,merged}.sbatch`, `ss_semstamp.sbatch`, `ss_clean.sbatch`,
  `ss_attacks.sbatch`, `ss_detect.sbatch`, `ss_judge.sbatch`.

Results:
- `runs/ss_llama3b/`: all texts, with per-unit records of every candidate's projections.
- `runs/detect_ss_llama3b.json`: detection.
- `runs/quality_vs_clean_ss_llama3b.json`, `runs/ss_llama3b/eval/api_pref_google_gemma-4-31b-it.json`: quality.
- `runs/fact_probe_ss_llama3b_r6.json`: factual drift.
- `runs/seg_report_numbers.json`, `runs/seg_report_bins.json`: this report's cost, mechanics, example and bin tables.

---

## Glossary

- **Unit:** the piece of text that gets one watermark decision. Here, a 16-word segment; in SemStamp, a sentence.
- **Bin:** which of the 8 regions (3 sign bits) a unit's embedding falls into.
- **Mask:** the 2 bins that count as "watermarked" for a unit, chosen from the key and the previous unit's bin.
- **Depth:** how far inside the mask a unit's embedding lies, in cosine. Positive means inside.
- **Margin:** the minimum distance from every hyperplane (0.02) that SemStamp requires, so small edits do not flip the
  bin.
- **Candidate:** one sampled continuation for a unit. The generator keeps one candidate per unit.
- **Push:** a bias of at most ±1 logit on the model's own likely tokens, toward the mask.
- **Detection rate / TPR:** the share of watermarked texts above the 1%-false-positive threshold.
- **AUROC:** the probability that a random watermarked text scores above a random unwatermarked one.
- **Win rate:** how often the judge prefers the watermarked text over the clean text, with ties counted as one half.
  The no-effect level here is 0.425.
