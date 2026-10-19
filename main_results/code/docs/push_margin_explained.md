# Push + Margin: a semantic text watermark guided by model uncertainty

This document explains one method for watermarking text written by a language model. It is self-contained: every term
is defined before it is used, every parameter is given with its value, and every number reported is a measured result
from our experiments (no estimates). Read the sections in order.

---

## 1. Summary in five sentences

1. A **watermark** is a hidden, key-dependent pattern in generated text that a **detector** holding the same key can
   recognise later; people reading the text should not notice it.
2. **SemStamp** (Hou et al., NAACL 2024) puts the pattern in the *meaning* of each sentence: it rejects sampled sentences
   until one lands in a key-selected region of a sentence-embedding space.
3. **Push + Margin** keeps SemStamp's key and detector unchanged and changes only how text is generated: it draws a
   bounded number of candidate sentences, keeps the best one, and gently steers word choice while each candidate is written.
4. The central idea is **uncertainty**: a sentence can only be steered into a region if the model has several plausible
   options that differ in meaning, so the method acts only where the model is uncertain and stops trying where it is not.
5. Measured result: on chat models, Push + Margin detects far more watermarked texts than SemStamp at the same false-alarm
   rate, draws about 4.5 times fewer candidate sentences, and the text is judged about as good as unwatermarked text.

---

## 2. Background: how SemStamp works

### 2.1 Turning a sentence into a region

1. **Sentence encoder.** Every sentence is converted to a vector of 768 numbers by a fine-tuned sentence encoder
   (`AbeHou/SemStamp-c4-sbert`). Sentences with similar meaning get nearby vectors.
2. **Key = 3 hyperplanes.** The secret key fixes 3 random hyperplanes through the origin of that 768-dimensional space
   (generated from random seed 1234).
3. **Projection.** For each hyperplane, compute the signed distance of the sentence vector from the plane. This gives 3
   numbers, called the sentence's **projections**.
4. **Region (also called bin).** The signs of the 3 projections form a 3-bit code, which names one of 8 regions
   (2³ = 8). Example: signs (−, −, +) → code 001 → region 1.

### 2.2 The mask

1. For every sentence after the first, the key and the region of the **previous** sentence select 2 of the 8 regions.
   These 2 regions are the **mask** for the current sentence: landing in the mask means "watermarked".
2. The selection is pseudo-random: it is seeded by a fixed hash key (15485863) multiplied by the previous region's number.
   Example for our key: after a sentence in region 4, the mask is {4, 1}; after region 3, the mask is {4, 6}.
3. Fraction of regions in the mask: γ = 2/8 = 0.25. Unwatermarked text therefore lands in its mask about 25% of the time
   by chance.

### 2.3 The margin

A sentence whose projection is very close to a hyperplane could flip to another region after a tiny edit. The **margin**
requires every projection to be at least 0.02 away from zero (|projection| ≥ 0.02 on all 3 planes).

### 2.4 How SemStamp generates

For each sentence: sample one complete sentence from the model, compute its region, and **reject** it unless it is inside
the mask **and** clears the margin. Repeat. After 100 rejected tries, SemStamp **gives up** and keeps an unchecked
sentence; that sentence carries no watermark.

### 2.5 How detection works (identical for SemStamp and Push + Margin)

1. Split the text into sentences.
2. Embed each sentence and compute its region.
3. For every sentence from the second onward, compute its mask from the previous sentence's region and check whether the
   sentence is in its mask. (The first sentence has no previous sentence in the text, so it is not scored.)
4. Let n = number of scored sentences and k = number of them inside their mask. Compute the z-score
   `z = (k − γ·n) / sqrt(γ·(1 − γ)·n)` with γ = 0.25.
5. Flag the text as watermarked if z exceeds a threshold. The threshold is set on human-written texts so that only 1%
   (or, in the stricter setting, 0.1%) of human texts are flagged. "TPR at 1% FPR" = the share of watermarked texts
   flagged at that threshold.

The detector needs only the text and the key. It does not need the language model or the prompt.

---

## 3. The uncertainty view (why the method is designed this way)

### 3.1 Three kinds of uncertainty

- **Token uncertainty (lexical).** How unsure the model is about its next token. Measured by the **entropy** of the
  next-token distribution, in nats: near 0 when one token is almost certain, larger when many tokens are plausible.
- **Sentence uncertainty (semantic).** How much the model's plausible *next sentences* differ in meaning. We measure it
  by drawing 16 candidate next sentences and computing the entropy of their distribution over the 8 regions, called the
  **bin entropy**: 0 when all 16 land in the same region, at most ln 8 ≈ 2.08 when they are spread evenly.
- **Model-level uncertainty.** Chat (instruction-tuned) models are much more confident about what comes next than base
  models, so they have lower semantic uncertainty overall.

### 3.2 Watermark capacity follows semantic uncertainty

Measured on 24,209 sentence positions of Llama-3.2-3B-Instruct, 16 real candidate sentences per position:

| Bin entropy of the 16 candidates | Positions with ≥ 1 usable candidate, plain sampling | Same, with the push |
|---|---|---|
| 0 (all 16 in one region) | 3.3% | 6.5% |
| 0 – 0.5 | 20.7% | 33.3% |
| 0.5 – 1.0 | 25.6% | 38.8% |
| 1.0 – 1.5 | 34.6% | 49.2% |
| 1.5 – 2.08 | 39.2% | 51.1% |

("Usable" = inside the mask and clear of the margin.) At 17.5% of positions all 16 candidates fall in one region, and 87%
of those are word-for-word duplicates: the model has essentially one thing to say. At such positions no amount of
resampling helps, which is exactly where SemStamp hits its 100-try limit.

### 3.3 Design consequences

1. **Do not keep resampling where the model is certain.** Cap the number of candidates per sentence and accept that some
   sentences carry no mark.
2. **Steer only where the model is uncertain.** At the token level, push word choices only when next-token entropy is
   high, so that the push changes meaning where the model itself was undecided and leaves fixed phrases alone.
3. **Expect the method to matter most for confident (chat) models.** Measured: SemStamp gives up on 28–32% of sentences
   with Llama-3B chat, on 10.4% with Llama-3B base.

---

## 4. The method: Push + Margin

Push + Margin changes three things relative to SemStamp. The encoder, key, mask rule and detector are unchanged.

### 4.1 Part 1 — bounded selection instead of rejection

For each sentence after the first:

1. Compute the mask from the previous sentence's region.
2. Draw candidate sentences **4 at a time** (one block), never more than **16** in total. Each candidate is written with
   the push active (Part 3).
3. After each block, embed the candidates and compute each one's region and **depth** (Section 4.4).
4. If the block contains at least one **usable** candidate (inside the mask and clear of the 0.02 margin): stop, and keep
   the usable candidate with the **largest depth**.
5. If 16 candidates have been drawn and none is usable: keep the candidate **closest to the mask** (largest depth, even
   though negative). That sentence carries no mark; the text continues.

The first sentence is a single plain sample with no push; it only seeds the chain (it sets the mask for sentence 2).
Every kept sentence is one of the model's own samples.

### 4.2 Part 2 — the margin as a stopping rule only

SemStamp requires every kept sentence to clear the margin. Push + Margin uses the margin only to decide when to **stop
drawing**: a block ends the search only if it holds a candidate that is both in the mask and clear of the margin.

### 4.3 Part 3 — the push (token-level steering)

At every token of every candidate sentence (except the first sentence of the text), before sampling:

1. **Entropy gate.** Compute the entropy H (nats) of the model's next-token distribution. The push strength is
   `g(H) = 0` if H ≤ 0.2; `g(H) = (H − 0.2) / 1.8` if 0.2 < H < 2.0; `g(H) = 1` if H ≥ 2.0. If g = 0, nothing changes.
2. **Shortlist.** Take the model's own plausible set: the top-p set (p = 0.95) of the distribution at temperature 0.8.
   Tokens outside it stay impossible, so the push never introduces a token the model would not have produced.
3. **Candidates for the bonus.** Take the 20 most likely **word** tokens of the shortlist. Punctuation, whitespace,
   newlines and end-of-text tokens (the **boundary** tokens) are never pushed.
4. **Score.** For each of the 20 tokens, form the text "sentence written so far + this token", embed it, and compute its
   depth into the mask (Section 4.4), capped at 0.05.
5. **Bonus.** Standardise the 20 scores (subtract their probability-weighted mean, divide by their probability-weighted
   standard deviation), clip to the range [−1, +1], and multiply by β = 1 and by g(H). Add the result to those tokens'
   logits. The largest possible change is ±1 logit, i.e. a token's odds change by at most a factor of about 2.7.
6. **Repetition guard.** A token that would repeat a 4-token sequence already present in the text gets no positive bonus.
7. **Mass preservation.** Shift the word tokens' logits by one constant so that the total probability of the boundary
   tokens is exactly the model's own. Consequence: the push never changes where sentences or the text end.
8. **Stop when deep enough.** Once the sentence so far is 0.05 deep inside the mask, all 20 scores are equal (they are
   capped at 0.05), so the bonus is zero.
9. **Sample** the next token from the adjusted distribution as usual. The push changes probabilities; it never picks a
   token deterministically.

Measured size of the push: averaged over 24,425 sentences it moves the model's next-token distributions by 1.02 nats per
sentence (KL divergence). Cost: one batched encoder call of 20 short strings per pushed token.

### 4.4 Depth

For one region, the depth of a sentence is the smallest of its 3 signed projections, with each sign chosen so that
"positive" means "on the region's side of that plane". Positive depth = inside the region; the value is the distance to
the nearest boundary of the region. For a mask of 2 regions, depth = the larger of the two regions' depths.

### 4.5 Where a sentence or a text ends (generation rules)

A sentence unit ends at sentence punctuation (`.`, `!`, `?`, optionally followed by closing quotes or brackets) followed by
whitespace, or at a newline. Three rules make units match what the detector sees:

1. **Leading line breaks join the next sentence.** Some tokenizers (OPT, Qwen) emit a line break as a separate token.
   Without this rule a line break became a one-token "sentence" that the detector never sees, which shifted the mask
   chain. Measured on OPT: generator and detector sentences matched in 2.9% of texts without the rule and in 94.5–96.9%
   with it.
2. **The model may end its answer after any sentence.** Before each new sentence, the model makes one ordinary draw of its
   next token; if that draw is "end of answer", the text ends. Inside a unit, end of answer is blocked only on the first
   token. (Needed for code and for answers that end with a short line.)
3. **A first sentence is never empty.** If the model ends the text before its first sentence, it is asked again, up to 8
   times.

### 4.6 Pseudocode

```
generate(prompt, key):
    text = plain_sample_one_sentence(prompt)            # first sentence, no push
    prev_region = region(text)
    while not finished(text):
        if model_draws_end_of_answer(prompt, text):     # rule 4.5.2
            break
        mask = key.mask(prev_region)                    # 2 of 8 regions
        candidates = []
        while len(candidates) < 16:
            block = [write_sentence_with_push(prompt, text, mask) for _ in range(4)]
            candidates += block
            if any(usable(c, mask) for c in block):     # in mask and margin >= 0.02
                break
        usable_ones = [c for c in candidates if usable(c, mask)]
        chosen = max(usable_ones or candidates, key=lambda c: depth(c, mask))
        text += chosen
        prev_region = region(chosen)
    return text

write_sentence_with_push(prompt, text, mask):
    sentence = ""
    repeat until a sentence boundary or end of answer:
        logits = model(prompt + text + sentence)
        g = entropy_gate(entropy(logits))               # 0 below 0.2 nats, 1 above 2.0
        if g > 0 and depth(sentence, mask) < 0.05:
            shortlist = top_p(logits, p=0.95, temperature=0.8)
            words = 20 most likely non-boundary tokens in shortlist
            scores = [min(0.05, depth(sentence + w, mask)) for w in words]
            bonus = clip(standardise(scores), -1, 1) * g
            logits[words] += bonus (minus repetition, then shift words so P(boundary) is unchanged)
        sentence += sample(logits, temperature=0.8, top_p=0.95)
    return sentence
```

---

## 5. Why it detects more than SemStamp (measured decomposition)

On the main benchmark (Llama-3B chat; ArXiv / BookSum / Wikipedia), split the detector's z-score into its two inputs:

| | SemStamp | Push + Margin |
|---|---|---|
| Sentences kept unmarked after giving up | 31.5% / 28.3% / 32.0% | (no give-up rule; capped at 16) |
| Scored sentences inside their mask, at detection | 71.4% / 73.4% / 70.6% | 92.2% / 90.3% / 91.5% |
| Sentences the detector scores per text | 4.7 / 7.3 / 3.8 | 4.6 / 7.0 / 3.9 |

Both methods give the detector the same number of sentences, so the entire difference is the share of sentences in their
mask, and SemStamp loses it to the sentences it gives up on. Re-reading the text loses almost nothing: for Push + Margin,
92.5% of sentences are in their mask when generated and 92.2% when the detector re-splits and re-embeds the text.

---

## 6. Measured results (TPR at 1% FPR unless stated; three numbers = three datasets in the order named)

### 6.1 Chat models

| Model, benchmark | Push + Margin | SemStamp |
|---|---|---|
| Llama-3.2-3B chat, ArXiv / BookSum / Wikipedia | 77.5 / 91.6 / 57.4 | 44.3 / 66.5 / 31.9 |
| Llama-3.2-3B chat, RealNews / BookSum / Reddit-TIFU | 79.8 / 90.8 / 81.2 | 59.6 / 67.6 / 62.1 |
| Llama-3.2-1B chat, ArXiv / BookSum / Wikipedia | 85.4 / 96.0 / 61.7 | 55.9 / 74.2 / 38.2 |
| Llama-3.2-1B chat, RealNews / BookSum / Reddit-TIFU | 82.3 / 96.3 / 70.9 | 65.7 / 74.3 / 48.4 |
| Llama-3.2-3B chat, Dolly instructions (50 texts) | 78 | 30 |

At the stricter 0.1% false-alarm threshold, Push + Margin detects 1.5–2.5 times as many texts as SemStamp on the Llama chat models.

### 6.2 Base models and SemStamp's own setting

| Model, benchmark | Push + Margin | SemStamp |
|---|---|---|
| Llama-3.2-3B base, ArXiv / BookSum / Wikipedia | 84.1 / 99.8 / 91.6 | 67.7 / 99.2 / 74.0 |
| Llama-3.2-3B base, RealNews / BookSum / Reddit-TIFU | 98.2 / 99.5 / 97.0 | 87.7 / 98.2 / 94.9 |
| Llama-3.2-1B base, ArXiv / BookSum / Wikipedia | 85.9 / 99.8 / 93.0 | 70.7 / 99.6 / 77.6 |
| Llama-3.2-1B base, RealNews / BookSum / Reddit-TIFU | 97.5 / 99.0 / 97.6 | 89.4 / 97.7 / 95.4 |
| OPT-1.3B (SemStamp's own model and setting), RealNews, 1% / 0.1% FPR | 99.1 / 98.5 | 97.7 / 93.6 (their paper: 93.9 at 1%) |

With base models both methods detect much more, consistent with Section 3: base models are more uncertain, so SemStamp
gives up less often.

### 6.3 Cost, quality, robustness

- **Cost (Llama-3B chat):** 10.8 candidate sentences per kept sentence against 49.0 for SemStamp; 40.5 s per text against
  113.6 s (6.0 s without a watermark).
- **Quality:** an independent judge model (Gemma-4-31B) compares each text with an unwatermarked text for the same prompt.
  On chat models Push + Margin scores about the same as SemStamp and close to the "no difference" level. On base models
  Push + Margin is clearly closer to unwatermarked quality than SemStamp.
- **Paraphrase:** after an LLM paraphrases the text, both methods lose most of the signal (light paraphrase leaves 15–40% detected on
  chat models and 15–70% on base models, depending on the dataset), and the two methods stay level with each other.

---

## 7. Limitations (measured)

- **Paraphrase** remains the main weakness of every meaning-based watermark tested, including this one.
- **Code:** on programming problems (MBPP), the push lowers correctness (pass@1) by 9–15 points, while SemStamp keeps
  correctness but barely marks code. Code has very low semantic freedom.
- **Qwen3.5-0.8B results** were affected by the line-break problem (rule 4.5.1); corrected runs are in progress.
- **One key, one judge model, no human evaluation** so far.
- **Learned alternatives** (training LoRA adapters with reinforcement learning, distilling the push into the model) did not
  beat the push and cost text quality.

---

## 8. Glossary

| Term | Meaning |
|---|---|
| Watermark | A hidden, key-dependent pattern in generated text, recognisable by a detector with the key |
| Encoder | The model that turns a sentence into a 768-number vector |
| Hyperplane | One of the 3 random flat cuts through the vector space that form the key |
| Projection | Signed distance of a sentence vector from one hyperplane |
| Region (bin) | One of the 8 parts of the space cut out by the 3 hyperplanes |
| Mask | The 2 regions that count as watermarked for a sentence, chosen from the key and the previous sentence's region |
| Margin | Requirement that all projections are at least 0.02 from zero |
| Depth | How far inside the mask a sentence is (positive = inside); see Section 4.4 |
| Usable candidate | A candidate sentence inside the mask and clear of the margin |
| Push | Token-level bonus (at most ±1 logit) toward the mask, applied only to the model's own plausible word tokens |
| Entropy gate | Push strength as a function of next-token entropy: 0 at ≤ 0.2 nats, 1 at ≥ 2.0 nats |
| Bin entropy | Entropy of 16 candidate sentences' distribution over the 8 regions; a measure of semantic uncertainty |
| Give up | SemStamp keeping an unchecked sentence after 100 rejected tries |
| z-score | Detector statistic: how far the number of in-mask sentences exceeds the 25% expected by chance |
| TPR at 1% FPR | Share of watermarked texts flagged when the threshold flags 1% of human texts |
| Chat vs base model | A chat model is tuned to follow instructions; a base model only continues text |
