# Pre-registered rules for the confirmation round

Written 19 September 2026, **before** any result of the trade-off sweep (`runs/tradeoff`), the long-generation
comparison (`runs/long768`), the detector-variant analysis or the full judge pass existed. Everything selected on
development data is selected by the rules below; the confirmation round is then run once and reported as is.

> **Amendment, 19 September 2026, committed 23:05:47 (+04)** as `a4dfffe` -- before the detector-variant analysis
> ran: job `wm_dumps` (265887) started at 23:06:40 by its own log, and its analysis step runs only after ~20 minutes of dumps. Reading the per-token traces showed that in the plain statistic the top 5%
> of tokens carry ~91% of a text's evidence (response openers and key-flavoured words). Four clipped-evidence
> candidates were therefore added to the detector list: `T_clip05`, `T_clip1`, `T_clip05_gate`, `T_sign`; and two
> generation arms with a semantic guard (no push on tokens with raw contrast above 1.0) were added to the sweep:
> `wm_c12wg`, `wm_c20wg`. The selection rules are unchanged.

> **Amendment 2, 19 September 2026, ~23:50 — made AFTER seeing the detector comparison on the development runs
> (`runs/detector_variants_dev.json`) and BEFORE any sweep or confirmation text existed** (the sweep jobs 266276/266277
> were still pending). What the data showed: clean false positives are clustered in a few texts (in `cross_long`, 22
> flagged text-key pairs come from 4 of 60 clean texts), so the per-run criterion "at most 2% on each run's clean arm"
> mostly measures which 60 texts a run happens to contain: the same plain statistic gives 1/720 on one run and 22/720 on
> another; pooled over the 180 clean texts every candidate sits between 1.0% and 1.6%.
> *Outcome under the original text:* plain `T` itself fails the per-run criterion (3.1% on `cross_long`), the best
> eligible variant (`matched_gate_hi`, mean 0.93) does not beat `T` (0.91) by 5 points, so **`T` stays**.
> *Amended rule:* eligibility is judged on the POOLED clean arms (2,160 text-key pairs, at most 2%). Under it
> **`T_clip05_gate` replaces `T`** (mean true-positive rate 0.974 vs 0.910, pooled clean rate 1.5% vs 1.4%).
> Both statistics are reported for every result from here on; `T` is labelled "original primary", `T_clip05_gate`
> "amended primary". Configuration selection uses `T_clip05_gate`, because the fixes under test deliberately move the
> evidence away from the few large-contrast tokens the unclipped statistic depends on. Readers who distrust a
> post-hoc amendment can read every table with `T`.

> **Amendment 3, 20 September 2026, ~00:50 — made after an EARLY LOOK at the sweep** (25% generated; blind judge on 15–19
> prompts per arm, generator-side signal estimates). What it showed: judged quality tracks total push strength (constant
> push 8–20: win rate 0.06–0.15; old controller 0.20), only constant push 4 clears the quality bar (0.45), and the
> semantic guard roughly doubles the win rate at equal strength. The grid had no arm between 4 and 8 and no guard
> below 12, i.e. nothing where the frontier is. Four arms were therefore added to the sweep: `wm_c4g`, `wm_c6`,
> `wm_c6g`, `wm_c8g` (constant push 4/6/8, `g` = semantic guard, no warm-up). The selection rule is unchanged and is
> applied to all 15 arms once the full sweep (60 prompts per arm) is judged; the confirmation round remains untouched.

## What is being chosen

1. **One detector.** Candidates: the text-only statistics in `wmworld/posstats.py` (`T`, `T_gate`, `T_gate_hi`,
   `T_centered`, `T_centered_gate`, `matched`, `matched_gate`, `matched_gate_hi`, `z_gate`).
   Rule: highest mean true-positive rate at the frozen 1% threshold over the marked arms of the finished development
   runs (`cross_llama`, `cross_long`, `cross_long2`), among variants whose false-positive rate on those runs' clean
   arms is at most 2%. A variant must beat plain `T` by at least 5 points to replace it; otherwise `T` stays.
2. **One generation configuration.** Candidates: the arms of `ablation_tradeoff` (incumbent adaptive controller;
   constant push 4/8/12/16/20; constant push 8/16 with a 48-token warm-up).
   Rule: highest true-positive rate at the frozen 1% threshold (chosen detector) among arms that the blind judge does
   not find clearly worse than clean text: win rate against the clean response >= 0.40 **and** increase in major-flaw
   rate over the clean response <= 10 points. If no arm qualifies, the arm with the highest win rate among those with a
   true-positive rate >= 0.50 is taken and the failure to meet the quality bar is reported as the headline.
   The incumbent adaptive controller (`wm_cross30`) is carried into confirmation as the "before" comparison regardless.

## The confirmation round (run once)

* Prompts: 150 unseen prompts from the `positive_test` partition (50 per domain, fixed hash order).
* Keys: the 16 held-out confirmation seed phrases, balanced assignment; none was used in any development run.
* Threshold: per key, from the 1,200 clean texts of `null_calibration` scored under the confirmation keys; frozen
  before the confirmation texts exist. False positives are measured on the confirmation run's own clean arm, which
  shares nothing with the null partition.
* Reported: true-positive rate with a Wilson interval, false-positive rate, key-standardised AUROC, wrong-key rate,
  length-matched prefix views (128/256/512 reference tokens), all per domain as well as pooled.
* Quality: blind pairwise judge through an external API model (`google/gemma-4-31b-it`; both orders; never sees key
  or arm): win rate, on-task rate, same-question rate, flaw rates, paired with the clean response to the same prompt.
* Robustness: light and strong paraphrase of marked AND clean texts by an external API model, then the unchanged
  detector and threshold. Attacked clean texts give the attacked false-positive rate.
* Instruction following: 200 IFEval prompts (fixed hash order), official evaluator, clean vs chosen configuration;
  detection on those responses with the same frozen thresholds (a deliberate domain shift).

## Not claimed

Human judgement (none collected), robustness to paraphrasers other than the one tested, cryptographic security of
seed-phrase keys, false-positive rates below what 1,200 clean texts can resolve.

## Amendment 4 (2026-09-21, written before the run): held-out confirmation of the bounded reference context

Arm `wm_w64_c3`: constant push 3, reference model sees the key plus the last 64-95 reference tokens (blocks of 32); detector
`T_clip05_gate` computed under the identical window rule, per-key 1% thresholds from the 1,200-text clean null re-scored through
the window under the 16 confirmation keys.  Prompts, keys and clean reference texts are those of `conf_cross`; one arm, run once.
Pass criteria, fixed in advance: (1) detection >= 75% at the frozen thresholds; (2) share of texts with insertions or drift and
share not fully on task both inside the held-out clean band (clean draws `conf_cross:base`, `conf_pos:base`).  The dev result
(80% detected, all five flaw measures inside the clean band, 60 prompts) is exploratory; the earlier dev-to-held-out
non-replication is the reason for this round.  The dev criterion "within 5 points of whole-text push 4" was missed (80% vs 90%).

Deviation noted 2026-09-21 16:50, before any held-out result of the windowed arm was read except the preference on 40 overlapping
prompts (0.48): the stage `conf_window` hashed its prompt subset with its own stage name and therefore drew a DIFFERENT 150 of the
600 held-out prompts than `conf_cross` (40 shared).  The registered design (same prompts, keys and clean texts as `conf_cross`) is
therefore run as `runs/conf_window_reg` (stage `conf_cross`, arm `wm_w64_c3`); it is the confirmatory result.  `runs/conf_window`
is kept and reported as an additional held-out sample for detection only (no clean texts exist for 110 of its prompts).

## Amendment 5 (2026-09-23, written before any result of these arms was read): semantic watermark on Llama-3.2-3B

Arms on the 150 held-out prompts of `conf_cross` (partition `positive_test`), generator Llama-3.2-3B-Instruct bf16
(rev 0cb88a4f), drafter Llama-3.2-1B-Instruct (rev 9213176), one shared key = SemStamp's shipped constants
(hyperplanes `RandomState(1234)`, mask seed `15485863 * prev_bin`), embedder AbeHou/SemStamp-c4-sbert unless the
embedder study (`runs/embedder_study.json`, zero generation) picks another; `beta` fixed from the smoke premise check
(`runs/ss_smoke/premise_check.json`) at the smallest value with delta >= 1 null-sd or acceptance >= 25%.
Stop rule (smoke): every beta gives delta < 0.5 null-sd AND r(pred, realised) < 0.1 AND best acceptance < 15% -> stop.
Success criteria, fixed now:
- steered_sent vs semstamp: candidates per accepted sentence <= 3 (shipped on Qwen: 24.5); detection at 1% FPR within
  5 points; preference and all five flaw measures not worse (paired, same prompts).
- steered_seg: detection at 1% FPR >= semstamp's; statistic bit-identical under reorder_sent / resegment; paraphrase
  retention (AUROC - 0.5 ratio) >= semstamp's.
- drafted twins (base_draft, steered_seg_draft): TPR and judge within CI of the non-drafted arm; *_draft rejection
  arms: 3B forwards per accepted sentence <= 1/3 of semstamp's, left_region_after_verify < 20%, judge not worse.
- wrong-key (alt seeds) flag rate within the Wilson CI of 1%; 3B clean cohort FPR <= 1.5% under Qwen-null thresholds.
Detection thresholds: per statistic, conservative 1% upper tail of the 1,200 Qwen-clean null texts; the 600 Llama-3B
clean null texts are an independent FPR cohort.  Judge: preference (both orders) and pointwise flaws vs the clean band
from two clean 3B draws (base seed 0, clean_seed1).

### Amendment 5 — outcome (2026-09-24, read after all arms, attacks and judge finished; runs/detect_ss_llama3b.json, runs/quality_vs_clean_ss_llama3b.json, runs/ss_llama3b/summary.txt)
Detection at the frozen 1% threshold (1,800 null texts), own statistic, then reorder / resegment / light / strong paraphrase; judge preference vs clean (clean-vs-clean 0.425) and flaw measures outside the clean band (of 5):
- semstamp (shipped, 3B): 68% | 12 / 10 / 27 / 7% | pref 0.485, 0/5 outside. 320 s/text, 60 candidates per accepted sentence, 47% forced through.
- steered_sent beta=4: 88% (continuous statistic; 74% chain z) | 19 / 28 / 43 / 17% | pref 0.040, 5/5 outside (major flaw 87% vs 20%). 169 s/text, 35 candidates, 28% forced through.
- steered_seg beta=4: 46% | 32 / 36 / 15 / 14% | pref 0.102, 5/5 outside. 17 s/text.  beta=2: 10% | 6 / 8 / 3 / 3% | pref 0.387, 0/5 outside.
- steered_seg + 1B assist: 39% (within CI of the twin), pref 0.058, 29 s/text (slower).
- Wrong-key flag rate <= 3% for every arm; 3B clean cohorts: t_seg 0.3%, t_cont 1.7%, chain z 2.3% pooled (chain z is discrete and slightly anti-conservative here).
Criteria: steered_sent <= 3 candidates FAILED (35); detection within 5 pts PASSED (88 vs 68); quality FAILED. steered_seg detection >= semstamp FAILED (46 vs 68); invariance under reorder/resegment holds only up to the split (32-36% retained vs 12% for the chain); paraphrase retention better than the chain's. Drafting twins: same distribution PASSED, no speedup. FPR bar (<= 1.5% pooled) passed for the segment and continuous statistics, failed for chain z.
Deviation: the beta rule's clause "whose text still reads normally" was not checked before the run (the premise check does not save texts); at beta=4 the bias reaches +/-8 logits (beta x 2 sd clip) and the judge found the texts degenerate. beta=2 is quality-neutral but weak. No beta was found where steering both helps and stays clean.

### Outcome of amendment 5 (read 2026-09-23 21:10, runs/detect_ss_llama3b.json, runs/quality_vs_clean_ss_llama3b.json)
Detection at 1% FPR (1,800 clean null texts; own statistic): semstamp 68% [60-75]; steered_sent (beta 4) 74% [67-81] by
the chain z, 88% [82-92] by the continuous variant; steered_seg (beta 4) 46% [38-54]; steered_seg (beta 2) 10% [6-16];
steered_seg+assist 39% [31-47]. Clean cohorts 0.0-3.3%, 600-text 3B null 0.3-0.8%. Wrong-key 0.2-2.8%.
Cost: semstamp 60.2 candidates/accepted sentence, 47% forced through, 320 s/text; steered_sent 35.2, 28%, 169 s/text
(criterion "<= 3" FAILED); steered_seg 17 s/text (no rejection); assist 29 s/text (slower, as in the smoke).
Attacks (TPR at the same thresholds): semstamp reorder 12% / resegment 10% / para-light 27% / para-strong 7%;
steered_sent 19 / 15 / 39 / 15; steered_seg beta 4: 32 / 36 / 15 / 14. The pre-registered claim that the segment statistic
is bit-identical under sentence reordering was WRONG as stated: it is invariant to permutation of its own W-word units,
but reordering sentences re-cuts those units (70-78% of detections retained vs 15-18% for the chained statistic).
Quality (judge, 150 prompts, 3B clean-vs-clean preference 0.425 [0.36-0.50]): semstamp 0.485 [0.42-0.55], all five flaw
measures inside the clean band (quality-neutral, as its paper claims); steered_sent beta 4 preference 0.04, score 1.46
vs 3.68, major flaws 87% vs 19% -- destroyed; steered_seg beta 4 0.10, score 1.82 -- destroyed; steered_seg beta 2
0.39 [0.32-0.46], all five measures inside the band, score 3.53 (-0.15 [-0.33, +0.03]) -- quality-neutral.
Verdict: the steering premise held (push moves the embedding as predicted) but at the strength needed for detection it
costs the distortion-free property that makes SemStamp quality-neutral; criteria "quality not worse" FAILED for both
beta-4 arms. The premise check's condition "text still reads normally" was not verified before choosing beta (no texts
were saved by the check) -- a procedural error. Detection-only gain: the continuous chain statistic lifts steered_sent
from 74% to 88% at no quality cost, but does not help shipped SemStamp (68% -> 65%).

## Amendment 6 (2026-09-23, written before any production result of these arms was read): round 2 of the segment arm

Why: round 1 failed on quality through mechanics (peak bias +/-8 logits before temperature, tail tokens pulled into
top-p, no repetition guard, boundary tokens relatively suppressed).  New processor `semwm.NucleusSteerProcessor`: push
only inside the unbiased top-p nucleus (bias on the top-20 lexical members, tail masked), gains standardised and
clamped to 1 sd (peak |bias| <= beta), repetition guard (no positive bias completing a repeated 4-gram), mass-preserving
tilt (P(EOS/punctuation/whitespace) unchanged at every step; verified in the smoke to 1e-8), push stops when the unit
is 0.05 deep (the detector clip).  Smoke (runs/ss_smoke_r2, 6 prompts x 200 tokens, texts read): no loops, no tail junk,
sentence length at base level.
Arms (runs/ss_llama3b, same 150 held-out prompts, same key, W=16, SemStamp sbert):
  segA_b2 / segA_b3  token-table gain, beta 2 / 3
  segB_b2            exact marginal gain (unit re-embedded with each top-20 candidate), beta 2
  bestof4_seg        best-of-4 exact samples per 16-word chunk, no bias (arm C; job 275199)
  bestof4B_b1        best-of-4 with the marginal push at beta 1 (arm E)
Primary statistic: t_seg (unchanged).  Step-0 rule (runs/ss_power.json): t_dir beats t_seg under reorder/resegment
(+8-12 pts) but loses under strong paraphrase (steered_seg 9% vs 14%) -> rule not met, t_seg stays primary; t_seg2
(two-bin, idea D) is worse everywhere -> dropped.
Power model (runs/ss_power.json; null units within-text correlation rho = 0.10, null variance 4.1x the independent
value; model reproduces round 1: beta 4 pred 0.37 obs 0.46, beta 2 pred 0.09 obs 0.10).  PREDICTIONS (t_seg, clean):
segA_b2 10-20%, segA_b3 15-30%, segB_b2 10-25%, bestof4_seg 10-20%, bestof4B_b1 20-35%.  No arm is expected above 40%;
reaching ~67% needs a mean unit shift of ~1.5 null unit-sd.
Selection rule unchanged: highest detection among arms with every flaw measure inside the clean band.  Extra,
non-selecting measure: factual probe (scripts/fact_probe.py; round 1: novel entities 0.57-0.62 for steered arms vs
0.47 [0.41-0.53] clean-vs-clean; novel numbers inside the band).

## Amendment 7 (2026-09-24, written before any production text of these arms exists): bounded SemStamp chain

Goal (the user's): cheaper than shipped SemStamp (60.2 candidates per accepted sentence, 320 s/text) and nearly as
detectable (68% at 1% FPR, chain z).  Method (scripts/steered_semstamp.py --arm chain): SemStamp's chain unchanged
(mask of unit i seeded by the bin of unit i-1, same key, same encoder, lambda 0.25), but NO margin and NO retry loop:
the 3B draws candidates in batched blocks of 4, stops at the first block that holds an in-mask candidate, keeps the
deepest drawn candidate; cap 16 candidates per unit.  Every candidate is an exact 3B sample (selection only).
Units: 16-word segments (primary, detected by z_segchain over split_segments) and sentences (z_chain).
Evidence before production (runs/ss_chain_smoke, 30 prompts; runs/ss_budget.json): 3B best-of-16 = 80% (segments) /
77% (sentences) at 35-37 s/text; replay of stop-at-first-in-mask capped at 16 = 88% (seg) / 80% (sent) at 6.4 / 7.0
candidates per unit; SemStamp's margin rule at the same cap = 13-15%.  1B drafting (v1, v2) rejected: 20-41% at
compute no lower than the 3B chain (runs/ss_budget_v2.json).
Arms (runs/ss_llama3b, 150 held-out prompts): chain_seg_blk4x16, chain_sent_blk4x16.
PREDICTIONS (clean, own statistic, frozen null thresholds): segments 75-90%, sentences 70-85%; 5-8 candidates per unit;
20-30 s/text; quality inside the clean band on all five flaw measures (exact samples; weaker selection than shipped
SemStamp, which was quality-neutral).  Attacks: reorder/resegment/paraphrase expected no better than shipped SemStamp
(chain statistics; not the goal of this amendment).
Success criterion: detection >= 63% (within 5 pts of shipped) at <= 1/5 of SemStamp's candidates per unit, every flaw
measure inside the clean band.  Also reported: fact probe (novel entities / numbers vs clean band).
Side experiment (not selecting): early abort -- chain_{seg,sent}_bo16_half on the 30 smoke prompts logs each
candidate's first-half projection, to replay "abort a candidate at mid-unit when its partial depth is low".

### Amendment 6 — outcome (read 2026-09-24 00:27; runs/ss_power_r2.json, runs/quality_vs_clean_ss_llama3b_r2.json, runs/fact_probe_ss_llama3b_r2.json)
t_seg detection at the frozen 1% threshold (clean / reorder / resegment / para-light / para-strong): segA_b2 1% / 3 / 1;
segA_b3 3% / 4 / 4; segB_b2 11% / 5 / 8; bestof4_seg 22% / 6 / 10; bestof4B_b1 52.7% [45-60] / 22.4 / 35.4 / 6.7 / 5.3.
Judge (win vs clean; clean-vs-clean 0.425): A β2 0.462, A β3 0.443, B 0.418, C 0.443, E 0.425.  Flaw measures outside
the clean band: A β2 insertions (0.027), A β3 insertions (0.020), E flaws/text (0.593, band <= 0.587, diff +0.093
[+0.003, +0.187]) and insertions (0.020, diff n.s.); B and C: none.  Rule -> selected arm C (22%).  The step-0 predicted
ranges were wrong for A (over) and E (under); predictions are no longer written into amendments (user instruction).

### Amendment 7 — outcome (read 2026-09-24 01:35; runs/detect_ss_llama3b.json, runs/quality_vs_clean_ss_llama3b.json, runs/fact_probe_ss_llama3b_r3.json, runs/ss_abort.json)
chain_sent_blk4x16: z_chain 90.0% [84-94] (AUROC 0.986); reorder 16.1 / resegment 8.1 / para-light 10.7 / para-strong
4.0%; 8.87 candidates per unit (median 8; 33% of units reach the cap of 16); 48.5 s/text; win 0.477; all five flaw
measures inside the clean band; novel numbers 0.17, entities 0.52 (clean 0.16 / 0.47).
chain_seg_blk4x16: z_segchain 83.3% [77-88] (AUROC 0.987); reorder 4.1 / resegment 23.8 / para-light 2.0 / para-strong
1.3%; 8.06 candidates per unit (median 4; 24% reach 16); 42.6 s/text; win 0.458; all five inside; numbers 0.18,
entities 0.47.  Shipped SemStamp on the same prompts: 68.0% [60-75], 60.2 candidates per sentence, 319.5 s/text, win 0.485,
reorder 12.3 / resegment 9.6 / para-light 27.3 / para-strong 7.3%.
Criterion (>= 63%, <= 1/5 of SemStamp's candidates, all flaw measures inside the band): PASSED for both arms.
Paraphrase: both chain arms keep less of the mark than shipped SemStamp under light paraphrase (10.7% and 2.0% vs 27.3%).
Early abort (30-prompt replay, runs/ss_abort.json): corr(half-unit depth, full depth) 0.58-0.60; at tau -0.04, 19% of
candidates aborted, token cost 8.06 -> 7.23 candidate-equivalents per unit (-10%), detection unchanged.

### Round 2 addendum (2026-09-24, not pre-registered; arms implemented by the user, evaluated by the same jobs)
Same run directory, null, thresholds and judge as amendment 5. `chain_sent_blk4x16` (SemStamp chain, per unit blocks
of 4 candidates up to 16, keep the deepest in the mask, no margin, no retry loop): 90% [84-94] at 1% FPR, preference
0.48 [0.41-0.54], all five flaw measures inside the clean band, 48 s/text. `chain_seg_blk4x16`: 83%, neutral, 43 s.
`bestof4B_b1`: 53%, near-neutral. Steering-only arms `segA_b2/b3`, `segB_b2`: 1-11%. Claude's `bestof4_seg`: 22%,
neutral. Attacks: no quality-neutral arm exceeds 7% under strong paraphrase; chained arms retain 4-16% under sentence
reordering. Conclusion recorded: selection among exact samples, not steering, is what improves SemStamp on this stack.

## Amendment 8 (2026-09-24, before any text of these arms exists): bounded chain + E-style push
Arms (runs/ss_llama3b, 150 held-out prompts): chain_sent_blk4x16_push1, chain_seg_blk4x16_push1 = the amendment-7 chain
(blocks of 4, stop at the first block with an in-mask candidate, keep the deepest, cap 16, no margin) with the
true-shift nucleus push of arm E at beta 1 (NucleusSteerProcessor, mode marginal), targeted at the nearest bin of each
unit's mask (target_bins = mask; first unit unpushed).  Reported as for amendment 7: detection (own chain statistic,
frozen thresholds), attacks, candidates per unit, s/text, judge vs the clean band, fact probe.  Decision rule: an arm
improves on its amendment-7 twin if it needs fewer candidates per unit at detection no lower than the twin's CI lower
bound (sent 84%, seg 77%) with every flaw measure inside the clean band.

## Amendment 9 (2026-09-24, before any text of these arms exists): bounded chain + soft margin
Arms (runs/ss_llama3b, 150 held-out prompts): chain_sent_blk4x16_m0.02, chain_seg_blk4x16_m0.02 = the amendment-7 chain
without push, where a block stops the unit only on a candidate that is in the mask AND at least 0.02 from every
hyperplane (SemStamp's margin); the deepest such candidate is kept; at the cap of 16 the deepest in-mask candidate
(else the closest) is kept -- no rejection loop, nothing forced beyond the cap.  Motivation (measured): the amendment-7
arms keep 10.7% (sent) / 2.0% (seg) under light paraphrase vs SemStamp's 27.3%; in the replay (runs/ss_margin_replay.json)
this rule raises the share of kept units clearing 0.02 from 0.17-0.18 to 0.30-0.35 at 13.1-13.5 candidates per unit.
Decision rule: the margin arm is preferred over its amendment-7 twin if light-paraphrase detection is higher (Wilson
intervals not overlapping) with clean detection within the twin's CI, <= 1/4 of SemStamp's candidates per unit
(15.05), and every flaw measure inside the clean band.

### Amendment 8 — outcome (read 2026-09-24 11:55; runs/detect_ss_llama3b.json (job 275831), runs/quality_vs_clean_ss_llama3b.json (job 275832), runs/fact_probe_ss_llama3b_r4.json)
chain_seg_blk4x16_push1: z_segchain 93.3% [88-96] (twin 83.3%); reorder 4.8 / resegment 27.6 / para-light 2.0 /
para-strong 1.3%; 6.90 candidates per unit (twin 8.06); 58.1 s/text (twin 42.6); win 0.400 [0.333-0.473] (twin 0.458);
all five flaw measures inside the clean band; numbers 0.16, entities 0.50.  -> IMPROVES on its twin by the rule.
chain_sent_blk4x16_push1: z_chain 89.9% [84-94] (twin 90.0%), t_cont 77.2% (twin 68.0%); reorder 16.3 / resegment 10.2 /
para-light 18.8 [13-26] (twin 10.7) / para-strong 8.1% (twin 4.0); 7.87 candidates per unit (twin 8.87); 62.0 s/text
(twin 48.5); win 0.405 [0.342-0.470] (twin 0.477); insertions 0.020 = 3/150 OUTSIDE the band (diff +0.017 [-0.003,
+0.043], n.s.), other four inside; numbers 0.16, entities 0.54.  -> does NOT meet the rule (one flaw measure outside).

### Amendment 9 — outcome (read 2026-09-24 12:25; runs/detect_ss_llama3b.json (job 275861), runs/quality_vs_clean_ss_llama3b.json (job 275862), runs/fact_probe_ss_llama3b_r5.json)
chain_sent_blk4x16_m0.02: z_chain 85.2% [78.7-90.0] (twin 90.0% [84.2-93.9]); para-light 22.8% [16.8-30.2] (twin 10.7%
[6.7-16.7]); para-strong 5.4%; reorder 19.0%; resegment 9.5%; 14.05 candidates per unit (79% of units reach the cap);
78.5 s/text; win 0.462 [0.393-0.528]; all five flaw measures inside the band; numbers 0.20, entities 0.54.
Rule: para-light intervals do not overlap (16.81% > 16.73%, by 0.08 pt), clean inside the twin's CI, 14.05 <= 15.05,
all flaws inside -> PREFERRED over its twin (narrowly on the interval criterion).
chain_seg_blk4x16_m0.02: z_segchain 86.7% [80-91] (twin 83.3%); para-light 4.0% (twin 2.0%, intervals overlap);
para-strong 0.0%; reorder 3.4%; resegment 30.1%; 13.63 candidates per unit (74% reach the cap); 69.5 s/text; win 0.485
[0.420-0.552]; all five inside; numbers 0.20, entities 0.51.  Rule -> NOT preferred (paraphrase gain not significant).

## Amendment 10 (2026-09-24, before any text of these arms exists): bounded chain + push + soft margin (merged)
Arms (runs/ss_llama3b, 150 held-out prompts): chain_sent_blk4x16_push1_m0.02 (primary) and chain_seg_blk4x16_push1_m0.02
= the amendment-7 chain with BOTH the amendment-8 push (true-shift, beta 1, toward the nearest mask bin) and the
amendment-9 stop rule (a block stops the unit only on an in-mask candidate >= 0.02 from every hyperplane; cap 16).
Motivation (measured): light-paraphrase survival for sentences 10.7% (plain) -> 18.8% (push) / 22.8% (margin); share of
candidates in the mask and clearing 0.02: 2.19% without push vs 4.04% with push (sentences), 2.74% vs 5.43% (segments).
Decision rule: the merged arm is preferred over the amendment-9 margin arm of the same unit if it needs fewer
candidates per unit with light-paraphrase detection inside or above the margin arm's Wilson interval (sent [16.8-30.2]),
clean detection >= the margin arm's CI lower bound (sent 78.7%), and every flaw measure inside the clean band.
Judge preference is reported next to the margin arm's (sent 0.462, seg 0.485).

### Amendment 10 — outcome (read 2026-09-24 15:05; runs/detect_ss_llama3b.json (job 276099), runs/quality_vs_clean_ss_llama3b.json (job 276100), runs/fact_probe_ss_llama3b_r6.json)
chain_sent_blk4x16_push1_m0.02: z_chain 91.3% [86-95], t_cont 84.7%; reorder 20.4 / resegment 19.0 / para-light 26.0%
[20-34] / para-strong 7.3%; 12.59 candidates per unit (65% of units reach 16); 111.6 s/text; win 0.433 [0.368-0.503];
all five flaw measures inside the clean band; numbers 0.18, entities 0.54.  Rule vs chain_sent_blk4x16_m0.02 (14.05 cand,
para-light [16.8-30.2], clean CI low 78.7%): fewer candidates, para-light inside/above, clean above, flaws inside -> PREFERRED.
Shipped SemStamp on the same prompts: 68.0% / reorder 12.3 / resegment 9.6 / para-light 27.3% [21-35] / para-strong 7.3%;
60.2 candidates; 319.5 s/text; win 0.485.
chain_seg_blk4x16_push1_m0.02: z_segchain 94.0% [89-97]; reorder 8.1 / resegment 39.9 / para-light 10.7% / para-strong 3.3%;
12.04 candidates per unit; 141.2 s/text; win 0.402 [0.337-0.473]; all five inside; numbers 0.20, entities 0.50.
Rule vs chain_seg_blk4x16_m0.02 (13.63 cand, para-light 4.0% [2-8]) -> PREFERRED.

### Amendment 10 — outcome (read 2026-09-24 15:05; runs/detect_ss_llama3b.json (job 276099), runs/quality_vs_clean_ss_llama3b.json (job 276100), runs/fact_probe_ss_llama3b_r6.json)
chain_sent_blk4x16_push1_m0.02: z_chain 91.3% [86-95]; para-light 26.0% [20-34]; para-strong 7.3%; reorder 20.4%;
resegment 19.0%; 12.59 candidates per unit (65% reach the cap); 111.6 s/text; win 0.433 [0.368-0.503]; all five flaw
measures inside; numbers 0.18, entities 0.54.  Rule vs the amendment-9 margin arm (14.05 cand; para-light [16.8-30.2];
clean >= 78.7%): PREFERRED.  Shipped SemStamp: 68.0% / 27.3 / 7.3 / 12.3 / 9.6%, 60.2 cand, 319.5 s/text, win 0.485.
chain_seg_blk4x16_push1_m0.02: z_segchain 94.0% [89-97]; para-light 10.7%; para-strong 3.3%; reorder 8.1%; resegment
39.9%; 12.04 candidates per unit (58% reach the cap); 141.2 s/text; win 0.402 [0.337-0.473]; all five inside;
numbers 0.20, entities 0.50.  Rule vs its margin twin (13.63 cand, para-light 4.0%): PREFERRED.

## Amendment 11 (2026-09-24, before any adapter-generated text exists): bin adapters
Training data: 1,350 fresh prompts (data/ss_train/prompts.jsonl; 400 train + 50 validation per domain) from the raw
sources, excluding every dedup group with a prepared member (report: 0 shared source ids, 0 shared prompt hashes).
Corpus: plain bounded chain on sentences (--arm chain --unit sent --best-of 16), all 16 candidates of every unit logged
(runs/ss_corpus, runs/ss_corpus_val); smoke (runs/ss_corpus_smoke) passed the integrity check (6/6 texts rebuilt,
200/200 bins reproduced).  Adapters (wmworld/multilora.py, scripts/ss_train_adapters.py): 8 LoRA adapters (r 16,
alpha 32, q/k/v/o/gate/up/down) on the frozen 3B, one per LSH bin of the shipped key; maximum likelihood of each
candidate under its own bin's adapter; adapter active from the last prefix token on; loss on logits / 0.8.
Held-out arms (150 prompts; each changes ONE thing vs the reference chain_sent_blk4x16_push1_m0.02):
  chain_sent_adapt_m0.02      the push is replaced by the adapters (blocks of 2+2 candidates, one pair per mask bin),
                              keep-deepest rule with stop margin 0.02, cap 16
  chain_sent_adapt_sir_m0.02  as above, but among in-region candidates one is resampled with probability proportional
                              to p/q_mix (processed sampler probabilities) instead of keeping the deepest
Reported next to the reference: detection (own statistic, frozen thresholds), attacks, candidates per unit, s/text,
judge vs the clean band, fact probe.  Decision rule: an adapter arm is preferred over the reference if it needs fewer
candidates per unit, with clean and light-paraphrase detection inside or above the reference's Wilson intervals and
every flaw measure inside the clean band.  No forecasts are recorded (user instruction).
Amendment 11, training note (2026-09-24 17:58): run v1 (lr 2e-4) validation NLL per token after 1,000 contexts: own
adapter 0.4958, base 0.4668, other adapter 0.4990 (worse than base; train loss rose from 0.434 to 0.479 at peak lr).
A second run at lr 5e-5 (runs/ss_adapters/v1_lr5e-5) started in parallel.  Checkpoints are saved only when the
own-adapter validation NLL improves on step 0, so an adapter that never beats the base stays identical to the base.
Amendment 11, training note (2026-09-24 18:03): run v1 (lr 2e-4) stopped at 1,600 contexts: train loss per token rose 0.434 -> 0.494; last validation own 0.4958 vs base 0.4668.  Ceiling of the objective (scripts/ss_bin_ceiling.py, validation corpus): a perfect bin-conditional model gains 1.308 nats per sentence = 0.0753 nats per token.

## Amendment 12 (2026-09-24): bin head (predicting where a sentence will land), evaluated on prediction quality
Frozen 3B, forward only; input = final hidden state at each position of each corpus candidate + next-token embedding.
v1 (runs/ss_binhead/v1, scripts/ss_train_binhead.py): 9-class region target (bin b clear of 0.02, or near a plane);
after 1 epoch (24,425 contexts) validation cross-entropy best 0.8019 vs 0.8134 for class frequencies alone; AUROC for
ending in the mask's clear region 0.683 at the first token, 0.758 mid-sentence.
v2 (runs/ss_binhead/v2_proj, scripts/ss_train_projhead.py): target = the 3 projections (Gaussian per plane), region
probabilities derived; judged on the same 300 validation contexts by R^2 / sign accuracy / 1-sd coverage per plane,
derived region cross-entropy and the same AUROCs.  Adapters lr 5e-5 (runs/ss_adapters/v1_lr5e-5), own-adapter
validation NLL at 1,000/2,000/3,000 contexts: 0.4648 / 0.4639 / 0.4648 vs base 0.4668 (ceiling gain 0.0753).

### Amendments 11-12 — outcome (2026-09-24 19:25): stopped by the user as not worth further research
Adapters (lr 5e-5), own-adapter validation NLL per token at 1,000-7,000 contexts: 0.4648, 0.4639, 0.4648, ..., 0.4646,
0.4648 vs base 0.4668 (about 3% of the 0.0753 ceiling); lr 2e-4 diverged (0.4958).  Bin head v1 (9 classes), final:
cross-entropy 0.8019 vs 0.8134 for class frequencies; AUROC for ending in the mask's clear region 0.683 (first token),
0.758 (mid-sentence).  Bin head v2 (projections), final (1 epoch): R^2 per plane 0.190 / 0.164 / 0.157 (0.139 in the first fifth of the
sentence, 0.189 in the last), sign accuracy 0.63-0.65, region cross-entropy 0.8127 (class frequencies 0.8134), AUROC
0.665 (first token) / 0.734 (mid-sentence).  No held-out arm was run with either.

## Amendment 13 (2026-09-24, before any RL result): RL for "in the mask AND meaningful"
scripts/ss_rl_mask.py.  Policy = frozen Llama-3.2-3B + 8 LoRA adapters (r 16), adapter index = mask id (previous
sentence's bin).  Sentence-level episodes in the training corpus contexts (runs/ss_corpus); per context G=8 sentences
from the policy under the 3B's sampler.  Reward = clip(depth in mask / 0.05, -1, 1) + (judge - 3) / 2 + guards (-1 for
no proper ending or a repeated 4-gram); judge = Qwen3.5-0.8B (local, non-thinking; different family from the policy
and from the evaluation judge gemma-4-31b), expected 1-5 rating of the sentence as a continuation.  Group-normalised
advantages, one update per batch, KL(pi || base) penalty (k3, beta 0.05), lr 5e-5.  Choices of "KL + local judge" and
"depth in mask" are the user's.  Monitored on fixed validation contexts: in-mask rate, clear-of-0.02 rate, depth, judge
score, KL per sentence, vs the base at step 0; the judge's reliability is checked (real next sentence vs a sentence
from another text).  Evaluation afterwards on the OpenStamp-style benchmark (data/ob_bench).
Amendment 13 note (2026-09-24 19:55): smoke (runs/ss_rl/smoke, 30 steps, 406 s) ran end to end; the repetition guard was narrowed after it fired on 15.6% of the base's own sentences: now a 4-gram repeated within the sentence or >= 8 tokens copied verbatim from the context.  Full run: runs/ss_rl/v1 (1,500 steps).
Amendment 13 note (2026-09-24 20:35): run v1 (lr 5e-5) monitor at step 100 vs base on the same 48 contexts and seeds: in-mask 0.229 vs 0.245, clear 0.042 vs 0.042, judge 3.340 vs 3.332, KL 0.012 nats/sentence.  Second run started at lr 2e-4 (runs/ss_rl/v1_lr2e-4), otherwise identical.
Amendment 13 note (2026-09-24 21:00): run lr 5e-5 stopped (no movement: KL 0.012 nats/sentence at step 100).  Judge via OpenRouter tested on 16 parallel calls: gemma-3-4b-it rate-limited (429), gemma-3-12b-it all 429, gemma-4-26b-a4b-it 7.1 s wall, gemma-3-27b-it 2.9 s wall with log-probabilities 16/16 -> chosen (a different Gemma generation from the evaluation judge gemma-4-31b).  New run: lr 2e-4 + gemma-3-27b-it judge (runs/ss_rl/v1_lr2e-4_gemma), to compare with lr 2e-4 + local Qwen3.5-0.8B judge (runs/ss_rl/v1_lr2e-4).
Amendment 13 note (2026-09-24 22:10): runs lr 2e-4 stopped.  Local judge: clear-in-mask 0.042 (base) -> 0.099 (step
100) -> 0.070 (step 200) with drift 5.9 -> 8.7 nats/sentence, judge 3.33 -> 3.38.  Gemma judge: 0.042 -> 0.063 -> 0.049,
drift 2.5 -> 10.6, judge 4.58 -> 4.31, guard failures 0.039 -> 0.109.  Run v2 (runs/ss_rl/v2, local judge): group 16,
beta 0.2, lr 2e-4, checkpoint at every monitor (every 100 steps) to /l/users/daniil.orel/wmworld_rl/v2, stop when drift
rose over 2 consecutive monitors without beating the best clear-in-mask share.

## Amendment 14 (2026-09-24, before any result): RL-trained steering head (second GPU, alongside LoRA run v2)
scripts/ss_rl_steer.py.  The 3B is frozen with no adapters; a 3.4M-parameter head biases the 64 most likely lexical
tokens of the 3B's own top-p nucleus per step (bias = 2 * tanh(.), last layer zero-initialised so step 0 == base;
EOS/punctuation/whitespace never biased and their total probability kept exact; tokens outside the nucleus stay
impossible).  Inputs: 3B hidden state, candidate token embedding, mask id.  Same reward, GRPO (group 16), KL weight
beta 0.2 (exact per-token KL), monitors, stop rule and checkpoint schedule (/l/users/daniil.orel/wmworld_rl/steer_v1)
as LoRA run v2; head lr 1e-3.
Amendment 14 note (2026-09-24 22:45): first launch crashed at the first update (NaN gradients through masked -inf entries -> device assert in sampling); fixed by masking inputs with finite values (test added); relaunched.
Amendment 14 outcome (2026-09-24 23:15): steering head stopped by the user's decision at step ~280: validation clear-in-mask 0.044 (step 100) / 0.042 (step 200) vs base 0.042, judge 3.349 / 3.350 vs 3.349, training-batch KL about 0.01 nats/sentence (the head did not move).  Next: amendment 15, the head trained to reproduce the true-shift push (supervised), with the partial sentence's projections as extra input.

## Amendment 15 (2026-09-24, before any result): the push distilled into a small head (A+B)
wmworld/pushhead.py, scripts/ss_pushhead.py.  Targets = the true-shift push's own standardised gains z_v (NucleusSteerProcessor
mode "marginal", mask target, top 20 lexical nucleus tokens, 20 encoder calls) at every token position of 2 corpus
candidates per context (1,500 training contexts; 300 validation contexts from the validation prompts).  Head inputs: 3B
hidden state, candidate token embedding, mask id, and 8 numbers about the partial unit from ONE encoder call (its 3
projections signed by the mask's 2 bins, its capped depth, its length).  Loss: MSE of the head's standardised outputs
against z_v.  Fidelity on validation positions: R^2 of z, within-step Spearman, top-token agreement, sign agreement.
If the fidelity is usable, the head replaces the push's 20 encoder calls in the chain (then RL fine-tuning, direction C).

## Amendment 16 (2026-09-24, before any result): rejection-sampling fine-tuning (RFT) of the mask adapters
scripts/ss_rft_lora.py (user's proposal: reproduce the accepted samples for a few rounds, then RL).  Same policy shape
as the RL runs (8 LoRA mask adapters, r 16).  Round 1: the training corpus's accepted candidates (in the mask and
>= 0.02 from its planes; no new generation).  Rounds 2-3: 16 samples per context from the current adapters on 1,500 fresh
training contexts, accepted = same rule + no guard failure.  One epoch of next-token likelihood per round, lr 5e-5.  After
each round the RL monitor (same 48 validation contexts x 16 sentences, same seeds, local judge) and a checkpoint to
/l/users/daniil.orel/wmworld_rl/rft_v1.  Afterwards RL (GRPO) can start from the last round's adapters.

### Amendment 13 run v2 — outcome (2026-09-25 00:20; runs/ss_rl/v2/rl_log.json, checkpoints /l/users/daniil.orel/wmworld_rl/v2/adapters_step{100..500}.pt)
Monitor (48 validation contexts x 16 sentences, fixed seeds): step 0 / 100 / 200 / 300 / 400 / 500
  clear-in-mask 0.042 / 0.055 / 0.053 / 0.081 / 0.076 / 0.053;  in-mask 0.233 / 0.236 / 0.232 / 0.285 / 0.295 / 0.271
  depth -0.0189 / -0.0183 / -0.0182 / -0.0149 / -0.0142 / -0.0158;  judge (Qwen3.5-0.8B) 3.349 / 3.379 / 3.362 / 3.366 / 3.420 / 3.439
  guard failures 0.042 / 0.046 / 0.042 / 0.031 / 0.030 / 0.025;  KL nats/sentence 0 / 1.28 / 2.46 / 2.79 / 4.86 / 7.88
Stopped by the rule at step 500 (drift rose over 2 monitors without beating clear-in-mask 0.081).  Best checkpoints:
step 300 (clear-in-mask) and step 400 (in-mask, depth).  Not yet evaluated in the chain or on the benchmark.

### Amendment 16 round 1 — outcome (2026-09-25 00:20)
Trained on 15,283 accepted corpus sentences from 6,778 contexts (1 epoch, lr 5e-5): monitor in-mask 0.224, clear-in-mask
0.044, depth -0.0189, judge 3.350, KL 0.321 nats/sentence -- the same as the base (0.233 / 0.042 / -0.0189 / 3.349).
Round 2 (on-policy sampling on 1,500 contexts) running.

### Amendment 16 — outcome (2026-09-25 02:46; runs/ss_rft/v1/rft_log.json, checkpoints /l/users/daniil.orel/wmworld_rl/rft_v1/adapters_round{1,2,3}.pt)
Monitor (same 48 validation contexts x 16, fixed seeds), base / round 1 / round 2 / round 3:
  clear-in-mask 0.042 / 0.044 / 0.059 / 0.056;  in-mask 0.233 / 0.224 / 0.237 / 0.250;  depth -0.0189 / -0.0189 / -0.0183 / -0.0176
  judge 3.349 / 3.350 / 3.338 / 3.352;  guard 0.042 / 0.040 / 0.038 / 0.038;  KL nats/sentence 0 / 0.32 / 0.53 / 0.86
Training sets: round 1 15,283 accepted corpus sentences (6,778 contexts); round 2 2,028 (528 of 1,500 contexts; on-policy
acceptance on those training contexts 2,528/23,973 = 0.105); round 3 2,233 (554 contexts; 2,741/23,979 = 0.114).
GRPO v2 for comparison: best clear-in-mask 0.081 (step 300), best in-mask 0.295 (step 400).

## Amendment 17 (2026-09-25, before any benchmark text exists): the OpenStamp-style benchmark, ArXiv / BookSum / Wikipedia
Test set = data/ob_bench test split of ob_arxiv, ob_booksum, ob_wikipedia (500 prompts each: first 50 tokens of a
document, 200-token continuation; the user's choice of datasets after OpenStamp, arXiv 2608.27899).  Run runs/ob_bench.
Arms: base (plain 3B), clean_seed1 (second clean draw: judge band), semstamp (shipped SemStamp on the plain 3B),
chain_reference (= assets/ss_reference.json: chain, push beta 1, stop margin 0.02), chain_grpo300 / chain_grpo400 (chain,
stop margin 0.02, candidates from the GRPO v2 adapters at step 300 / 400, mask-indexed, no push), chain_rft3 (same with
the RFT round-3 adapters).  Detection (scripts/ob_detect.py): z_chain and t_cont per dataset at 1% and 0.1% FPR from
that dataset's 2,000 human null continuations, 1% from the plain 3B's own test continuations, AUROC; clean and under
reorder / resegment / light and strong paraphrase (gemma-4-31b).  Quality: gemma-4-31b preference vs base and the five
flaw measures vs the clean band (base vs clean_seed1).  Cost: candidates per unit, s/text.  No forecasts recorded.
Amendment 17 addition (2026-09-25, before any of their text exists): segment arms chain_seg_push (chain, 16-word segments, push beta 1) and chain_seg_push_m (+ stop margin 0.02), the best-detecting segment arms on the held-out set (93.3% / 94.0%); read with z_segchain / t_segcont.
Amendment 17 note (2026-09-25 13:20): the /l filesystem hung (even `ls` on the login node blocks); job 277394 froze
loading chain_grpo300 from /l after finishing base, clean_seed1 and chain_reference.  The 3B was re-downloaded at the
same pinned revision (0cb88a4f) to /home/daniil.orel/.cache/huggingface (assets/ss_models.json updated; old file kept
as assets/ss_models.json.bak_l).  The RL/RFT checkpoints exist only on /l: the three adapter arms
(slurm/conf/ob_gen_adapters.sbatch, reading /home/daniil.orel/wmworld_rl) run once /l responds and they are copied.
Amendment 17 note (2026-09-25 13:30): the GRPO v2 recipe (to step 400) and the 3 RFT rounds are re-run with checkpoints on /home (slurm/conf/ss_retrain_home.sbatch -> /home/daniil.orel/wmworld_rl/{v2_rerun,rft_v1_rerun}); RL sampling is not bit-exact on GPU, so these are re-runs of the recipe, not the /l checkpoints.  Early attacks + judge for base, clean_seed1, chain_reference started (job ob_early_tail).
Amendment 17 note (2026-09-25 13:40): the user's decision: no copies -- when /l is restored the adapter arms read the original checkpoints from /l (slurm/conf/ob_gen_adapters.sbatch); the /home retrain (job 277623) was cancelled before it started.
Amendment 17 note (2026-09-25 13:40): /l responded again; adapter arms submitted reading /l (job 277641); second attacks/judge/detect pass 277642/277643/.
Amendment 17 note (2026-09-25 17:22): second detect pass submitted as job 278029 (after attacks 277642; the first attempt hit the QoS submit limit).
Amendment 17 note (2026-09-25 21:18): chain_rft3 generated in parallel by job 278429 (slurm/conf/ob_gen_rft3.sbatch, same flags as 277641, which skips existing outputs); the second pass (277642/277643/278029) still starts after 277641.
Amendment 17 note (2026-09-25 21:32): adapter decoding sped up without changing the computation (wmworld/multilora.py: row pattern decided on the host, no per-layer GPU syncs, single-adapter path without gather; test in tests/test_multilora.py). Speed test on 18 ArXiv prompts (runs/ob_speedtest, 3 processes per GPU): 214.8 -> 38.8 s/text, candidates per sentence 13.22 -> 12.83, kept in mask 0.851 -> 0.866. Jobs 277641 and 278429 (old code) cancelled; texts already written are kept and resumed by jobs 278450 (GRPO 300, 400) and 278451 (RFT round 3); the second pass waits for both.

## Amendment 18 (2026-09-26, before any result of these runs): GRPO run 3, two arms, queued behind the benchmark
Both from the BASE model (as v2), v2's recipe (lr 2e-4, KL coefficient 0.2, local Qwen3.5-0.8B judge, same 48 monitor
contexts) with 32 rollouts per context (v2: 16; the monitor therefore reads 48 x 32 = 1,536 sentences) and the stop rule
"5 monitors in a row without a new best clear-in-mask share" (--stale-rule any --stop-patience 5; v2 stopped after 2
monitors of rising drift without a new best).  Checkpoints every 100 steps to /l/users/daniil.orel/wmworld_rl/<tag>.
  v3a_patience (job 278721): v2's reward.
  v3b_para (job 278722): v2's reward + w_para * clip(depth of the paraphrased sentence / 0.05, +-1), w_para = 1; the paraphrase
  is written by google/gemma-3-12b-it through OpenRouter (temperature 0.7, one sentence, meaning/facts/names/numbers kept);
  the evaluation attacker (gemma-4-31b) is NOT used in training.  Monitor adds in-mask / clear rates after paraphrase.
Each job runs a 2-step smoke first (memory at 32 rollouts, API) and trains only if it passes.  Both start after the second
evaluation pass (277642, 277643, 278029).  Caveat recorded: training paraphrases single sentences in context, the
evaluation paraphrases whole texts.
Amendment 18 note (2026-09-26 02:08): on the user's request v3a_patience (job 278721) was released before the evaluation finished (one GPU; the other slot stays free for detection 278029); v3b_para (278722) still starts after the evaluation.
Amendment 18 note (2026-09-26 10:46): benchmark evaluation of run-3 checkpoints v3a step 1000 (best local-judge monitor), v3a step 1400 (best clear share 0.490), v3b step 600 (best clear-after-paraphrase 0.323): generation 278873 (slurm/conf/ob_gen_v3.sbatch, same flags as the other adapter arms), attacks 278874, judge 278875 (gemma-4-31b), detection 278876 -> runs/ob_detect_v3.json, runs/quality_vs_clean_ob_bench_v3.json.
Note (2026-09-26 10:53): arm chain_grpo300_push = GRPO v2 step-300 adapters with the true-shift push (beta 1) applied to the adapter logits inside adapter_candidates (scripts/steered_semstamp.py; backup .bak_adapush); jobs gen 278890, attacks 278891, judge 278892, detect 278893 -> runs/ob_detect_grpopush.json, runs/quality_vs_clean_ob_bench_grpopush.json.

## Amendment 19 (2026-09-26, before any result of these runs): larger reward judge; push distillation as warm-up
Motivation (measured): run 3a reached clear share 0.490 (step 1,400) at 36 nats/sentence drift with degenerate in-mask
sentences ("And And something interesting occurred.") that the local Qwen3.5-0.8B judge scored 3.05; the judge's
real-vs-mismatched AUROC is 0.845 against 0.989 for gemma-3-27b-it.  On the benchmark the push beats every adapter arm
(ArXiv first-block usable draws 11.3% vs 5.4% for GRPO step 300).
  v4a_bigjudge: run 3a's recipe (base model, 32 rollouts, patience 5, stale-rule any) with the reward judge
  google/gemma-3-27b-it through OpenRouter (not the evaluation judge gemma-4-31b).  One change against 3a.
  distill_push: corpus of pushed samples (sentence chain, true-shift push beta 1, all 16 candidates per position in one
  batch, logged; 600 training prompts, 200 per domain); mask-indexed LoRA adapters fitted by next-token likelihood
  (T 0.8, lr 5e-5, 1 epoch, up to 8 candidates per position) on ALL candidates = distillation of the pushed sampler
  (scripts/ss_rft_lora.py --accept all --rounds 1); RL monitor after.  Sequence-level, not token-level, distillation:
  it reuses the tested corpus and fine-tuning code.
  v4c_distill_bigjudge: v4a's recipe started from the distilled adapters (the "warm-up, then RL" plan).
Queued by job 278894 after the benchmark detections 278876 and 278893.  No drift budget in any of the three (one change
at a time); the monitor still reports KL per sentence.
Note (2026-09-26 15:09): chain_grpo300_push continued on 2 GPUs as 6 shards by job 279171 (the generator now skips ids written by any shard file of the arm); 278890 cancelled; its attacks/judge (278891/278892) now depend on 279171.
Amendment 19 note (2026-09-26 15:11): the three runs are one 2-GPU job 279178 (slurm/conf/ss_train_v4.sbatch; lane 0 v4a_bigjudge, lane 1 distill_push then v4c_distill_bigjudge), started while GRPO+push generation continues on the other 2 GPUs.
Amendment 19 note (2026-09-26 15:16): on the user's request training waits for all evaluations: job 279178 cancelled a few minutes in (smokes had passed, no checkpoint), requeued as 279184 after the detections 278876 and 278893.
Note (2026-09-26 15:22): chain_grpo300_push continued on 3 GPUs / 9 shards by job 279185 (279171 cancelled; 867 texts kept); attacks/judge 278891/278892 now depend on 279185; 1 GPU left for detection.
Amendment 19 note (2026-09-26 15:24): to use every GPU under the 2-running-job limit: v4a_bigjudge runs now as job 279186 (1 GPU, next to the 3-GPU GRPO+push generation 279185); after 279185 a 3-GPU job  runs both benchmark detections (GPU 0, each as soon as its attacks end) and the push distillation + v4c (GPUs 1-2).  Replaced: 279184, 278876, 278893.
Amendment 19 note (2026-09-26 15:25): the combined job was rejected by the cluster (waiting loop in a GPU job); now worker 279188 (3 GPUs, after 279185 and the run-3 attacks 278874: run-3 detection on GPU 0 then corpus shards, corpus on GPUs 1-2, then distillation + v4c) and the GRPO+push detection on CPU as job 279189 (after its attacks 278891; arms base, clean_seed1, human, semstamp, chain_reference, chain_grpo300, chain_grpo300_push).
Note (2026-09-26 16:39): scripts/reorder_attack.py overwrote runs/ob_bench_{reorder_sent,resegment} with only the arms of each call (the run-3 attack job, then the GRPO+push job, dropped the other arms; runs/ob_detect.json of 02:57 predates this and is complete). Fixed to merge (backup .bak_overwrite); both files regenerated for all 13 arms (deterministic, seeded by text id) by job 279422; detections re-run on CPU: 279423 -> runs/ob_detect_grpopush.json, 279424 -> runs/ob_detect_v3.json (the GPU detection of 16:36 is superseded).
Amendment 18 note (2026-09-26 16:40): run 3 (v3a_patience, v3b_para) DROPPED by the user after the benchmark judge (win 0.028-0.291, major flaws 58-95%, score 1.17-2.43 vs ~3.1 unwatermarked): reward hacking of the 0.8B local judge with unbounded drift. Kept as a negative result: clean detection + paraphrase in runs/ob_detect_v3.json (GPU pass 16:36), judge in runs/quality_vs_clean_ob_bench_v3.json; its reorder/resegment re-run (279424) cancelled.
Amendment 19 note (2026-09-26 16:47), before any result of these runs: two more arms so no GPU idles after the corpus (user's go):
  v4b_para_bigjudge: v4a + the run-3b paraphrase reward (gemma-3-12b-it paraphraser, w_para 1) -- the paraphrase idea
  re-tested with the larger judge (3b failed through the weak judge, not through the paraphrase term).
  v4d_bigjudge_klcap: v4a + hard drift cap (--kl-cap 3 --kl-cap-penalty 2: a sampled sentence whose log(pi/base) exceeds
  3 nats gets reward - 2); 3 nats ~ GRPO v2 step 300 (2.79 nats, benchmark win 0.455), far below run 3 (20-94 nats).
  Worker 279188 cancelled during the corpus and replaced by job 279435 (corpus resumed on 3 GPUs, then lanes: GPU0 v4b,
  GPU1 distillation -> v4c, GPU2 v4d).  Code: scripts/ss_rl_mask.py --kl-cap/--kl-cap-penalty (backup .bak_klcap).
Amendment 19 note (2026-09-26 22:07): v4d_bigjudge_klcap stopped by the patience rule at step 600: clear share never beat 0.054 (step 100; base 0.046), drift 0.45-1.86 nats.
Amendment 19 note (2026-09-26 23:56): v4a_bigjudge stopped by the user at step ~1,340: from step 1,200 the clear share jumped 0.103 -> 0.257 while 3 of the 12 stored monitor samples were a lone ':' (in the mask, clear, judged 4-5 by gemma-3-27b) -- reward hacking of the larger judge via content-free units; drift 14-15 nats.  Clean candidate for the benchmark: step 800 (clear 0.103, drift 8.3 nats, 0/12 degenerate samples).
Amendment 19 note (2026-09-26 23:58): benchmark evaluation of v4a_bigjudge step 800 (arm chain_v4a800, same flags as the other adapter arms): generation 279772 (1 GPU), attacks 279773, judge 279774 (gemma-4-31b), detection 279775 (GPU, after the attacks) -> runs/ob_detect_v4a800.json, runs/quality_vs_clean_ob_bench_v4a800.json.
Amendment 19 note (2026-09-27 07:24): v4 runs final -- v4a stopped ~1340 (hack), v4b stopped by patience at 1000 (best clear 0.074 @500), v4c ran 1500 (best 0.112 @1500), v4d stopped at 600 (best 0.054 @100).  v4a step 800 on the benchmark: 62.2/91.0/52.0 at 1% FPR, win 0.433, score -0.10 (significant).  Benchmark evaluation of each run's best monitor step: v4c 1500 (job 280001), v4b 500 (279996), v4d 100 (280002, starts when a GPU job ends); attacks 280003, judge 280004, detection 280005 -> runs/ob_detect_v4bcd.json, runs/quality_vs_clean_ob_bench_v4bcd.json.
Amendment 19 note (2026-09-27 12:32): arm chain_v4c1500_push (v4c step-1500 adapters + push, as chain_grpo300_push): generation 280218 (3 GPUs), attacks 280219 (after it and after the v4bcd attacks 280003, which rewrite the same attack files), judge 280220, detection 280221 -> runs/ob_detect_v4cpush.json, runs/quality_vs_clean_ob_bench_v4cpush.json.

## Amendment 20 (2026-09-27, before any result of this run): token-level push distillation
Motivation (measured): the sequence-level distillation (amendment 19) left the adapters at plain-model level (monitor
clear share 0.043-0.057, base 0.046) although its pushed corpus was in the mask more often than the plain corpus
(0.324 vs 0.242 per candidate; clear 0.073 vs 0.042).  Method (scripts/ss_distill_token.py): for every training
context of runs/ss_corpus, one logged plain candidate; one base-model pass gives next-token logits at each position;
the push (NucleusSteerProcessor, marginal, beta 1, target = the position's mask; settings of the push arms) gives the
pushed logits over the sampler's top-p nucleus.  Adapters (8, r 16, index = previous bin, RL scope) are trained on
KL(pushed || adapter) over that nucleus at every token, T 0.8, lr 1e-4, 2 epochs, accumulation 8; RL monitor (local
judge, 48 contexts x 16) before and after each epoch.  Smoke (20 sentences): teacher KL(push || base) = 2.97 nats per
sentence (first measurement of the push's own drift).  Job 280246 (1 GPU), after the v4bcd detection 280005.

## Amendment 21 (2026-09-27, before any result of these runs): does push + margin generalise to other generators?
The LoRA/RL line is closed (amendments 13-20: no adapter arm beat the push on the benchmark, all cost quality);
push + margin (chain_sent_blk4x16_push1_m0.02 = chain_reference) is the method.  Same benchmark (ArXiv, BookSum,
Wikipedia; 500 test prompts each; 200 tokens; human-null thresholds from runs/ob_bench_null), same key, encoder,
sampler (T 0.8, top-p 0.95) and arms as the Llama-3.2-3B run, on two more generators:
  Llama-3.2-1B-Instruct (rev 9213176): gen 281030, attacks 281031, judge 281032, detection 281033
  Qwen3.5-0.8B (rev 2fc0636, text model, thinking off): gen 281034, attacks 281035, judge 281036, detection 281037
Arms per generator: base, clean_seed1 (second unwatermarked draw), chain_reference (push + margin), semstamp (shipped).
Code: scripts/steered_semstamp.py --target (Qwen: text-model class, <|im_end|> end token, <|fim_pad|> pad, no 128000
id filter in decode; Llama unchanged); smoke (2 prompts each) passed.  Results: runs/ob_detect_<target>.json,
runs/quality_vs_clean_ob_bench_<target>.json.

## Amendment 22 (2026-09-28, before any result of these runs): SemStamp's datasets and protocol, all three generators
User's request: RealNews, BookSum, Reddit-TIFU (SemStamp's datasets: Hou et al., NAACL 2024, verified in the paper:
32-token prompts, 200-token generations, 1,000 random samples) plus MBPP (not in SemStamp; the user chose to include
it although code domains had been excluded before).  data/sp_bench (scripts/sp_bench_build.py): 1,000 test prompts +
2,000 human-null continuations per text dataset; MBPP: all 974 problems as prompts (task + tests), the 974 reference
solutions as the human null (a 0.1% FPR threshold is not resolvable for MBPP).  Generators: Llama-3.2-3B, Llama-3.2-1B,
Qwen3.5-0.8B; arms: base, clean_seed1, chain_reference (push + margin), semstamp; same key, encoder, sampler, attacks
(gemma-4-31b paraphrases), judge (gemma-4-31b).  Detection on CPU (human-null thresholds from runs/sp_bench_null).
First-sentence fix (scripts/steered_semstamp.py): the first sentence is redrawn up to 8 times (the last time without the
min_new rule) instead of ending the text empty; smoke (20 Qwen texts over the 4 datasets): 0 empty, 1 redraw.  Each
model's job first regenerates its empty push + margin texts of data/ob_bench (23 / 34 / 148) and re-evaluates them
(runs/ob_detect_<run>_refill.json, runs/quality_vs_clean_<run>_refill.json).
  llama3b: generation 282715, attacks 282716, judge 282717, detection 282718
  llama1b: generation 282719, attacks 282720, judge 282721, detection 282722
  qwen08b: generation 282723, attacks 282724, judge 282725, detection 282726 (starts when either of the other generation jobs ends)
