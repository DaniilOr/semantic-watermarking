# Push + Margin vs SemStamp: main results snapshot

Everything behind `RESULTS.md` except the generated texts (2.6 GB, kept on the cluster). The experiment code lives in
`code/` as a snapshot of the working tree it was run from; on the cluster this folder is refreshed with `python scripts/export_main.py ~/wm_main_results` (code, prompts, generations) and
`python scripts/main_results.py ~/wm_main_results/RESULTS.md` (tables), both run from `~/wmworld`.

| Path | Contents |
|---|---|
| `RESULTS.md` | compact tables of the main results (measured numbers only) |
| `code/wmworld/` | the package; the method is in `semwm.py` (`NucleusSteerProcessor` = the push, depth, masks, detector statistic) |
| `code/scripts/steered_semstamp.py` | generator: SemStamp (`--arm semstamp`) and Push + Margin (`--arm chain --unit sent --block 4 --best-of 16 --steer marginal --beta 1 --stop-margin 0.02`) |
| `code/scripts/` | everything else: detection (`ob_detect.py`), attacks (`sent_paraphrase_attack.py`, paraphrase), judge (`api_judge.py`), task scoring (`it_eval.py`), tables |
| `code/slurm/` | the job files that produced the runs |
| `code/docs/` | method write-ups (`paper_approach.tex`, `push_margin_explained.md`) |
| `prompts/<bench>/prompts.jsonl` | prompts of SemStamp's datasets (`sp_bench`), our benchmark (`ob_bench`), instruction tasks (`it_bench`) |
| `generations/<run>/<arm>.jsonl` | generated texts with their full records, one file per arm (not in the git repo: 2.6 GB) |
| `results/` | summary JSON of every evaluation behind RESULTS.md: detection (`*detect*.json`), judge (`judge/`), task scores (`it_quality_*`), flaw measures (`quality_vs_clean_*`), perplexity (`ppl_part_*`), k-SemStamp (`ks_detect_*`), Dolly by category |
| `judge_win_tie_loss.png` | judge verdicts (watermarked better / tie / plain better) per task and model |

Arms: `base` = plain sample, `clean_seed1` = second plain sample (judge's no-difference level), `semstamp` = SemStamp,
`chain_reference` = Push + Margin; `it_fix_llama3b` also has the push variants (`chain_spread`, `chain_protect`,
`chain_spread_protect`, `chain_beta05`). Run suffixes: `_para_light` / `_para_strong` = Gemma paraphrase attacks;
`_pegasus`, `_parrot` (+ `_bigram`) = SemStamp's sentence-level attacks (`_parrot` = top beam; `_parrotss` = Parrot run exactly as SemStamp's code does); `*_null` = human-written texts that set the
detection thresholds. `sp_bench_llama3b` / `_llama1b` also contain MBPP code generations (domain `sp_mbpp`), not used in
the results.

Not included: models, the SBERT token table (`assets/etok_*.npz`, 321 MB), third-party IFEval code, judge logs.
