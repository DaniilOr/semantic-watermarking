#!/bin/bash
#SBATCH --job-name=pmark_tables
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/daniil.orel/wm_main_results/results/pmark_tables.txt
cd /home/daniil.orel/wmworld
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
echo "PMark (online, released settings) vs SemStamp and Push + Margin: % detected at the human-null 1% / 0.1% thresholds"
echo "(IFEval: the plain model's 1% threshold, no human null).  n = PMark texts scored.  Generated $(date +%F)."
echo; echo "== Llama-3B chat, SemStamp's datasets"; $PY scripts/pmark_compare.py runs/pmark_detect_sp.json pm_sp_bench_llama3b runs/sentpara_detect_sp_bench_llama3b.json sp_bench_llama3b sp_realnews sp_booksum
echo; echo "== Llama-3B chat, our benchmark"; $PY scripts/pmark_compare.py runs/pmark_detect_ob.json pm_ob_bench runs/sentpara_detect_ob_bench.json ob_bench ob_arxiv ob_booksum ob_wikipedia
echo; echo "== OPT-1.3B, SemStamp's setting"; $PY scripts/pmark_compare.py runs/pmark_detect_opt.json pm_setting_opt13b_ss runs/sentpara_detect_semstamp_setting_opt13b_ss.json semstamp_setting_opt13b_ss sp_realnews
echo; echo "== Llama-3B chat, instruct tasks (SemStamp and push have no attack runs on these tasks)"
$PY scripts/pmark_compare.py runs/pmark_detect_it.json pm_it_bench_llama3b runs/it_detect_llama3b.json it_bench_llama3b it_writingprompts it_dolly it_ifeval it_gsm8k
echo; echo "== plain model's own texts flagged by PMark's detector at the human-null 1% threshold"
for s in sp ob opt it; do $PY -c "
import json; d = json.load(open('runs/pmark_detect_${s}.json'))
for k, v in sorted(d['results'].items()):
    run, arm, dom = k.split('|')
    if arm == 'base' and 'null' not in run:
        print(f'{run:28s} {dom:18s} n={v[\"n\"]:5d} flagged {\"-\" if not v.get(\"human_1%\") else round(100 * v[\"human_1%\"][\"rate\"], 1)}')"; done
echo; echo "== diagnostic, NOT PMark's rule: neutral ties (tie = 0.5, miss = 0), Llama-3B chat, SemStamp's datasets"
$PY scripts/pmark_neutral_ties.py runs/pmark_scores/sp sp_bench_null pm_sp_bench_llama3b sp_bench_llama3b sp_booksum sp_realnews
