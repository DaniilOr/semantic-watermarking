# Ablation round 1 (spec Section 8, stage 4): 12 configurations, one axis each, on the frozen 60-prompt dev subset

All runs use `--stage ablation` (arm `llm_global`, the `ablation_subset` prompts: 20 per domain chosen by hash before
any result was read), seed 0, development keys, development scales. Each command needs its own `--dry-run` first
(budget manifest). Configurations that change the unit require the extra scale entries: uncomment the 8/32-token and
sentence unit settings in `configs/pilot.yaml` (`fit_scorer.unit_settings`) and re-run
`fit-scorer --partition scale_fit` (pools and scores resume; only the new units are computed).

| # | axis | override | run name |
|---|---|---|---|
| 1 | unit length | `{"unit_tokens": 8}` | ablation_unit8 |
| 2 | unit length | `{"unit_tokens": 32}` | ablation_unit32 |
| 3 | unit type | `{"unit": "sentence", "sentence_cap_tokens": 64}` | ablation_sentence64 |
| 4 | candidates | `{"candidates": 2}` | ablation_N2 |
| 5 | candidates | `{"candidates": 4}` | ablation_N4 |
| 6 | candidates | `{"candidates": 16}` | ablation_N16 |
| 7 | strength | `{"beta": 1.0}` | ablation_beta1 |
| 8 | strength | `{"beta": 4.0}` | ablation_beta4 |
| 9 | strength | `{"selection": "greedy"}` | ablation_greedy |
| 10 | coverage | `{"active_fraction": 0.5}` | ablation_active50 |
| 11 | coverage | `{"active_fraction": 0.25}` | ablation_active25 |
| 12 | output budget | `{"max_new_tokens": 256}` | ablation_max256 |

Proposed additions (amendments from the smoke trajectories; they exceed the 12-configuration first round and should be
traded against the table above if wanted):

| + | selection rule | `{"selection": "rank"}` (P ∝ exp(β·r), r = +1 best … −1 worst; no σ/floor/clip) | ablation_rank |
| + | selection rule | pool-centred gains before clipping (identical to the spec rule unless the ±5 clip binds) | ablation_centered |

(`beta = 0` is not spent here: it equals uniform selection by construction and is covered by the tests and by the
`budget_control` arm. `max_new_tokens = 128` and the `N=1`-inactive-slot cost ablation are second-round candidates.)

```bash
P="python -m wmworld.cli"; CFG=configs/pilot.yaml
$P generate --config $CFG --stage ablation --run-name ablation_N16 --override '{"candidates": 16}' --dry-run
$P generate --config $CFG --stage ablation --run-name ablation_N16 --override '{"candidates": 16}'
$P evaluate --config $CFG --run runs/ablation_N16
```

Cost guide (fp32, A100-40GB, from the dev benchmark): ~45 s per 512-token `llm_global` output at N=8; N=16 roughly
doubles proposal and scoring work; sentence units add discarded-tail tokens. 60 outputs per configuration ->
roughly 0.5–1.5 GPU-hours each.
