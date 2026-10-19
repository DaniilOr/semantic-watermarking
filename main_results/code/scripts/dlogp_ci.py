import json, sys
for r in sys.argv[1:]:
    try: a = json.load(open(f"{r}/eval/results.json"))["per_arm"]["poe_steer"]["quality_proxy"]["paired_delta_vs_base"]
    except Exception as e: print(r, "n/a"); continue
    print(f"{r.split('/')[-1]:28s} d_logp={a['mean']:+.3f}  95% CI [{a['ci95'][0]:+.3f}, {a['ci95'][1]:+.3f}]  n={a['n']}")
