echo "### compare"; python scripts/poe_compare.py runs/conf_pos runs/code
echo "### frozen-calibration flags and per-domain"; python - <<'PY'
import json
for run in ("runs/conf_pos", "runs/code"):
    r = json.load(open(f"{run}/eval/results.json"))
    print(run, "calibration:", r.get("calibration_id"), (r.get("calibration") or {}).get("label") if isinstance(r.get("calibration"), dict) else r.get("calibration"))
    for arm, a in r["per_arm"].items():
        fl = {k: v for k, v in a.items() if k.startswith("flag_rate")}
        print("  ", arm, {k: (v.get("k"), v.get("n"), round(v.get("rate", 0), 3)) for k, v in fl.items() if isinstance(v, dict)})
    pd = r.get("per_domain") or {}
    for dom, d in pd.items():
        print("   domain", dom, {arm: (x.get("flag_rate") if isinstance(x, dict) else x) for arm, x in (d.items() if isinstance(d, dict) else [])})
PY
echo "### attribution (same-family bank)"; python scripts/rank_false_attribution.py --bank decoy_rw runs/conf_pos runs/code
echo "### code100b pass rate"; python -m wmworld.cli code-eval --config configs/pilot.yaml --run runs/code100b --timeout 10 2>&1 | grep -E "code-eval\] (base|wm_)"
python scripts/quick_shift.py runs/code100b 2>/dev/null | grep -E "^base|^wm_"
