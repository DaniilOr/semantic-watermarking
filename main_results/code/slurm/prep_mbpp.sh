python -m wmworld.cli prepare-data --config configs/pilot.yaml --domains mbpp --append --budgets '{"mbpp": {"code": 400}}'
python - <<'PY'
import json
m = json.load(open("data/prepared/manifest.json"))
rows = [r for r in m["rows"] if r["domain"] == "mbpp"]
print("mbpp rows:", len(rows), "partitions:", {p: sum(1 for r in rows if r["partition"] == p) for p in set(r["partition"] for r in rows)},
      "reference-eligible:", sum(1 for r in rows if r["reference_eligible"]))
print("other domains rows:", {d: sum(1 for r in m["rows"] if r["domain"] == d) for d in ("writingprompts", "dolly", "c4_realnewslike")})
PY
