set -e
python -m wmworld.cli generate --config configs/pilot.yaml --stage conf_null --run-name null_calibration --dry-run
python -m wmworld.cli generate --config configs/pilot.yaml --stage conf_pos --dry-run
python -m wmworld.cli generate --config configs/pilot.yaml --stage code --dry-run
python - <<'PY'
import json
for st in ("null_calibration", "conf_pos", "code"):
    b = json.load(open(f"runs/{st}/budget.json")); e = b["enumeration"]
    print(st, {k: e[k] for k in e if k in ("n_items", "arms", "n_prompts", "n_keys")}, "limits", b["limits"])
PY
