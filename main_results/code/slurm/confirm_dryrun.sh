set -e
P=runs/frozen_protocol_draft.json
python -m wmworld.cli make-keys --protocol $P --set confirmation --device cpu
for st in null_calibration null_test positive_test; do
  python -m wmworld.cli generate --protocol $P --stage $st --dry-run
  python - <<PY
import json; b=json.load(open("runs/$st/budget.json"))
est=b.get("estimates", b); lim=b.get("limits", {})
print("$st:", json.dumps({k: est[k] for k in est if k in ("n_outputs","expected_outputs","proposal_tokens","scoring_tokens","scoring_forward_calls","wall_hours","est_wall_hours","gpu_hours")}, default=str)[:400])
print("   keys:", list(b.keys())[:12])
PY
done
