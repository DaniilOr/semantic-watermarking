python -m pytest -q tests 2>&1 | tail -3
python - <<'PY'
import json
conf = {w for k in json.load(open("assets/keys_confirmation.json"))["keys"] for w in k["prefix"].split()}
dec = json.load(open("assets/steering_decoys.json"))["by_family"].get("random_words", [])
print("steering decoys:", [d["prefix"] for d in dec])
print("words shared with confirmation keys:", sorted({w for d in dec for w in d["prefix"].split()} & conf))
bank = {k["prefix"] for k in json.load(open("assets/keys_decoy_rw.json"))["keys"]}
print("confirmation phrases colliding with the decoy bank:", len({k["prefix"] for k in json.load(open("assets/keys_confirmation.json"))["keys"]} & bank))
PY
