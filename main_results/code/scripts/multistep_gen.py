"""Multi-step generation: the same response produced in K sequential steps instead of one long call.

Each step continues the assistant turn from the text written so far, but the adaptive controller and the reference
model's state RESTART at every step, exactly as they would if a service generated a long document chunk by chunk
(or turn by turn in a chat).  This is the variable under test: in one long call the controller stops pushing once
the text clears its target, so the tail is unmarked; restarting it per step should keep the mark accumulating.

Writes two run directories in the standard format so the existing detectors work unchanged:
  runs/<name>_turns  - one record per step  (how detectable is a single chunk?)
  runs/<name>_full   - one record per document (does the mark accumulate over the whole thing?)

Usage: python scripts/multistep_gen.py --config configs/pilot.yaml --name multistep --steps 3 --step-tokens 256 --beta 30
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wmworld.config import load_config  # noqa: E402
from wmworld.generation import GENERATOR_SYSTEM, SamplingParams, poe_block_ids  # noqa: E402
from wmworld.keys import assign_keys  # noqa: E402
from wmworld.records import JsonlWriter, write_json  # noqa: E402
from wmworld.reference import build_reference  # noqa: E402
from wmworld.stages import build_backend, keyspec_with_decoys, load_key_records, stage_prompts  # noqa: E402
from wmworld.textnorm import normalize_text, seed_from, text_hash  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", default="ablation_rw")
    ap.add_argument("--name", default="multistep")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--step-tokens", type=int, default=256)
    ap.add_argument("--beta", type=float, default=30.0)
    ap.add_argument("--clip", type=float, default=0.5)
    ap.add_argument("--t-target", type=float, default=4.0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.validate(require_revision=True)
    backend = build_backend(cfg, args.device)
    ref = build_reference(cfg)
    tmap = ref.token_map(backend, cache_path=cfg.path_of("assets") / f"token_map_{Path(ref.cfg.model_id).name}.npz")
    prompts = stage_prompts(cfg, args.stage)
    keys = load_key_records(cfg, cfg.stage(args.stage)["keys"])
    assignment = assign_keys([p.prompt_id for p in prompts], [k.key_id for k in keys], cfg.seed, args.stage)
    kmap = {k.key_id: k for k in keys}
    sampling = SamplingParams(float(cfg["generation"]["temperature"]), float(cfg["generation"]["top_p"]),
                              int(cfg["generation"].get("top_k", 0)))
    runs = {}
    for suffix in ("turns", "full"):
        d = cfg.path_of("runs") / f"{args.name}_{suffix}"
        d.mkdir(parents=True, exist_ok=True)
        runs[suffix] = (d, JsonlWriter(d / "generations.jsonl"))
        write_json(d / "manifest.json", {"stage": f"multistep_{suffix}", "partition": cfg.stage(args.stage)["partition"],
                                         "arms": ["base", "wm_multi"], "seeds": [0], "key_set": cfg.stage(args.stage)["keys"],
                                         "steps": args.steps, "step_tokens": args.step_tokens, "beta": args.beta,
                                         "reference": ref.info, "config_hash": cfg.content_hash(),
                                         "note": "each step continues the assistant turn; controller and reference state restart per step"})
    t0 = time.time()
    for i, p in enumerate(prompts):
        key = kmap[assignment[p.prompt_id]]
        ks = keyspec_with_decoys(cfg, key)
        block = poe_block_ids(backend, ks, True, False)
        base_ctx = backend.chat_prefix_ids(GENERATOR_SYSTEM, p.prompt)
        for arm, beta in (("base", 0.0), ("wm_multi", args.beta)):
            acc = ""
            for step in range(args.steps):
                ctx = base_ctx + (backend.encode_text(acc) if acc else [])
                out = backend.sample_poe_cross(
                    ctx,
                    ref.stepper(ref.chat_prefix_ids(ks.system_text(False), ks.user_message)),
                    ref.stepper(ref.chat_prefix_ids(ks.system_text(True), ks.user_message)),
                    tmap, args.step_tokens, sampling, seed_from("multistep", cfg.seed, p.prompt_id, arm, step),
                    args.step_tokens,                      # EOS suppressed: every step is exactly step_tokens long
                    beta, args.clip, "entropy", 2.0, 0.2, False, block, "tstat", args.t_target, 0.0, 4)
                piece = normalize_text(backend.decode(out["token_ids"]))
                acc += piece
                rec = {"id": f"{p.prompt_id}.{arm}.s{step}", "prompt_id": p.prompt_id, "group_id": p.group_id,
                       "domain": p.domain, "arm": arm, "key_id": key.key_id, "assigned_key_id": key.key_id, "seed": 0,
                       "step": step, "text": piece, "text_hash": text_hash(piece),
                       "visible_tokens": len(backend.encode_text(piece)), "status": "ok", "stop_reason": "step_budget",
                       "ends_mid_sentence": False, "config_hash": cfg.content_hash(),
                       "cost": {"mean_beta_t": (sum(out["beta_t"]) / len(out["beta_t"])) if out["beta_t"] else 0.0,
                                "final_t_run": out["t_run"][-1] if out["t_run"] else None}}
                runs["turns"][1].write(rec)
            full = {"id": f"{p.prompt_id}.{arm}", "prompt_id": p.prompt_id, "group_id": p.group_id, "domain": p.domain,
                    "arm": arm, "key_id": key.key_id, "assigned_key_id": key.key_id, "seed": 0, "text": acc,
                    "text_hash": text_hash(acc), "visible_tokens": len(backend.encode_text(acc)), "status": "ok",
                    "stop_reason": "steps_done", "ends_mid_sentence": False, "config_hash": cfg.content_hash()}
            runs["full"][1].write(full)
        if (i + 1) % 5 == 0:
            print(f"[multistep] {i + 1}/{len(prompts)} prompts ({time.time() - t0:.0f}s)", flush=True)
    for d, w in runs.values():
        w.close()
        print(f"[multistep] wrote {d}", flush=True)


if __name__ == "__main__":
    main()
