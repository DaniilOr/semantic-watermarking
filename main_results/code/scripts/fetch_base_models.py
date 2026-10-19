"""Download the base (non-chat) generators for the SemStamp-setting check to /l and smoke-load them on CPU:
AbeHou/opt-1.3b-semstamp (SemStamp's released generator) meta-llama/Llama-3.2-3B and -1B (base).  Adds them to
assets/ss_models.json as opt13b_ss, llama3b_base, llama1b_base."""
import json, sys
from pathlib import Path
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, "/home/daniil.orel/wmworld/scripts")
from steered_semstamp import semstamp_prompt
CACHE = "/l/users/daniil.orel/hf_cache/hub"
M = json.loads(Path("assets/ss_models.json").read_text())
doc = json.loads(next(l for l in open("data/sp_bench/prompts.jsonl") if '"sp_realnews"' in l and '"test"' in l))
prompt = semstamp_prompt(doc["excerpt"] + doc["human_continuation"])
print("prompt:", repr(prompt))
for name, repo in (("opt13b_ss", "AbeHou/opt-1.3b-semstamp"), ("llama3b_base", "meta-llama/Llama-3.2-3B"), ("llama1b_base", "meta-llama/Llama-3.2-1B"),
                   ("qwen08b_base", "Qwen/Qwen3.5-0.8B-Base")):
    if sys.argv[1:] and name not in sys.argv[1:]:
        continue
    p = snapshot_download(repo, cache_dir=CACHE, allow_patterns=["*.json", "*.safetensors", "*.bin", "tokenizer*", "*.txt", "*.model"])
    from huggingface_hub import list_repo_files
    idx = Path(p, "model.safetensors.index.json")
    if idx.exists() and not all(Path(p, w).exists() for w in set(json.loads(idx.read_text())["weight_map"].values())):
        # AbeHou/opt-1.3b-semstamp ships a safetensors index without its shards (repo files: see below) next to the real
        # pytorch_model.bin; drop the dangling index locally so the .bin loads (SemStamp itself loads use_safetensors=False)
        print("  repo files:", list_repo_files(repo)); idx.unlink(); print("  removed dangling", idx.name)
    files = sorted(x.name for x in Path(p).iterdir())
    print(name, p, files, flush=True)
    tok = AutoTokenizer.from_pretrained(p)
    if (json.loads(Path(p, "config.json").read_text()).get("architectures") or [""])[0] == "Qwen3_5ForConditionalGeneration":
        from transformers import Qwen3_5ForCausalLM                    # as steered_semstamp.LM: the text model of the checkpoint
        model = Qwen3_5ForCausalLM.from_pretrained(p, dtype=torch.float32).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(p, dtype=torch.float32).eval()
    ids = tok(prompt, return_tensors="pt")["input_ids"]
    torch.manual_seed(0)
    out = model.generate(ids, max_new_tokens=40, do_sample=True, temperature=0.7, top_k=0, repetition_penalty=1.05)
    print("  arch", model.config.architectures, "bos/eos/pad", tok.bos_token_id, model.generation_config.eos_token_id, tok.pad_token,
          "chat_template", tok.chat_template is not None, "first ids", ids[0, :3].tolist())
    print("  sample:", repr(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)), flush=True)
    M[name] = {"path": p, "revision": Path(p).name, "repo": repo}
Path("assets/ss_models.json").write_text(json.dumps(M, indent=1))
print("assets/ss_models.json updated")
