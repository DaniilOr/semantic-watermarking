import sys

import numpy as np
import torch

GENERATOR_ID = "Qwen/Qwen3.5-0.8B"
GENERATOR_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
REFERENCE_ID = "meta-llama/Llama-3.2-1B-Instruct"
REFERENCE_REVISION = "9213176726f574b556790deb65791e0c5aa438b6"


def _device(device):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


class Generator:
    def __init__(self, model_id=GENERATOR_ID, revision=GENERATOR_REVISION, device="auto", dtype=torch.float32, cache_dir=None,
                 torch_kernels=True):
        if torch_kernels and "transformers.models.qwen3_5.modeling_qwen3_5" not in sys.modules:
            sys.modules["fla"] = None
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = _device(device)
        kw = {"cache_dir": cache_dir} if cache_dir else {}
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, **kw)
        if "qwen3.5" in model_id.lower():
            from transformers import Qwen3_5ForCausalLM
            model, info = Qwen3_5ForCausalLM.from_pretrained(model_id, revision=revision, dtype=dtype, output_loading_info=True, **kw)
            unexpected = [k for k in info.get("unexpected_keys", []) if not (k.startswith("mtp") or "visual" in k)]
            if info.get("missing_keys") or info.get("mismatched_keys") or unexpected:
                raise RuntimeError("checkpoint did not load cleanly into the text-only model")
        else:
            model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=dtype, **kw)
        self.model = model.to(self.device).eval()
        eos = set()
        for name in ("<|im_end|>", "<|endoftext|>"):
            tid = self.tokenizer.convert_tokens_to_ids(name)
            if isinstance(tid, int) and tid >= 0 and tid != self.tokenizer.unk_token_id:
                eos.add(tid)
        for src in (self.tokenizer.eos_token_id, getattr(self.model.config, "eos_token_id", None)):
            if isinstance(src, int):
                eos.add(src)
            elif isinstance(src, (list, tuple)):
                eos |= {int(x) for x in src}
        self.eos_ids = sorted(eos)

    def encode(self, text):
        return [int(t) for t in self.tokenizer(text, add_special_tokens=False)["input_ids"]]

    def decode(self, ids):
        ids = [int(i) for i in ids if int(i) not in self.eos_ids]
        return self.tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    def chat_ids(self, system, user):
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return self.encode(text)

    @torch.no_grad()
    def prefill(self, ids):
        from transformers import DynamicCache
        cache = DynamicCache(config=self.model.config)
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        out = self.model(input_ids=x, past_key_values=cache, use_cache=True, logits_to_keep=1)
        return out.past_key_values, out.logits[:, -1, :].float()

    @torch.no_grad()
    def step(self, token, cache):
        x = torch.tensor([[token]], device=self.device)
        out = self.model(input_ids=x, past_key_values=cache, use_cache=True, logits_to_keep=1)
        return out.logits[:, -1, :].float()

    def next_token_probs(self, logits, sampling, suppress_eos):
        z = logits.clone()
        if suppress_eos:
            z[:, self.eos_ids] = -float("inf")
        if sampling.temperature <= 0:
            p = torch.zeros_like(z)
            p.scatter_(1, z.argmax(dim=1, keepdim=True), 1.0)
            return p[0]
        z = z / sampling.temperature
        if sampling.top_k and 0 < sampling.top_k < z.shape[1]:
            kth = torch.topk(z, sampling.top_k, dim=1).values[:, -1:]
            z = torch.where(z >= kth, z, torch.full_like(z, -float("inf")))
        p = torch.softmax(z, dim=1)
        if 0 < sampling.top_p < 1.0:
            ps, idx = torch.sort(p, dim=1, descending=True)
            keep = (torch.cumsum(ps, dim=1) - ps) < sampling.top_p
            keep[:, 0] = True
            mask = torch.zeros_like(p, dtype=torch.bool).scatter_(1, idx, keep)
            p = torch.where(mask, p, torch.zeros_like(p))
            p = p / p.sum(dim=1, keepdim=True)
        return p[0]


class Stepper:
    def __init__(self, ref, ctx_ids):
        self.ref = ref
        self.ctx = list(ctx_ids)
        self.ids = []
        self.cache = None
        self.logits = None

    @torch.no_grad()
    def logprobs_after(self, text):
        new = self.ref.encode(text) if text else []
        model, device = self.ref.model, self.ref.device
        if self.cache is None:
            out = model(input_ids=torch.tensor([self.ctx + new], device=device), use_cache=True)
            self.cache, self.logits, self.ids = out.past_key_values, out.logits[0, -1, :].float(), new
            return torch.log_softmax(self.logits, dim=-1)
        n = 0
        while n < min(len(new), len(self.ids)) and new[n] == self.ids[n]:
            n += 1
        if n == len(new) and len(new) == len(self.ids):
            return torch.log_softmax(self.logits, dim=-1)
        self.cache.crop(len(self.ctx) + n)
        tail = new[n:] or [self.ctx[-1]]
        out = model(input_ids=torch.tensor([tail], device=device), past_key_values=self.cache, use_cache=True)
        self.cache, self.logits, self.ids = out.past_key_values, out.logits[0, -1, :].float(), new
        return torch.log_softmax(self.logits, dim=-1)


class Reference:
    def __init__(self, model_id=REFERENCE_ID, revision=REFERENCE_REVISION, device="auto", dtype=torch.float32,
                 max_input_tokens=4096, cache_dir=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = _device(device)
        kw = {"cache_dir": cache_dir} if cache_dir else {}
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, **kw)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=dtype, **kw).to(self.device).eval()
        self.max_input_tokens = int(max_input_tokens)
        self.vocab_size = int(self.model.get_output_embeddings().weight.shape[0])
        self.pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

    def encode(self, text):
        return [int(x) for x in self.tokenizer(text, add_special_tokens=False)["input_ids"]]

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    def chat_ids(self, system, user):
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            messages = [{"role": "user", "content": f"{system}\n\n{user}"}]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self.encode(text)

    def stepper(self, ctx_ids):
        return Stepper(self, ctx_ids)

    def token_map(self, generator):
        tok = generator.tokenizer
        n = len(tok)
        special = {int(x) for x in (tok.all_special_ids or [])}
        out = np.full(n, -1, dtype=np.int64)
        for start in range(0, n, 8192):
            ids = list(range(start, min(start + 8192, n)))
            strings = tok.batch_decode([[i] for i in ids], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            for i, enc in zip(ids, self.tokenizer(strings, add_special_tokens=False)["input_ids"]):
                if enc and i not in special:
                    out[i] = int(enc[0])
        return out

    @torch.no_grad()
    def target_logprobs(self, ctx_ids, targets, n_max):
        c = len(ctx_ids)
        rows, length = len(targets), c + max(len(t) for t in targets)
        input_ids = torch.full((rows, length), self.pad_id, dtype=torch.long)
        attention = torch.zeros((rows, length), dtype=torch.long)
        for r, t in enumerate(targets):
            seq = ctx_ids + t
            input_ids[r, :len(seq)] = torch.tensor(seq, dtype=torch.long)
            attention[r, :len(seq)] = 1
        logits = self.model(input_ids=input_ids.to(self.device), attention_mask=attention.to(self.device), use_cache=False).logits
        out = torch.zeros((rows, n_max, logits.shape[-1]), dtype=torch.float32, device=self.device)
        for r, t in enumerate(targets):
            out[r, :len(t)] = torch.log_softmax(logits[r, c - 1:c - 1 + len(t)].float(), dim=-1)
        return out
