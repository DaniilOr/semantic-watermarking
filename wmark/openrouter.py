import json
import os
import time
from collections import Counter
from pathlib import Path

import requests

URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemma-4-31b-it"


def api_key():
    k = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not k:
        p = Path.home() / ".config" / "wmworld" / "openrouter.key"
        k = p.read_text().strip() if p.exists() else ""
    if not k:
        raise SystemExit("no API key: set OPENROUTER_API_KEY")
    return k


def clip_words(text, n):
    w = text.split()
    return text if len(w) <= n else " ".join(w[:n]) + " ..."


def parse_json(text):
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):]
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b <= a:
        return None
    try:
        return json.loads(text[a:b + 1])
    except json.JSONDecodeError:
        return None


def slug(model):
    return model.replace("/", "_").replace(":", "_")


class Client:
    def __init__(self, model=DEFAULT_MODEL, key=None):
        self.model, self.key = model, key or api_key()
        self.structured, self.no_reasoning = True, True
        self.usage = Counter()

    def chat(self, messages, schema=None, accept=None, temperature=0.0, max_tokens=1500, timeout=180):
        body = {"model": self.model, "temperature": temperature, "seed": 0, "max_tokens": max_tokens, "messages": messages}
        for attempt in range(6):
            req = dict(body)
            if schema is not None and self.structured:
                req["response_format"] = {"type": "json_schema", "json_schema": schema}
            if self.no_reasoning:
                req["reasoning"] = {"enabled": False}
            try:
                r = requests.post(URL, json=req, timeout=timeout, headers={"Authorization": f"Bearer {self.key}"})
            except requests.RequestException:
                time.sleep(min(60, 2 ** attempt))
                continue
            if r.status_code == 200:
                d = r.json()
                if "error" in d:
                    time.sleep(min(60, 2 ** attempt))
                    continue
                u = d.get("usage") or {}
                self.usage["prompt_tokens"] += int(u.get("prompt_tokens") or 0)
                self.usage["completion_tokens"] += int(u.get("completion_tokens") or 0)
                content = d["choices"][0]["message"].get("content") or ""
                out = content if accept is None else accept(parse_json(content))
                if out:
                    return out
                continue
            txt = r.text[:300].lower()
            if r.status_code in (400, 404, 422) and self.no_reasoning and "reasoning" in txt:
                self.no_reasoning = False
                continue
            if r.status_code in (400, 404, 422) and self.structured and any(s in txt for s in ("response_format", "schema", "structured")):
                self.structured = False
                continue
            if r.status_code in (401, 402, 403):
                raise SystemExit(f"API refused the request (HTTP {r.status_code})")
            time.sleep(min(60, 2 ** attempt))
        return None
