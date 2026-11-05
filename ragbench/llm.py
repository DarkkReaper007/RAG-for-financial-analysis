"""Small cached chat client. Model prefixes: groq:<id>, ollama:<id>, otherwise NVIDIA. Keys come from the environment only
(GROQ_API_KEY, NVIDIA_API_KEY). Empty replies are never cached; 429/5xx are retried with backoff."""
import hashlib
import json
import os
import time
from pathlib import Path

import requests

from ipo_rag.keys import post_with_pool

CACHE = Path(__file__).resolve().parent / "out" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)
ENDPOINTS = {"groq": ("https://api.groq.com/openai/v1/chat/completions", "GROQ_API_KEY"),
             "nvidia": ("https://integrate.api.nvidia.com/v1/chat/completions", "NVIDIA_API_KEY")}


def chat(model, prompt, system=None, temperature=0.0, max_tokens=1500, retries=8, timeout=300):
    key = hashlib.sha1(json.dumps([model, system, prompt, temperature, max_tokens]).encode()).hexdigest()
    cp = CACHE / f"{key}.json"
    if cp.exists():
        return json.load(open(cp))["text"]
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    for i in range(retries):
        try:
            if model.startswith("ollama:"):
                r = requests.post("http://localhost:11434/api/chat", json={"model": model[7:], "messages": msgs, "stream": False,
                                  "options": {"temperature": temperature, "num_predict": max_tokens}}, timeout=timeout)
                r.raise_for_status()
                text = r.json()["message"]["content"].strip()
            else:
                prov, real = (model.split(":", 1) if model.startswith("groq:") else ("nvidia", model))
                url, env = ENDPOINTS[prov]
                r = post_with_pool(requests.post, env, url, {"model": real, "messages": msgs, "temperature": temperature, "max_tokens": max_tokens}, timeout)
                if r.status_code != 200:
                    time.sleep(min(60, 2 ** i * 3))
                    continue
                text = (r.json()["choices"][0]["message"].get("content") or "").strip()
            if text:
                json.dump({"text": text}, open(cp, "w"))
            return text
        except requests.RequestException:
            time.sleep(min(60, 2 ** i * 3))
    return ""
