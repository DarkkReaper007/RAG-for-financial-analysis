"""Shared helpers for gold-set generation: manifest, page store, cached/retrying NVIDIA chat client."""
import hashlib, json, os, re, time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "goldgen" / "out"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "cache").mkdir(exist_ok=True)
API = "https://integrate.api.nvidia.com/v1/chat/completions"

ALIASES = {  # question must mention the company by one of these (corpus has many companies)
    "zomato": ["Zomato"], "nykaa": ["Nykaa", "FSN"], "swiggy": ["Swiggy"], "lic": ["LIC", "Life Insurance Corporation"],
    "ola_electric": ["Ola Electric", "Ola"], "hyundai_india": ["Hyundai"], "hdb_financial": ["HDB"],
    "lg_india": ["LG Electronics", "LG"], "delhivery": ["Delhivery"], "manipal_health": ["Manipal"],
    "jio_platforms": ["Jio"], "nse": ["NSE", "National Stock Exchange"]}


def manifest():
    return {d["id"]: d for d in json.load(open(ROOT / "data" / "manifest.json"))["docs"]}


def norm(s):
    return re.sub(r"\s+", " ", s.replace("­", "").replace("’", "'").replace("“", '"').replace("”", '"')).strip().lower()


def read_jsonl(p):
    return [json.loads(l) for l in open(p)] if Path(p).exists() else []


def write_jsonl(p, rows):
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


GROQ_API = "https://api.groq.com/openai/v1/chat/completions"


def _key(model, system, prompt, temperature, max_tokens):
    return hashlib.sha1(json.dumps([model, system, prompt, temperature, max_tokens]).encode()).hexdigest()


def cached_only(model, prompt, system=None, temperature=0.2, max_tokens=500):
    """Cached reply or None; never calls the API."""
    cp = OUT / "cache" / f"{_key(model, system, prompt, temperature, max_tokens)}.json"
    return json.load(open(cp))["text"] if cp.exists() else None


def chat_fb(primary, fallback, prompt, temperature=0, max_tokens=4000, fb_tokens=800):
    """Use the primary model's cached reply if it exists, otherwise call the fallback. -> (text, model_used)."""
    hit = cached_only(primary, prompt, None, temperature, max_tokens)
    if hit:
        return hit, primary
    return chat(fallback, prompt, temperature=temperature, max_tokens=fb_tokens), fallback


def chat(model, prompt, system=None, temperature=0.2, max_tokens=500, retries=8):
    """Cached on (model, system, prompt, temperature). Retries with backoff on 429/5xx. Key from NVIDIA_API_KEY only."""
    cp = OUT / "cache" / f"{_key(model, system, prompt, temperature, max_tokens)}.json"
    if cp.exists():
        return json.load(open(cp))["text"]
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    for i in range(retries):
        try:
            url, keyenv, real = (GROQ_API, "GROQ_API_KEY", model[5:]) if model.startswith("groq:") else (API, "NVIDIA_API_KEY", model)
            r = requests.post(url, headers={"Authorization": "Bearer " + os.environ[keyenv]},
                              json={"model": real, "messages": msgs, "temperature": temperature, "max_tokens": max_tokens}, timeout=300)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(90, float(r.headers.get("retry-after") or 0) or 2 ** i * 3))
                continue
            r.raise_for_status()
            text = (r.json()["choices"][0]["message"].get("content") or "").strip()
            if text:                       # never cache empty replies (reasoning models can exhaust max_tokens)
                json.dump({"text": text}, open(cp, "w"))
            return text
        except requests.RequestException:
            time.sleep(min(60, 2 ** i * 3))
    return ""


def parse_json(text):
    """Extract the first JSON object from a model reply (tolerates ``` fences and chatter)."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


DISPLAY = {"zomato": "Zomato", "nykaa": "FSN E-Commerce Ventures (Nykaa)", "swiggy": "Swiggy", "lic": "LIC", "ola_electric": "Ola Electric",
           "hyundai_india": "Hyundai Motor India", "hdb_financial": "HDB Financial Services", "lg_india": "LG Electronics India",
           "delhivery": "Delhivery", "manipal_health": "Manipal Health Enterprises", "jio_platforms": "Jio Platforms", "nse": "NSE"}
