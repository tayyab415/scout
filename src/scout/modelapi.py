"""Meta Model API: https://api.meta.ai/v1, an OpenAI-compatible chat completions endpoint.

One key (`MODEL_API_KEY`) covers the whole Muse line and SAM:

  muse-spark-1.3   Muse Spark, latest (default brain)
  muse-spark-1.2   Muse Spark, previous
  muse-spark-1.1   Muse Spark, original and smallest
  muse-wonder-1.0  Muse Wonder (preview), the most capable model
  sam-3.1          Segment Anything 3.1, same base URL and auth

`SCOUT_MODEL` overrides the default model, `MODEL_API_BASE_URL` the base URL, and `MODEL_API_KEY_FILE` reads the
key from a file instead of the environment. Docs: https://dev.meta.ai/docs/models
"""
from __future__ import annotations

import os
import time

BASE_URL = os.environ.get("MODEL_API_BASE_URL", "https://api.meta.ai/v1").rstrip("/")
DEFAULT_MODEL = os.environ.get("SCOUT_MODEL", "muse-spark-1.3")


def api_key(key_name: str = "MODEL_API_KEY") -> str | None:
    """The key from the environment, a file, or the SCOUT_LLM_* aliases the old provider config used."""
    key = os.environ.get(key_name) or os.environ.get("SCOUT_LLM_API_KEY")
    if not key:
        path = os.environ.get(f"{key_name}_FILE") or os.environ.get("SCOUT_LLM_API_KEY_FILE")
        if path and os.path.exists(os.path.expanduser(path)):
            key = open(os.path.expanduser(path)).read().strip()
    return key


def chat(system: str, user: str, model: str | None = None, max_tokens: int = 3000, temperature: float = 0.0,
         retries: int = 3, base_url: str | None = None, key_name: str = "MODEL_API_KEY") -> tuple[str, dict]:
    """One chat completion. Returns (text, info with latency, token counts and the model that answered)."""
    import requests
    model = model or DEFAULT_MODEL
    url = (base_url or BASE_URL).rstrip("/") + "/chat/completions"
    key = api_key(key_name)
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    body = {"model": model, "temperature": temperature, "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    err = "Model API call failed"
    for k in range(retries + 1):
        t = time.time()
        try:
            r = requests.post(url, json=body, headers=headers, timeout=180)
            r.raise_for_status()
            j = r.json()
            u = j.get("usage", {})
            return j["choices"][0]["message"]["content"], {
                "latency_s": round(time.time() - t, 2), "in_tokens": u.get("prompt_tokens"),
                "out_tokens": u.get("completion_tokens"), "model": model}
        except Exception as e:  # network or throttling: the caller decides what to do
            err = f"{type(e).__name__}: {str(e)[:200]}"
            if k < retries:
                time.sleep(2)
    raise RuntimeError(err)
