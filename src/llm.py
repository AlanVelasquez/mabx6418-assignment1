"""Minimal OpenAI-compatible chat client using stdlib only (urllib).

Configured for the class endpoint and the reasoning-mode model. We send
chat_template_kwargs={"enable_thinking": false} so the model answers directly
with content and no thinking/reasoning tokens, which keeps parsing simple.
"""
import json
import time
import urllib.error
import urllib.request

from . import config
from . import prompt as prompt_mod


def _post(payload: dict):
    req = urllib.request.Request(
        config.LLM_BASE_URL + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {config.LLM_API_KEY}",
                 "Content-Type": "application/json"},
        method="POST")
    with urllib.request.urlopen(req, timeout=config.TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def chat(user_message: str) -> str:
    payload = {
        "model": config.LLM_MODEL,
        "messages": [{"role": "user", "content": user_message}],
        "max_tokens": config.MAX_TOKENS,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    last = None
    for attempt in range(config.RETRIES):
        try:
            data = _post(payload)
            content = data["choices"][0]["message"].get("content")
            return content or ""
        except (urllib.error.URLError, urllib.error.HTTPError, KeyError, Exception) as e:
            last = e
            if attempt < config.RETRIES - 1:
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"LLM request failed after {config.RETRIES} tries: {last}")


def classify(title: str, body: str, *, three_class: bool = False,
             include_emotion: bool = False):
    """Classify one review. Returns (sentiment, emotion)."""
    user_msg = prompt_mod.build_prompt(
        title, body, three_class=three_class, include_emotion=include_emotion)
    content = chat(user_msg)
    return prompt_mod.parse_response(content, include_emotion=include_emotion)
