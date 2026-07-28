"""
rag/generate.py

Turns retrieved context into a written answer.

Provides claude_generator() backed by the Anthropic SDK. The SDK is imported
lazily so importing this module never requires the package or an API key. Set
ANTHROPIC_API_KEY in the environment to use it.
"""

from __future__ import annotations


def claude_generator(model: str = "claude-sonnet-4-6", max_tokens: int = 800, temperature: float = 0.2):
    """Return a `generator(prompt) -> str` that calls Claude. The SDK is only
    imported/initialized when the returned function is first invoked."""
    client = {"c": None}

    def generate(prompt: str) -> str:
        if client["c"] is None:
            import anthropic
            client["c"] = anthropic.Anthropic()
        msg = client["c"].messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        # concatenate text blocks
        return "".join(getattr(b, "text", "") for b in msg.content)

    return generate


def default_generator(model: str = "claude-haiku-4-5-20251001", max_tokens: int = 700):
    """Best available generator, or None.

    * ANTHROPIC_API_KEY unset  -> None (pipeline returns extractive answers).
    * anthropic SDK installed  -> claude_generator().
    * SDK missing              -> minimal urllib client (no extra dependency).
    """
    import os
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic  # noqa: F401
        return claude_generator(model=model, max_tokens=max_tokens)
    except ImportError:
        pass

    def generate(prompt: str) -> str:
        import json
        import urllib.request
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps({
                "model": model, "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }).encode(),
            headers={
                "Content-Type": "application/json",
                "x-api-key": os.environ["ANTHROPIC_API_KEY"],
                "anthropic-version": "2023-06-01",
            })
        with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
            data = json.loads(resp.read().decode())
        return "".join(b.get("text", "") for b in data.get("content", []))

    return generate
