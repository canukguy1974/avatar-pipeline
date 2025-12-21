# app/llm_stream.py
import os, asyncio
from typing import AsyncIterator, Optional
from openai import AsyncOpenAI

"""
Streaming LLM -> yields clause-sized chunks ideal for realtime TTS.
Works with OpenAI (gpt-4o, gpt-4o-mini) or OpenRouter (OpenAI-compatible).
ENV:
  OPENAI_API_KEY=...           # or OPENROUTER_API_KEY for OpenRouter
  OPENAI_BASE_URL=...          # set to https://openrouter.ai/api/v1 for OpenRouter
  OPENAI_MODEL=gpt-4o-mini     # default if not provided
"""

def _make_client() -> AsyncOpenAI:
    base_url = os.environ.get("OPENAI_BASE_URL")
    api_key  = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY or OPENROUTER_API_KEY")
    return AsyncOpenAI(api_key=api_key, base_url=base_url)

async def _token_stream(
    prompt: str,
    system: Optional[str],
    model: str,
    temperature: float,
) -> AsyncIterator[str]:
    client = _make_client()
    messages = [{"role": "user", "content": prompt}]
    if system:
        messages.insert(0, {"role": "system", "content": system})

    stream = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        stream=True,
    )
    async for event in stream:
        delta = event.choices[0].delta
        if delta and (txt := (delta.content or "")):
            yield txt

async def gpt_text_stream(
    prompt: str,
    system: Optional[str] = None,
    model: str = os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
    temperature: float = 0.6,
    min_chars: int = 24,
    delimiters: str = ".?!,;:—\n",
    max_lag_ms: int = 450,
) -> AsyncIterator[str]:
    """
    Buffer tiny tokens into readable clauses.
    Flush when: punctuation, buffer long enough, or lag threshold exceeded.
    """
    buf = []
    chars = 0
    loop = asyncio.get_event_loop()
    last_flush = loop.time()

    async for tok in _token_stream(prompt, system, model, temperature):
        buf.append(tok)
        chars += len(tok)
        now = loop.time()

        flush = False
        if tok and tok[-1] in delimiters and chars >= 8:
            flush = True
        elif chars >= min_chars:
            flush = True
        elif (now - last_flush) * 1000 >= max_lag_ms and chars > 0:
            flush = True

        if flush:
            out = "".join(buf).strip()
            if out:
                yield out
            buf.clear()
            chars = 0
            last_flush = now

    if buf:
        out = "".join(buf).strip()
        if out:
            yield out
