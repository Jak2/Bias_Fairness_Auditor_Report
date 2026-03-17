"""
LLM Execution Engine — async, rate-limited, multi-provider.

Supports: Anthropic Claude, OpenAI, Ollama (local).
Each variant is run `runs_per_variant` times concurrently within the
semaphore limit to stay inside API quotas.
"""
from __future__ import annotations
import asyncio
import time
import logging
from typing import AsyncIterator
from auditor.report_models import LLMResponse, VariantPrompt
from config import get_settings

log = logging.getLogger(__name__)
cfg = get_settings()


async def _call_claude(prompt: str, model: str, client) -> tuple[str, int]:
    msg = await client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    text = msg.content[0].text
    tokens = msg.usage.input_tokens + msg.usage.output_tokens
    return text, tokens


async def _call_openai(prompt: str, model: str, client) -> tuple[str, int]:
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
    )
    text = resp.choices[0].message.content
    tokens = resp.usage.total_tokens
    return text, tokens


async def _call_ollama(prompt: str, model: str, base_url: str) -> tuple[str, int]:
    import httpx
    async with httpx.AsyncClient(timeout=cfg.llm_timeout) as http:
        resp = await http.post(
            f"{base_url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["response"]
        tokens = data.get("eval_count", len(text.split()))
        return text, tokens


def _get_client(llm: str):
    """Lazily create the appropriate client."""
    if llm == "claude":
        import anthropic
        return anthropic.AsyncAnthropic(api_key=cfg.anthropic_api_key)
    elif llm == "openai":
        import openai
        return openai.AsyncOpenAI(api_key=cfg.openai_api_key)
    return None   # ollama uses httpx directly


async def execute_variants(
    variants: list[VariantPrompt],
    runs_per_variant: int = 5,
    llm: str | None = None,
    model: str | None = None,
    on_progress: AsyncIterator | None = None,
) -> list[LLMResponse]:
    """
    Execute all variant prompts concurrently, returning all responses.

    Uses asyncio.Semaphore to cap concurrent calls at max_concurrent_calls.
    """
    llm = llm or cfg.default_llm
    model = model or cfg.default_model
    sem = asyncio.Semaphore(cfg.max_concurrent_calls)
    client = _get_client(llm)
    results: list[LLMResponse] = []
    lock = asyncio.Lock()
    completed = 0
    total = len(variants) * runs_per_variant

    async def _single_call(variant: VariantPrompt, run: int) -> None:
        nonlocal completed
        async with sem:
            t0 = time.monotonic()
            try:
                if llm == "claude":
                    text, tokens = await _call_claude(variant.rendered, model, client)
                elif llm == "openai":
                    text, tokens = await _call_openai(variant.rendered, model, client)
                else:
                    text, tokens = await _call_ollama(
                        variant.rendered, model, cfg.ollama_base_url
                    )
                latency = (time.monotonic() - t0) * 1000
                resp = LLMResponse(
                    variant_prompt=variant.rendered,
                    context=variant.context,
                    response_text=text,
                    latency_ms=round(latency, 1),
                    token_count=tokens,
                    run_number=run,
                    model=model,
                )
            except Exception as exc:
                log.warning("LLM call failed (run %d): %s", run, exc)
                latency = (time.monotonic() - t0) * 1000
                resp = LLMResponse(
                    variant_prompt=variant.rendered,
                    context=variant.context,
                    response_text="[ERROR]",
                    latency_ms=round(latency, 1),
                    token_count=0,
                    run_number=run,
                    model=model,
                )
            async with lock:
                results.append(resp)
                completed += 1
                log.debug("Progress: %d/%d", completed, total)

    tasks = [
        _single_call(v, r)
        for v in variants
        for r in range(1, runs_per_variant + 1)
    ]
    await asyncio.gather(*tasks)
    return results
