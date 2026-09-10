"""LLM 调用封装：OpenAI 兼容接口，SSE 流式输出；未配置 Key 时返回 None。"""
from __future__ import annotations

import json
from typing import AsyncGenerator

import httpx

from .config import load_settings


class LLMError(Exception):
    pass


async def stream_chat(
    messages: list[dict],
    base_url: str,
    api_key: str,
    model: str,
) -> AsyncGenerator[str, None]:
    """逐 token 产出内容。错误时抛出 LLMError。"""
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "temperature": 0.3,
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=20.0)) as client:
        try:
            async with client.stream(
                "POST", url, json=payload, headers=headers
            ) as resp:
                if resp.status_code != 200:
                    body = (await resp.aread()).decode("utf-8", "ignore")[:300]
                    raise LLMError(f"LLM 接口返回 {resp.status_code}：{body}")
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    delta = obj.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        yield content
        except httpx.HTTPError as e:
            raise LLMError(f"连接 LLM 服务失败：{e}") from e


def llm_configured() -> tuple[bool, dict]:
    """返回 (是否已配置, 设置副本)。"""
    s = load_settings()
    ok = bool(s.get("llm_api_key", "").strip())
    return ok, s
