"""RAG 编排：文档入库 + 检索 + 生成（LLM 流式 / 检索摘要模式）。"""
from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import AsyncGenerator

from . import document_loader, vector_store as vs
from .chunker import chunk_document
from .config import UPLOAD_DIR, ensure_dirs
from .embeddings import embed_query
from .llm import LLMError, llm_configured, stream_chat

SYSTEM_TEMPLATE = (
    "你是「知识库问答助手」。请严格基于下面提供的资料片段回答用户的问题，"
    "不要编造资料中没有的信息；资料不足时明确说明。"
    "回答时优先引用片段中的关键事实，条理清晰、简洁。\n\n"
    "【资料片段】\n{context}\n\n"
    "请开始回答。"
)


def _summarize(text: str, limit: int = 130) -> str:
    t = re.sub(r"\s+", " ", text).strip()
    return t if len(t) <= limit else t[:limit] + "…"


def _build_context(refs: list[dict]) -> tuple[str, list]:
    """组装提示词上下文，返回 (context_str, 精简引用列表)。"""
    blocks, brief = [], []
    for i, r in enumerate(refs, start=1):
        heading = f"（{r['source']} · {r['heading']}）" if r.get("heading") else f"（{r['source']}）"
        blocks.append(f"[{i}]{heading}\n{r['text']}")
        brief.append(
            {
                "index": i,
                "source": r["source"],
                "heading": r.get("heading", ""),
                "score": r.get("score", 0),
                "snippet": _summarize(r["text"]),
            }
        )
    return "\n\n".join(blocks), brief


# ---------- 文档入库 ----------

def ingest_file(file_bytes: bytes, filename: str) -> dict:
    ensure_dirs()
    name = Path(filename).name
    ext = Path(name).suffix.lower()
    if ext not in document_loader.SUPPORTED_EXTS:
        raise document_loader.DocumentParseError(
            f"不支持的文件类型：{ext or '无扩展名'}（支持 pdf / md / txt / docx）"
        )
    # 防重名
    import uuid as _uuid

    saved = UPLOAD_DIR / f"{_uuid.uuid4().hex[:12]}{ext}"
    saved.write_bytes(file_bytes)

    sections = document_loader.parse_document(saved)
    records = chunk_document(sections, doc_id=0, source=name)
    doc_id = vs.add_document(name, len(file_bytes), records)
    return {"doc_id": doc_id, "name": name, "chunks": len(records)}


# ---------- 问答 ----------

async def answer_stream(
    conversation_id: int,
    question: str,
    top_k: int,
    doc_id: int | None = None,
    mode: str = "auto",  # auto | retrieve
) -> AsyncGenerator[dict, None]:
    """产出事件：{type:'token'|'done'|'error', ...}"""
    question = question.strip()
    if not question:
        yield {"type": "error", "message": "问题不能为空"}
        return

    try:
        query_vec = await asyncio.to_thread(embed_query, question)
    except Exception as e:  # noqa: BLE001
        yield {"type": "error", "message": f"问题向量化失败：{e}"}
        return

    refs = await asyncio.to_thread(vs.search, query_vec, top_k, doc_id)
    if not refs:
        yield {
            "type": "error",
            "message": "知识库中没有找到相关片段。请先上传文档，或换一个问法。",
        }
        return

    context, brief = _build_context(refs)
    has_llm, settings = llm_configured()
    use_llm = has_llm and mode != "retrieve"

    if not use_llm:
        # 检索摘要模式：透明展示召回片段
        head = f"🔍 未配置大模型 Key，以下为知识库检索到的 {len(refs)} 段相关片段：\n\n"
        body = []
        for r in refs:
            loc = f"{r['source']} · {r['heading']}" if r.get("heading") else r["source"]
            body.append(f"**片段 {r['doc_id']}**（{loc}，相似度 {r['score']:.2f}）：\n{r['text']}")
        yield {"type": "token", "content": head + "\n\n".join(body)}
        yield {"type": "done", "references": brief, "mode": "retrieve"}
        return

    history = vs.list_messages(conversation_id)[-6:]
    msgs: list[dict] = [{"role": "system", "content": SYSTEM_TEMPLATE.format(context=context)}]
    for m in history:
        if m["role"] in ("user", "assistant") and m["content"]:
            msgs.append({"role": m["role"], "content": m["content"][:2000]})
    msgs.append({"role": "user", "content": question})

    try:
        async for token in stream_chat(
            msgs,
            settings.get("llm_base_url", ""),
            settings.get("llm_api_key", ""),
            settings.get("llm_model", ""),
        ):
            yield {"type": "token", "content": token}
        yield {"type": "done", "references": brief, "mode": "llm"}
    except LLMError as e:
        yield {"type": "error", "message": str(e)}
