"""轻量 RAG 知识库问答系统 - FastAPI 入口。"""
from __future__ import annotations

import asyncio
import json
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .rag import pipeline, vector_store as vs
from .rag.config import FRONTEND_DIR, public_settings, save_settings

app = FastAPI(title="轻量 RAG 知识库问答系统", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


# ---------- 文档 ----------

@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(400, "文件内容为空")
    try:
        return await asyncio.to_thread(pipeline.ingest_file, data, file.filename or "unnamed")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, str(e)) from e


@app.get("/api/documents")
async def documents():
    docs = vs.list_documents()
    total = vs.count_chunks()
    return {"documents": docs, "total_chunks": total}


@app.delete("/api/documents/{doc_id}")
async def remove_document(doc_id: int):
    vs.delete_document(doc_id)
    return {"ok": True}


@app.get("/api/documents/{doc_id}/chunks")
async def document_chunks(doc_id: int):
    return {"chunks": vs.get_document_chunks(doc_id)}


# ---------- 会话 ----------

@app.get("/api/conversations")
async def conversations():
    return {"conversations": vs.list_conversations()}


@app.post("/api/conversations")
async def new_conversation():
    cid = vs.create_conversation("新对话")
    return {"conversation_id": cid}


@app.delete("/api/conversations/{conv_id}")
async def remove_conversation(conv_id: int):
    vs.delete_conversation(conv_id)
    return {"ok": True}


@app.get("/api/conversations/{conv_id}/messages")
async def conversation_messages(conv_id: int):
    return {"messages": vs.list_messages(conv_id)}


# ---------- 问答（SSE） ----------

@app.post("/api/chat/stream")
async def chat_stream(payload: dict):
    conv_id = payload.get("conversation_id")
    question = (payload.get("question") or "").strip()
    if not conv_id or not question:
        raise HTTPException(400, "缺少会话 ID 或问题")
    top_k = max(1, min(10, int(payload.get("top_k", 4))))
    doc_id = payload.get("doc_id")  # None 表示全部文档
    if doc_id is not None:
        doc_id = int(doc_id)
    mode = payload.get("mode", "auto")

    async def gen():
        # 先落库用户问题
        vs.add_message(conv_id, "user", question)
        vs.rename_conversation_if_untitled(conv_id, question)
        refs_cache = []
        answer_parts = []
        errored = False
        try:
            async for ev in pipeline.answer_stream(conv_id, question, top_k, doc_id, mode):
                if ev["type"] == "token":
                    answer_parts.append(ev["content"])
                    yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                elif ev["type"] == "done":
                    refs_cache = ev.get("references", [])
                    yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                elif ev["type"] == "error":
                    errored = True
                    yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        finally:
            if answer_parts and not errored:
                vs.add_message(conv_id, "assistant", "".join(answer_parts), refs_cache)
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ---------- 设置 ----------

@app.get("/api/settings")
async def get_settings():
    return public_settings()


@app.post("/api/settings")
async def post_settings(payload: dict):
    allowed = {k: payload.get(k) for k in ("llm_base_url", "llm_api_key", "llm_model", "top_k")}
    allowed = {k: v for k, v in allowed.items() if v is not None}
    s = save_settings(allowed)
    return {
        "has_llm": bool(s.get("llm_api_key", "").strip()),
        "llm_model": s.get("llm_model", ""),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=False)
