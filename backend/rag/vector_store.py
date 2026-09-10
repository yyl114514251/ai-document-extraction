"""Chroma 向量库封装 + SQLite 元数据存储。"""
from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path

import chromadb

from .config import CHROMA_DIR, DB_PATH, ensure_dirs
from .embeddings import embed_texts

COLLECTION = "rag_kb"

_CLIENT = None
_COLL = None


def _get_collection():
    global _CLIENT, _COLL
    if _COLL is None:
        ensure_dirs()
        _CLIENT = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _COLL = _CLIENT.get_or_create_collection(
            COLLECTION, metadata={"hnsw:space": "cosine"}
        )
    return _COLL


def _db() -> sqlite3.Connection:
    ensure_dirs()
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS documents(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            size INTEGER DEFAULT 0,
            chunks INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now','localtime'))
        )"""
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now','localtime'))
        )"""
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS messages(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conv_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            refs TEXT DEFAULT '[]',
            created_at TEXT DEFAULT (datetime('now','localtime'))
        )"""
    )
    conn.commit()
    return conn


# ---------- 文档 ----------

def add_document(name: str, size: int, records: list[dict]) -> int:
    """入库：写入 chroma + 元数据表，返回 doc_id。"""
    coll = _get_collection()
    conn = _db()
    cur = conn.execute(
        "INSERT INTO documents(name, size, chunks) VALUES(?,?,?)",
        (name, size, len(records)),
    )
    doc_id = cur.lastrowid
    conn.commit()

    if records:
        coll.add(
            ids=[f"{doc_id}-{r['seq']}" for r in records],
            embeddings=embed_texts([r["text"] for r in records]),
            documents=[r["text"] for r in records],
            metadatas=[
                {
                    "doc_id": str(doc_id),
                    "source": r["source"],
                    "heading": r["heading"],
                    "seq": r["seq"],
                }
                for r in records
            ],
        )
    return doc_id


def list_documents():
    conn = _db()
    rows = conn.execute(
        "SELECT id, name, size, chunks, created_at FROM documents ORDER BY id DESC"
    ).fetchall()
    return [dict(r) for r in rows]


def delete_document(doc_id: int):
    coll = _get_collection()
    try:
        coll.delete(where={"doc_id": str(doc_id)})
    except Exception:
        pass
    conn = _db()
    conn.execute("DELETE FROM documents WHERE id=?", (doc_id,))
    conn.commit()


def count_chunks() -> int:
    coll = _get_collection()
    try:
        return coll.count()
    except Exception:
        return 0


def get_document_chunks(doc_id: int, limit: int = 200):
    """返回文档的切片预览（按 seq 排序）。"""
    coll = _get_collection()
    try:
        res = coll.get(
            where={"doc_id": str(doc_id)},
            include=["documents", "metadatas"],
            limit=limit,
        )
    except Exception:
        return []
    out = []
    for text, meta in zip(res.get("documents", []) or [], res.get("metadatas", []) or []):
        out.append(
            {
                "seq": meta.get("seq", 0),
                "heading": meta.get("heading", ""),
                "text": text,
            }
        )
    out.sort(key=lambda x: x["seq"])
    return out


def search(query_embedding, top_k: int, doc_id: int | None = None):
    coll = _get_collection()
    where = {"doc_id": str(doc_id)} if doc_id is not None else None
    res = coll.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    out = []
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    for text, meta, dist in zip(docs, metas, dists):
        score = max(0.0, 1.0 - float(dist))  # cosine 距离 -> 相似度
        out.append(
            {
                "text": text,
                "source": meta.get("source", ""),
                "heading": meta.get("heading", ""),
                "doc_id": meta.get("doc_id", ""),
                "score": round(score, 4),
            }
        )
    return out


# ---------- 会话与消息 ----------

def create_conversation(title: str) -> int:
    conn = _db()
    cur = conn.execute("INSERT INTO conversations(title) VALUES(?)", (title,))
    conn.commit()
    return cur.lastrowid


def list_conversations():
    conn = _db()
    rows = conn.execute(
        "SELECT id, title, created_at FROM conversations ORDER BY id DESC"
    ).fetchall()
    return [dict(r) for r in rows]


def delete_conversation(conv_id: int):
    conn = _db()
    conn.execute("DELETE FROM conversations WHERE id=?", (conv_id,))
    conn.execute("DELETE FROM messages WHERE conv_id=?", (conv_id,))
    conn.commit()


def list_messages(conv_id: int):
    conn = _db()
    rows = conn.execute(
        "SELECT id, role, content, refs, created_at FROM messages "
        "WHERE conv_id=? ORDER BY id",
        (conv_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def add_message(conv_id: int, role: str, content: str, refs: list | None = None):
    conn = _db()
    conn.execute(
        "INSERT INTO messages(conv_id, role, content, refs) VALUES(?,?,?,?)",
        (conv_id, role, content, json.dumps(refs or [], ensure_ascii=False)),
    )
    conn.commit()


def rename_conversation_if_untitled(conv_id: int, first_question: str):
    conn = _db()
    row = conn.execute(
        "SELECT title FROM conversations WHERE id=?", (conv_id,)
    ).fetchone()
    if row and (row["title"] == "新对话" or not row["title"]):
        title = (first_question.strip() or "新对话")[:24]
        conn.execute("UPDATE conversations SET title=? WHERE id=?", (title, conv_id))
        conn.commit()
        return title
    return row["title"] if row else "新对话"
