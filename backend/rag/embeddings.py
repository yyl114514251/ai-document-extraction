"""Embedding 封装：懒加载本地模型，批量编码。"""
from __future__ import annotations

import os
from functools import lru_cache

# 中国大陆访问 huggingface.co 不稳定，默认走国内镜像下载模型
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from .config import EMBED_MODEL

BATCH = 32


@lru_cache(maxsize=1)
def get_embedder():
    from sentence_transformers import SentenceTransformer

    print(f"[rag] 正在加载嵌入模型 {EMBED_MODEL}（首次运行会自动下载，约 100MB）...")
    model = SentenceTransformer(EMBED_MODEL)
    print("[rag] 嵌入模型加载完成")
    return model


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    model = get_embedder()
    vectors: list[list[float]] = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i : i + BATCH]
        vecs = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        vectors.extend(v.tolist() for v in vecs)
    return vectors


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
