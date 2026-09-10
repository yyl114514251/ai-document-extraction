"""文本切片：按段落 + 字符窗口（支持中英文），保留来源元数据。"""
from __future__ import annotations

CHUNK_SIZE = 500
OVERLAP = 80
MIN_CHUNK = 30


def _split_long(text: str, size: int = CHUNK_SIZE, overlap: int = OVERLAP):
    if len(text) <= size:
        return [text.strip()] if text.strip() else []
    out = []
    start = 0
    n = len(text)
    while start < n:
        piece = text[start : start + size]
        piece = piece.strip()
        if piece:
            out.append(piece)
        if start + size >= n:
            break
        start += size - overlap
    return out


def chunk_text(text: str, max_size: int = CHUNK_SIZE, overlap: int = OVERLAP):
    """先按空行分段，再对超长段做滑动窗口。"""
    chunks = []
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [p.strip() for p in text.splitlines() if p.strip()]
    if not paragraphs:
        return []
    for para in paragraphs:
        if len(para) <= max_size:
            if len(para) >= MIN_CHUNK or len(paragraphs) == 1:
                chunks.append(para)
            continue
        chunks.extend(_split_long(para, max_size, overlap))
    # 合并过短的碎片（避免信息碎片化）
    merged: list[str] = []
    for c in chunks:
        if merged and len(c) < MIN_CHUNK and len(merged[-1]) + len(c) < max_size:
            merged[-1] = merged[-1] + "\n" + c
        else:
            merged.append(c)
    return merged


def chunk_document(sections, doc_id: int, source: str):
    """sections: [(标题/页码, 文本)] -> 返回带元数据的 chunk 记录。"""
    records = []
    seq = 0
    for heading, text in sections:
        for piece in chunk_text(text):
            records.append(
                {
                    "id": f"{doc_id}-{seq}",
                    "doc_id": str(doc_id),
                    "source": source,
                    "heading": heading,
                    "seq": seq,
                    "text": piece,
                    "char_count": len(piece),
                }
            )
            seq += 1
    return records
