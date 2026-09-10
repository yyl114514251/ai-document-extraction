"""文档解析：把 pdf / md / txt / docx 转成带页码或标题的文本段列表。"""
from pathlib import Path

SUPPORTED_EXTS = {".pdf", ".md", ".markdown", ".txt", ".docx"}


class DocumentParseError(Exception):
    pass


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8", "gbk", "utf-16"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def parse_pdf(path: Path):
    import pdfplumber

    sections = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                sections.append((f"第{i}页", text))
    if not sections:
        raise DocumentParseError("PDF 未解析出任何文本（可能是扫描件/图片型 PDF）")
    return sections


def parse_md(path: Path):
    text = _read_text(path)
    lines = text.splitlines()
    sections = []
    current_title, current = "正文", []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            if current:
                sections.append((current_title, "\n".join(current).strip()))
            current_title = stripped.lstrip("#").strip() or "正文"
            current = []
        else:
            current.append(line)
    if current:
        sections.append((current_title, "\n".join(current).strip()))
    return [(t, c) for t, c in sections if c]


def parse_txt(path: Path):
    text = _read_text(path).strip()
    if not text:
        raise DocumentParseError("文件内容为空")
    return [("正文", text)]


def parse_docx(path: Path):
    from docx import Document

    doc = Document(path)
    parts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    if not parts:
        raise DocumentParseError("Word 文档未解析出任何文本")
    return [("正文", "\n".join(parts))]


_PARSERS = {
    ".pdf": parse_pdf,
    ".md": parse_md,
    ".markdown": parse_md,
    ".txt": parse_txt,
    ".docx": parse_docx,
}


def parse_document(path: Path):
    ext = path.suffix.lower()
    if ext not in _PARSERS:
        raise DocumentParseError(f"不支持的文件类型：{ext or '无扩展名'}")
    try:
        return _PARSERS[ext](path)
    except DocumentParseError:
        raise
    except Exception as e:  # noqa: BLE001
        raise DocumentParseError(f"解析失败：{e}") from e
