"""文本切分。

策略：先按标题层级切成语义块，超长块再按句末标点二次切分，块间保留重叠，
避免一个知识点被切成两半导致检索到的上下文残缺。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

# 句末标点：按这些字符断句
SENT_END = "。！？；!?;"

# 标题识别：Markdown 标题，或「第一章 / 一、 / 3.2 xxx」这类编号行
_HEADING_RE = re.compile(
    r"^(?:#{1,4}\s+\S.*"
    r"|第[一二三四五六七八九十百零〇\d]+[章节部分]\s*\S{0,30}"
    r"|[\d一二三四五六七八九十]+(?:[、.．]|\s)\s*\S{1,30})$"
)


@dataclass
class Chunk:
    """一个入库单元。meta 里的字段会被带进 prompt 用于溯源。"""

    text: str
    source: str          # 文件名，如 第三章.pdf
    page: int | None     # 页码，PPT 为空
    section: str         # 所属小节标题
    chunk_id: str = ""

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "source": self.source,
            "page": self.page,
            "section": self.section,
            "chunk_id": self.chunk_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(**d)


def _split_by_heading(text: str) -> list[tuple[str, str]]:
    """按标题行切段，返回 [(小节标题, 正文)]。没有标题时整体作为一段。"""
    lines = text.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_title = ""
    buffer: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            buffer.append("")
            continue
        if _HEADING_RE.match(stripped) and len(stripped) <= 40:
            if buffer:
                sections.append((current_title, buffer))
            current_title = stripped.lstrip("#").strip()
            buffer = []
        else:
            buffer.append(stripped)

    if buffer:
        sections.append((current_title, buffer))

    return [(title, "\n".join(b).strip()) for title, b in sections if "\n".join(b).strip()]


def _split_long(body: str, max_chars: int, overlap: int) -> list[str]:
    """把超长正文切成不超过 max_chars 的块，相邻块共享末尾 overlap 个字符。"""
    if len(body) <= max_chars:
        return [body]

    sentences: list[str] = []
    buf = ""
    for ch in body:
        buf += ch
        if ch in SENT_END or ch == "\n":
            if buf.strip():
                sentences.append(buf.strip())
            buf = ""
    if buf.strip():
        sentences.append(buf.strip())

    pieces: list[str] = []
    current = ""
    for sent in sentences:
        if current and len(current) + len(sent) > max_chars:
            pieces.append(current)
            # 用上一块的尾部做重叠，保证跨块的语义不丢
            current = (current[-overlap:] if overlap > 0 else "") + sent
        else:
            current += sent
    if current.strip():
        pieces.append(current)

    return [p for p in pieces if p.strip()]


def split_document(
    text: str,
    source: str,
    page: int | None = None,
    max_chars: int = 500,
    overlap: int = 80,
) -> list[Chunk]:
    """把一份文档的正文切成一串 Chunk。"""
    chunks: list[Chunk] = []
    for title, body in _split_by_heading(text):
        for piece in _split_long(body, max_chars, overlap):
            chunks.append(Chunk(text=piece, source=source, page=page, section=title))

    # chunk_id 由 来源+页码+序号 生成，保证增量导入时同一块 id 稳定
    for i, c in enumerate(chunks):
        raw = f"{c.source}|{c.page}|{i}|{c.text[:40]}"
        c.chunk_id = hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]
    return chunks
