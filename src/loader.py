"""文档解析：按扩展名分发到对应解析器，统一输出纯文本 + 页码。

第三方解析库都做延迟导入，这样只跑切分 / 检索时不需要装全套依赖。
"""

from __future__ import annotations

from pathlib import Path

SUPPORTED = {".pdf", ".docx", ".pptx", ".md", ".txt"}


def load_document(path: str | Path) -> list[tuple[str, int | None]]:
    """返回 [(该页/该段的文本, 页码)]。非分页格式页码为 None。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    suffix = p.suffix.lower()
    if suffix not in SUPPORTED:
        raise ValueError(f"不支持的格式：{suffix}（支持 {sorted(SUPPORTED)}）")

    if suffix == ".pdf":
        return _load_pdf(p)
    if suffix == ".docx":
        return _load_docx(p)
    if suffix == ".pptx":
        return _load_pptx(p)
    return _load_plain(p)


def _load_pdf(p: Path) -> list[tuple[str, int | None]]:
    try:
        import pymupdf  # PyMuPDF
    except ImportError as e:
        raise ImportError("解析 PDF 需要 pymupdf：pip install pymupdf") from e

    pages: list[tuple[str, int | None]] = []
    with pymupdf.open(p) as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text().strip()
            if text:
                pages.append((text, i))
    return pages


def _load_docx(p: Path) -> list[tuple[str, int | None]]:
    try:
        import docx
    except ImportError as e:
        raise ImportError("解析 DOCX 需要 python-docx：pip install python-docx") from e

    document = docx.Document(str(p))
    parts: list[str] = []
    for para in document.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        # 保留标题层级信息，切分时才能按标题切
        style = (para.style.name or "").lower()
        if style.startswith("heading") and style[-1:].isdigit():
            level = int(style[-1])
            parts.append("#" * min(level, 4) + " " + text)
        else:
            parts.append(text)

    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))

    return [("\n".join(parts), None)] if parts else []


def _load_pptx(p: Path) -> list[tuple[str, int | None]]:
    try:
        from pptx import Presentation
    except ImportError as e:
        raise ImportError("解析 PPTX 需要 python-pptx：pip install python-pptx") from e

    prs = Presentation(str(p))
    slides: list[tuple[str, int | None]] = []
    for i, slide in enumerate(prs.slides, start=1):
        texts: list[str] = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    line = "".join(run.text for run in para.runs).strip()
                    if line:
                        texts.append(line)
        if texts:
            slides.append(("\n".join(texts), i))
    return slides


def _load_plain(p: Path) -> list[tuple[str, int | None]]:
    return [(p.read_text(encoding="utf-8", errors="ignore"), None)]


def iter_files(raw_dir: str | Path):
    """遍历资料目录，产出所有受支持的文件路径。"""
    root = Path(raw_dir)
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in SUPPORTED:
            yield p
