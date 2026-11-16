"""建库：解析 data/raw/ 下所有资料 → 切分 → 向量化 → 写入 data/index/。

用法：
    python ingest.py                 # 增量导入（跳过已入库的块）
    python ingest.py --reset         # 清空重建
    python ingest.py --src my_docs   # 指定资料目录
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from src import config, loader
from src.embedder import build_embedder
from src.splitter import Chunk, split_document
from src.store import VectorStore

BATCH = 32


def build_index(
    src_dir: str | Path,
    index_dir: str | Path,
    cfg: dict,
    reset: bool = False,
    verbose: bool = True,
) -> tuple[int, int]:
    """返回 (文件数, 新增块数)。"""
    src_path = Path(src_dir)
    index_path = Path(index_dir)

    if reset and index_path.exists():
        shutil.rmtree(index_path)
    index_path.mkdir(parents=True, exist_ok=True)

    store = VectorStore.load(index_path) if (index_path / "index.json").exists() else VectorStore()
    embedder = build_embedder(cfg)

    files = list(loader.iter_files(src_path))
    if not files:
        raise SystemExit(f"{src_path} 下没有找到可解析的文件（支持 .pdf/.docx/.pptx/.md/.txt）")

    total_added = 0
    for f in files:
        pages = loader.load_document(f)
        chunks: list[Chunk] = []
        for text, page in pages:
            chunks.extend(
                split_document(
                    text,
                    source=f.name,
                    page=page,
                    max_chars=int(cfg["max_chars"]),
                    overlap=int(cfg["overlap"]),
                )
            )
        if not chunks:
            if verbose:
                print(f"  跳过（未抽到文本）：{f.name}")
            continue

        vectors: list[list[float]] = []
        for i in range(0, len(chunks), BATCH):
            vectors.extend(embedder.embed([c.text for c in chunks[i : i + BATCH]]))

        added = store.add(chunks, vectors)
        total_added += added
        if verbose:
            print(f"  {f.name}: 解析 {len(pages)} 段 → 切分 {len(chunks)} 块 → 新增 {added}")

    store.save(index_path)
    return len(files), total_added


def main() -> None:
    parser = argparse.ArgumentParser(description="建立课程知识库索引")
    parser.add_argument("--src", default="data/raw", help="资料目录")
    parser.add_argument("--index", default="data/index", help="索引输出目录")
    parser.add_argument("--reset", action="store_true", help="清空已有索引后重建")
    args = parser.parse_args()

    cfg = config.load()
    if cfg.get("offline"):
        print("[offline] 使用离线哈希向量，仅用于验证链路，检索质量不代表真实效果")

    print(f"建库中：{args.src} → {args.index}")
    n_files, n_chunks = build_index(args.src, args.index, cfg, reset=args.reset)
    print(f"完成：{n_files} 个文件，新增 {n_chunks} 个 chunk")


if __name__ == "__main__":
    main()
