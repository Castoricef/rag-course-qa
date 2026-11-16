"""向量库：纯 Python 实现的极简余弦检索。

单机万级 chunk 用不着上 Milvus / FAISS，用 math 手算余弦就够了，
少一个依赖，也方便看清楚检索到底在做什么。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from src.splitter import Chunk


class VectorStore:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.vectors: list[list[float]] = []

    # ---------- 写入 ----------

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        """追加写入；chunk_id 已存在的会跳过（支持增量导入）。"""
        if len(chunks) != len(vectors):
            raise ValueError("chunks 与 vectors 数量不一致")

        existing = {c.chunk_id for c in self.chunks}
        added = 0
        for chunk, vec in zip(chunks, vectors):
            if chunk.chunk_id in existing:
                continue
            self.chunks.append(chunk)
            self.vectors.append(vec)
            existing.add(chunk.chunk_id)
            added += 1
        return added

    # ---------- 检索 ----------

    def search(self, query_vec: list[float], top_k: int = 3) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            return []
        scored = [
            (chunk, _cosine(query_vec, vec))
            for chunk, vec in zip(self.chunks, self.vectors)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    # ---------- 持久化 ----------

    def save(self, index_dir: str | Path) -> None:
        path = Path(index_dir)
        path.mkdir(parents=True, exist_ok=True)
        payload = {
            "chunks": [c.to_dict() for c in self.chunks],
            "vectors": self.vectors,
        }
        (path / "index.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, index_dir: str | Path) -> "VectorStore":
        path = Path(index_dir) / "index.json"
        if not path.exists():
            raise FileNotFoundError(f"索引不存在：{path}，先跑 python ingest.py 建库")
        payload = json.loads(path.read_text(encoding="utf-8"))
        store = cls()
        store.chunks = [Chunk.from_dict(d) for d in payload["chunks"]]
        store.vectors = payload["vectors"]
        return store

    def __len__(self) -> int:
        return len(self.chunks)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
