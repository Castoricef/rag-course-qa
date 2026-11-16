"""向量化：默认走大模型 embedding 接口，没配 Key 时退化为离线哈希向量。

离线模式只是为了让你 clone 下来不配 Key 也能跑通链路（验证解析 / 切分 / 检索 / 排序），
检索质量不能当真——语义检索必须用真实 embedding。
"""

from __future__ import annotations

import hashlib
import os
import re

DIM = 256
_ASCII_RE = re.compile(r"[a-z0-9_]+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")


def _tokens(text: str) -> list[str]:
    """英文按词切，中文按字符二元组切。

    中文没有空格，整段当 token 会导致查询与文档零重叠（相似度恒为 0），
    用 bigram 才能在无分词库的前提下拿到可用的词面重合度。
    """
    text = text.lower()
    toks = _ASCII_RE.findall(text)
    for run in _CJK_RE.findall(text):
        if len(run) == 1:
            toks.append(run)
        else:
            toks.extend(run[i : i + 2] for i in range(len(run) - 1))
    return toks


class OfflineEmbedder:
    """词袋哈希投影（bigram），确定性强、无需网络。

    本质是词面匹配，不是语义检索：换个说法就召回不到。
    只用于 clone 下来不配 Key 也能把链路跑通，检索质量不能当真。
    """

    name = "offline-hash-256"
    offline = True

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * DIM
        for token in _tokens(text):
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest()[:8], 16)
            vec[h % DIM] += 1.0
        norm = sum(v * v for v in vec) ** 0.5
        return [v / norm for v in vec] if norm else vec


class OpenAIEmbedder:
    name = "openai"

    def __init__(self, model: str, api_key: str, base_url: str | None = None) -> None:
        from openai import OpenAI  # 延迟导入

        self.model = model
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]


def build_embedder(cfg: dict):
    """按配置返回 embedding 客户端；无 Key 或显式 offline 时用离线实现。"""
    if cfg.get("offline") or not os.getenv("OPENAI_API_KEY"):
        return OfflineEmbedder()
    return OpenAIEmbedder(
        model=cfg.get("embedding_model", "text-embedding-3-small"),
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.getenv("OPENAI_BASE_URL") or None,
    )
