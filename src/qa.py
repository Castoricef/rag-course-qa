"""检索 + 生成 + 引用拼装。

两个关键设计：
1. 出处由 chunk 的元数据带出来，不让模型自己写页码；
2. 最高相似度低于阈值时直接拒答，避免模型拿知识库外的常识硬编。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from src.embedder import build_embedder
from src.splitter import Chunk
from src.store import VectorStore

SYSTEM_PROMPT = (
    "你是课程资料问答助手。只依据【参考资料】回答，不要引入资料以外的知识。\n"
    "回答时在对应句子后用 [编号] 标注出处，编号与参考资料一致。\n"
    "如果参考资料不足以回答，直接说「资料中未提及」，不要推测。"
)

REFUSAL = "知识库中未收录与该问题相关的内容。可以换个问法，或先把对应课件加入 data/raw/ 重新建库。"


@dataclass
class Answer:
    question: str
    answer: str
    citations: list[str] = field(default_factory=list)
    hits: list[tuple[Chunk, float]] = field(default_factory=list)
    best_score: float = 0.0
    refused: bool = False


class QA:
    def __init__(self, cfg: dict, index_dir: str = "data/index") -> None:
        self.cfg = cfg
        self.store = VectorStore.load(index_dir)
        self.embedder = build_embedder(cfg)
        self.top_k = int(cfg.get("top_k", 3))
        self.threshold = float(cfg.get("min_score", 0.35))
        # 离线哈希向量是词面匹配，相似度整体偏低，阈值要单独放宽
        if getattr(self.embedder, "offline", False):
            self.threshold = min(self.threshold, float(cfg.get("min_score_offline", 0.2)))
        self._client = None

    # ---------- 大模型 ----------

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError(
                    "未配置 OPENAI_API_KEY。离线模式下只能验证检索，生成回答需要真实 Key。"
                )
            self._client = OpenAI(
                api_key=os.environ["OPENAI_API_KEY"],
                base_url=os.getenv("OPENAI_BASE_URL") or None,
            )
        return self._client

    # ---------- 主流程 ----------

    def ask(self, question: str, top_k: int | None = None) -> Answer:
        k = top_k or self.top_k
        query_vec = self.embedder.embed([question])[0]
        hits = self.store.search(query_vec, top_k=k)

        if not hits:
            return Answer(question, REFUSAL, refused=True)

        best = hits[0][1]
        if best < self.threshold:
            return Answer(question, REFUSAL, hits=hits, best_score=best, refused=True)

        context, citations = _build_context(hits)
        if getattr(self.embedder, "offline", False):
            # 离线模式没有可用的大模型，直接返回检索到的原文，保证链路可跑通
            answer_text = _extractive_answer(hits[0][0])
        else:
            answer_text = self._generate(question, context)
        return Answer(question, answer_text, citations, hits, best)

    def _generate(self, question: str, context: str) -> str:
        user_msg = f"【参考资料】\n{context}\n\n【问题】\n{question}"
        resp = self.client.chat.completions.create(
            model=self.cfg.get("chat_model", "gpt-4o-mini"),
            temperature=float(self.cfg.get("temperature", 0.2)),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
        return resp.choices[0].message.content or ""


def _extractive_answer(chunk: Chunk) -> str:
    where = f"{chunk.source}" + (f" 第{chunk.page}页" if chunk.page else "")
    return (
        "【离线模式】未配置大模型 API，下面直接返回检索命中的原文片段，未做生成：\n\n"
        f"{chunk.text.strip()}\n\n（来源：{where} · {chunk.section}）"
    )


def _build_context(hits: list[tuple[Chunk, float]]) -> tuple[str, list[str]]:
    """把命中的 chunk 拼成带编号的参考资料，同时返回人类可读的出处列表。"""
    blocks: list[str] = []
    citations: list[str] = []
    for i, (chunk, _score) in enumerate(hits, start=1):
        where = f"{chunk.source}" + (f" 第{chunk.page}页" if chunk.page else "")
        if chunk.section:
            where += f" · {chunk.section}"
        blocks.append(f"[{i}] （{where}）\n{chunk.text}")
        citations.append(f"[{i}] {where}")
    return "\n\n".join(blocks), citations
