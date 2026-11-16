"""参数对比实验：同一批问题，跑不同 (max_chars, overlap, top_k)，输出对比表。

    python scripts/compare_params.py
    python scripts/compare_params.py --questions eval/questions.jsonl

命中率 = 参考答案里的关键词全部出现在回答中的比例；
引用正确率 = 回答命中的关键词，出现在被引用的 chunk 里的比例（粗粒度近似）。
离线模式（无 API Key）下只统计检索侧指标：命中率按「命中块是否含关键词」计算。
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingest import build_index  # noqa: E402
from src import config  # noqa: E402
from src.qa import QA  # noqa: E402

GRID = [
    {"max_chars": 300, "overlap": 30, "top_k": 3},
    {"max_chars": 500, "overlap": 50, "top_k": 3},
    {"max_chars": 500, "overlap": 50, "top_k": 5},
    {"max_chars": 800, "overlap": 80, "top_k": 3},
]


def load_questions(path: str) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def keywords_hit(text: str, keywords: list[str]) -> bool:
    return all(k in text for k in keywords)


def run_one(cfg: dict, questions: list[dict], src_dir: str) -> tuple[float, float]:
    tmp = tempfile.mkdtemp(prefix="ragcmp-")
    build_index(src_dir, tmp, cfg, reset=True, verbose=False)
    qa = QA(cfg, index_dir=tmp)

    hits = ref_ok = 0
    for q in questions:
        ans = qa.ask(q["question"])
        if keywords_hit(ans.answer, q["keywords"]):
            hits += 1
            cited = "\n".join(c.text for c, _ in ans.hits)
            if keywords_hit(cited, q["keywords"]):
                ref_ok += 1

    n = len(questions) or 1
    return hits / n, ref_ok / n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="eval/questions.jsonl")
    parser.add_argument("--src", default="data/raw")
    args = parser.parse_args()

    base = config.load()
    questions = load_questions(args.questions)
    print(f"评测集：{len(questions)} 题　资料目录：{args.src}\n")
    print(f"{'max_chars':>10}{'overlap':>9}{'top_k':>7}{'命中率':>10}{'引用正确率':>12}")
    print("-" * 48)

    for g in GRID:
        cfg = {**base, **g}
        hit, ref = run_one(cfg, questions, args.src)
        print(f"{g['max_chars']:>10}{g['overlap']:>9}{g['top_k']:>7}{hit:>9.0%}{ref:>11.0%}")


if __name__ == "__main__":
    main()
