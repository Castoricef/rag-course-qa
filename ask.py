"""命令行问答。

用法：
    python ask.py "第三章讲了几种排序算法？"
    python ask.py                  # 进入交互模式
    python ask.py -k 5 "xxx"       # 临时指定召回数量
"""

from __future__ import annotations

import argparse
import sys

from src import config
from src.qa import QA


def render(ans) -> None:
    print("\n" + ans.answer.strip())
    if ans.citations and not ans.refused:
        print("\n出处：")
        for c in ans.citations:
            print("  " + c)
    if ans.hits:
        print(f"\n（最高相似度 {ans.best_score:.3f}）")


def main() -> None:
    parser = argparse.ArgumentParser(description="课程知识库问答")
    parser.add_argument("question", nargs="*", help="问题；不传则进入交互模式")
    parser.add_argument("-k", "--top-k", type=int, default=None, help="召回数量")
    parser.add_argument("--index", default="data/index", help="索引目录")
    args = parser.parse_args()

    cfg = config.load()
    try:
        qa = QA(cfg, index_dir=args.index)
    except FileNotFoundError as e:
        sys.exit(str(e))

    if len(qa.store) == 0:
        sys.exit("索引为空，先把课件放进 data/raw/ 并运行 python ingest.py")

    question = " ".join(args.question).strip()
    if question:
        render(qa.ask(question, top_k=args.top_k))
        return

    print("输入问题回车提问，输入 q 退出。")
    while True:
        try:
            line = input("\n你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if line.lower() in {"q", "quit", "exit"}:
            break
        if line:
            render(qa.ask(line, top_k=args.top_k))


if __name__ == "__main__":
    main()
