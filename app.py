"""Gradio 界面（可选）。没装 gradio 时不影响命令行使用。

    pip install gradio
    python app.py
"""

from __future__ import annotations

import sys

from src import config
from src.qa import QA


def build():
    try:
        import gradio as gr
    except ImportError:
        sys.exit("需要先安装 gradio：pip install gradio")

    cfg = config.load()
    qa = QA(cfg)

    def answer(question: str, top_k: int):
        question = (question or "").strip()
        if not question:
            return "请输入问题", ""
        ans = qa.ask(question, top_k=int(top_k))
        sources = "\n".join(ans.citations) if ans.citations else "（无）"
        return ans.answer, sources

    with gr.Blocks(title="课程知识问答") as demo:
        gr.Markdown("## 课程知识问答\n答案来自本地知识库，附带原文出处。")
        with gr.Row():
            inp = gr.Textbox(label="问题", placeholder="例如：第三章讲了几种排序算法？", scale=4)
            k = gr.Slider(1, 8, value=cfg.get("top_k", 3), step=1, label="召回数量")
        btn = gr.Button("提问", variant="primary")
        out = gr.Textbox(label="回答", lines=8)
        src = gr.Textbox(label="出处", lines=4)
        btn.click(answer, [inp, k], [out, src])
        inp.submit(answer, [inp, k], [out, src])

    return demo


if __name__ == "__main__":
    build().launch()
